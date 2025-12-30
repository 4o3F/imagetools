use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemapEntry {
    pub id: String,

    pub image_old: String,
    pub image_new: String,

    pub label_old: String,
    pub label_new: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemapJson {
    pub version: u32,
    pub image_dir: String,
    pub label_dir: String,
    pub entries: Vec<RemapEntry>,
}

/// Mask (rename) an image/label dataset pair:
/// - Recursively reads all files under `image_dir` and `label_dir`
/// - Ensures every image has a corresponding label *by relative folder + stem*
///   (stem = filename without extension; label extension can differ)
/// - Renames both to the same numeric masked ID like "0001", "0002", ... (zero-padded)
/// - Preserves extensions and subfolder structure
/// - Writes a `remap.json` at `<image_dir>/remap.json` containing old->new mapping
///
/// Uses synchronous filesystem operations (std::fs).
pub async fn mask_dataset(image_dir: &String, label_dir: &String) -> Result<()> {
    // Function signature kept async per your original; body is sync FS.
    let image_root = PathBuf::from(&image_dir);
    let label_root = PathBuf::from(&label_dir);

    if !image_root.is_dir() {
        bail!("image_dir is not a directory: {}", image_root.display());
    }
    if !label_root.is_dir() {
        bail!("label_dir is not a directory: {}", label_root.display());
    }

    // Key: "<relative parent>/<stem>"
    let image_index = index_by_relparent_and_stem(&image_root)
        .with_context(|| format!("Indexing image_dir {}", image_root.display()))?;
    let label_index = index_by_relparent_and_stem(&label_root)
        .with_context(|| format!("Indexing label_dir {}", label_root.display()))?;

    // Ensure 1:1 correspondence
    let image_keys: BTreeSet<_> = image_index.keys().cloned().collect();
    let label_keys: BTreeSet<_> = label_index.keys().cloned().collect();

    if image_keys != label_keys {
        let only_in_images: Vec<_> = image_keys
            .difference(&label_keys)
            .take(50)
            .cloned()
            .collect();
        let only_in_labels: Vec<_> = label_keys
            .difference(&image_keys)
            .take(50)
            .cloned()
            .collect();

        let mut msg = String::new();
        if !only_in_images.is_empty() {
            msg.push_str("Missing in label_dir (present in image_dir):\n");
            for k in &only_in_images {
                msg.push_str(&format!("  - {k}\n"));
            }
        }
        if !only_in_labels.is_empty() {
            msg.push_str("Missing in image_dir (present in label_dir):\n");
            for k in &only_in_labels {
                msg.push_str(&format!("  - {k}\n"));
            }
        }
        bail!(
            "image_dir and label_dir are not correspondent (by relpath parent + stem).\n{}",
            msg
        );
    }

    // Choose padding width based on total count, minimum 4 (0001 style).
    let total = image_index.len();
    let width = std::cmp::max(3, digits_needed(total));

    // Build mapping entries (deterministic order via BTreeMap)
    let mut entries: Vec<RemapEntry> = Vec::with_capacity(total);

    for (i, (key, image_path)) in image_index.iter().enumerate() {
        let label_path = label_index
            .get(key)
            .ok_or_else(|| anyhow!("Internal error: key missing from label_index: {key}"))?;

        let id = format!("{:0width$}", i + 1, width = width);

        let image_ext = image_path
            .extension()
            .and_then(|s| s.to_str())
            .ok_or(anyhow!("Failed to get file extension"))?;
        let label_ext = label_path
            .extension()
            .and_then(|s| s.to_str())
            .ok_or(anyhow!("Failed to get file extension"))?;

        let image_parent = image_path
            .parent()
            .ok_or_else(|| anyhow!("No parent for image file {}", image_path.display()))?;
        let label_parent = label_path
            .parent()
            .ok_or_else(|| anyhow!("No parent for label file {}", label_path.display()))?;

        let image_new_name = format!("{id}.{image_ext}");
        let label_new_name = format!("{id}.{label_ext}");

        let image_new = image_parent.join(&image_new_name);
        let label_new = label_parent.join(&label_new_name);

        entries.push(RemapEntry {
            id,
            image_old: to_string_path(image_path),
            image_new: to_string_path(&image_new),
            label_old: to_string_path(label_path),
            label_new: to_string_path(&label_new),
        });
    }

    // 2-phase rename to avoid collisions:
    // Phase 1: old -> temp (same dir)
    // Phase 2: temp -> final
    //
    // Temp file names are deterministic per target name to avoid needing randomness.
    // (still unique enough: ".<target>.__mask_tmp__")
    let mut temp_moves: Vec<(PathBuf, PathBuf)> = Vec::with_capacity(entries.len() * 2);
    let mut final_moves: Vec<(PathBuf, PathBuf)> = Vec::with_capacity(entries.len() * 2);

    for e in &entries {
        let image_old = PathBuf::from(&e.image_old);
        let label_old = PathBuf::from(&e.label_old);
        let image_new = PathBuf::from(&e.image_new);
        let label_new = PathBuf::from(&e.label_new);

        // Refuse to overwrite existing files
        if image_new.exists() || label_new.exists() {
            bail!(
                "Target path already exists (refusing to overwrite):\n  {}\n  {}",
                image_new.display(),
                label_new.display()
            );
        }

        let image_tmp = temp_path_for_target(&image_old, &image_new)?;
        let label_tmp = temp_path_for_target(&label_old, &label_new)?;

        if image_tmp.exists() || label_tmp.exists() {
            bail!(
                "Temp path already exists (refusing to proceed):\n  {}\n  {}",
                image_tmp.display(),
                label_tmp.display()
            );
        }

        temp_moves.push((image_old, image_tmp.clone()));
        temp_moves.push((label_old, label_tmp.clone()));

        final_moves.push((image_tmp, image_new));
        final_moves.push((label_tmp, label_new));
    }

    for (from, to) in &temp_moves {
        fs::rename(from, to)
            .with_context(|| format!("Renaming {} -> {}", from.display(), to.display()))?;
    }
    for (from, to) in &final_moves {
        fs::rename(from, to)
            .with_context(|| format!("Renaming {} -> {}", from.display(), to.display()))?;
    }

    // Write remap.json at image_dir root
    let remap = RemapJson {
        version: 1,
        image_dir: canonical_or_lossy(&image_root),
        label_dir: canonical_or_lossy(&label_root),
        entries,
    };

    let remap_path = image_root.join("remap.json");
    let json = serde_json::to_string_pretty(&remap)?;
    fs::write(&remap_path, json).with_context(|| format!("Writing {}", remap_path.display()))?;

    Ok(())
}

/// Build an index of all files under `root` by:
/// key = "<relative parent>/<stem>"
fn index_by_relparent_and_stem(root: &Path) -> Result<BTreeMap<String, PathBuf>> {
    let mut out: BTreeMap<String, PathBuf> = BTreeMap::new();

    for entry in WalkDir::new(root).follow_links(false) {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path().to_path_buf();

        let rel = path
            .strip_prefix(root)
            .with_context(|| format!("strip_prefix failed for {}", path.display()))?;

        let parent = rel.parent().unwrap_or(Path::new(""));
        let stem = rel
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("Non-UTF8 filename stem: {}", rel.display()))?;

        let key = if parent.as_os_str().is_empty() {
            stem.to_string()
        } else {
            format!("{}/{}", parent.to_string_lossy(), stem)
        };

        if let Some(existing) = out.get(&key) {
            bail!(
                "Duplicate key '{}' under {}:\n  - {}\n  - {}",
                key,
                root.display(),
                existing.display(),
                path.display()
            );
        }

        out.insert(key, path);
    }

    Ok(out)
}

fn digits_needed(n: usize) -> usize {
    // digits in base-10 for n>=1
    let mut x = n.max(1);
    let mut d = 0;
    while x > 0 {
        d += 1;
        x /= 10;
    }
    d
}

fn temp_path_for_target(original: &Path, target: &Path) -> Result<PathBuf> {
    let parent = original
        .parent()
        .ok_or_else(|| anyhow!("No parent for {}", original.display()))?;
    let target_name = target
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("Non-UTF8 target filename: {}", target.display()))?;

    // Example final: "0001.jpg" => temp: ".0001.jpg.__mask_tmp__"
    Ok(parent.join(format!(".{target_name}.__mask_tmp__")))
}

fn to_string_path(p: &Path) -> String {
    p.to_string_lossy().to_string()
}

fn canonical_or_lossy(p: &Path) -> String {
    p.canonicalize()
        .map(|c| c.to_string_lossy().to_string())
        .unwrap_or_else(|_| p.to_string_lossy().to_string())
}
