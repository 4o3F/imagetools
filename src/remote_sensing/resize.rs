use std::{fs, path::PathBuf, sync::Arc};

use anyhow::{anyhow, bail, Result};
use indicatif::ProgressStyle;
use opencv::core::{MatTraitConst, ModifyInplace};
use tokio::{sync::Semaphore, task::JoinSet};
use tracing::info_span;
use tracing_indicatif::span_ext::IndicatifSpanExt;

use crate::THREAD_POOL;

pub async fn resize_images(src_path: &str, ratio: f32, blur_sigma: f32) -> Result<()> {
    if ratio >= 1.0 {
        bail!("Can only shrink image, currently super resolution is not supported")
    }

    let src_path = PathBuf::from(src_path);

    let mut src_entries = Vec::<PathBuf>::new();
    let output_dir = if src_path.is_file() {
        src_entries.push(src_path.clone());
        src_path
            .parent()
            .ok_or(anyhow!("Failed to read parent dir for src path"))?
            .join("rs_resize_output")
    } else {
        // Find all files inside
        for entry in fs::read_dir(&src_path)? {
            src_entries.push(entry?.path());
        }
        src_path.join("rs_resize_output")
    };
    fs::create_dir_all(&output_dir)?;

    let mut threads = JoinSet::new();

    let sem = Arc::new(Semaphore::new(
        (*THREAD_POOL
            .read()
            .map_err(|_| anyhow!("THREAD_POOL lock poisoned"))?)
        .into(),
    ));

    let parent_span = info_span!("remote_sensing_resize");
    parent_span.pb_set_style(&ProgressStyle::with_template(
        "{spinner} Processing \n{wide_bar} {pos}/{len}",
    )?);
    parent_span.pb_set_length(src_entries.len() as u64);

    let _parent_span_enter = parent_span.enter();

    for entry in src_entries {
        let permit = sem.clone().acquire_owned().await?;
        let output_dir = output_dir.clone();
        let parent_span = parent_span.clone();
        threads.spawn_blocking(move || -> Result<()> {
            let _permit = permit;

            let file_name = entry
                .file_name()
                .ok_or(anyhow!("Failed to get file name"))?
                .to_str()
                .ok_or(anyhow!("Failed to convert filename to str"))?;

            let input_path = entry
                .as_path()
                .to_str()
                .ok_or(anyhow!("Failed to convert input path to string"))?;
            let mut img =
                opencv::imgcodecs::imread(input_path, opencv::imgcodecs::IMREAD_UNCHANGED)?;

            tracing::info!(
                "Read image complete, {} channels available.",
                img.channels()
            );

            unsafe {
                img.modify_inplace(|src, dst| -> Result<()> {
                    opencv::imgproc::gaussian_blur(
                        src,
                        dst,
                        opencv::core::Size::new(0, 0),
                        blur_sigma.into(),
                        blur_sigma.into(),
                        opencv::core::BORDER_REFLECT_101,
                    )
                    .map_err(|e| anyhow!("Failed to apply gaussian blur to image, {}", e))
                })?;
            }

            let original_size = img.size()?;

            let new_size = opencv::core::Size::new(
                (f64::from(original_size.width) / f64::from(ratio)).round() as i32,
                (f64::from(original_size.height) / f64::from(ratio)).round() as i32,
            );

            unsafe {
                img.modify_inplace(|src, dst| -> Result<()> {
                    opencv::imgproc::resize(src, dst, new_size, 0., 0., opencv::imgproc::INTER_AREA)
                        .map_err(|e| anyhow!("Failed to apply resize to image, {}", e))
                })?;
            }

            let output_path = output_dir.join(file_name);
            let output_path = output_path
                .to_str()
                .ok_or(anyhow!("Failed to generate output path string"))?;

            opencv::imgcodecs::imwrite(output_path, &img, &opencv::core::Vector::new())?;

            parent_span.pb_inc(1);

            Ok(())
        });
    }

    while let Some(result) = threads.join_next().await {
        result??;
    }

    Ok(())
}
