use std::{collections::HashMap, fs, path::PathBuf, sync::Arc};

use anyhow::{anyhow, bail, Result};
use indicatif::ProgressStyle;
use opencv::core::{MatExprTraitConst, MatTrait, MatTraitConst, ModifyInplace};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
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
                "Read image {} complete, {} channels available.",
                file_name,
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

            tracing::info!("Image {} blur complete.", file_name);

            let original_size = img.size()?;

            let new_size = opencv::core::Size::new(
                (f64::from(original_size.width) * f64::from(ratio)).round() as i32,
                (f64::from(original_size.height) * f64::from(ratio)).round() as i32,
            );

            unsafe {
                img.modify_inplace(|src, dst| -> Result<()> {
                    opencv::imgproc::resize(src, dst, new_size, 0., 0., opencv::imgproc::INTER_AREA)
                        .map_err(|e| anyhow!("Failed to apply resize to image, {}", e))
                })?;
            }

            tracing::info!("Image {} resize complete.", file_name);

            let output_path = output_dir.join(file_name);
            let output_path = output_path
                .to_str()
                .ok_or(anyhow!("Failed to generate output path string"))?;

            tracing::info!(
                "Writing type={}, channels={}, size={:?} to {}",
                img.typ(),
                img.channels(),
                img.size()?,
                output_path
            );

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

pub async fn resize_labels(src_path: &str, ratio: f32) -> Result<()> {
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

            let task_span = tracing::info_span!(parent: &parent_span, "task_span");
            let _task_span = task_span.enter();
            task_span.pb_set_style(&ProgressStyle::with_template(
                "{spinner} Processing {msg} \n{wide_bar} {pos}/{len}",
            )?);
            task_span.pb_set_message(file_name);

            let img = opencv::imgcodecs::imread(input_path, opencv::imgcodecs::IMREAD_COLOR)?;
            if img.typ() != opencv::core::CV_8UC3 {
                bail!(
                    "Label must be CV_8UC3 (8-bit 3-channel BGR), currently {}",
                    img.typ()
                );
            }

            tracing::info!("Read image {} complete", file_name);

            let window_size = (1.0 / ratio).round() as usize;

            let original_size = img.size()?;

            let new_size = opencv::core::Size::new(
                (f64::from(original_size.width) * f64::from(ratio)).round() as i32,
                (f64::from(original_size.height) * f64::from(ratio)).round() as i32,
            );
            let out_h = new_size.height as usize;
            let out_w = new_size.width as usize;

            task_span.pb_set_length(out_h as u64);

            let rows: Result<Vec<Vec<opencv::core::Vec3b>>> = (0..out_h)
                .into_par_iter()
                .map(|oy| -> Result<Vec<opencv::core::Vec3b>> {
                    let iy0 = oy * window_size;

                    let mut out_row: Vec<opencv::core::Vec3b> = Vec::with_capacity(out_w);

                    for ox in 0..out_w {
                        let ix0 = ox * window_size;

                        let mut counts: HashMap<[u8; 3], u16> =
                            HashMap::with_capacity(window_size * window_size);
                        let mut best_color = opencv::core::Vec3b::from([0, 0, 0]);
                        let mut best_count = 0u16;

                        for dy in 0..window_size {
                            for dx in 0..window_size {
                                let y = (iy0 + dy) as i32;
                                let x = (ix0 + dx) as i32;

                                if y >= original_size.height {
                                    continue;
                                }

                                if x >= original_size.width {
                                    continue;
                                }

                                // Safe read
                                let pix = *img.at_2d::<opencv::core::Vec3b>(y, x)?;

                                let e = counts.entry(pix.0).or_insert(0);
                                *e += 1;

                                if *e > best_count {
                                    best_count = *e;
                                    best_color = pix;
                                }
                            }
                        }

                        out_row.push(best_color);
                    }

                    task_span.pb_inc(1);

                    Ok(out_row)
                })
                .collect();
            let rows = rows?;
            let mut output =
                opencv::core::Mat::zeros(out_h as i32, out_w as i32, opencv::core::CV_8UC3)?
                    .to_mat()?;

            task_span.pb_set_style(&ProgressStyle::with_template(
                "{spinner} Merging {msg} \n{wide_bar} {pos}/{len}",
            )?);
            task_span.pb_set_position(0);
            task_span.pb_set_length(rows.len() as u64);
            for (oy, row) in rows.into_iter().enumerate() {
                for (ox, pix) in row.into_iter().enumerate() {
                    *output.at_2d_mut::<opencv::core::Vec3b>(oy as i32, ox as i32)? = pix;
                }
                task_span.pb_inc(1);
            }
            let output_path = output_dir.join(file_name);
            let output_path = output_path
                .to_str()
                .ok_or(anyhow!("Failed to generate output path string"))?;

            tracing::info!(
                "Writing type={}, channels={}, size={:?} to {}",
                output.typ(),
                output.channels(),
                output.size()?,
                output_path
            );

            opencv::imgcodecs::imwrite(output_path, &output, &opencv::core::Vector::new())?;

            parent_span.pb_inc(1);

            Ok(())
        });
    }

    while let Some(result) = threads.join_next().await {
        result??;
    }

    Ok(())
}
