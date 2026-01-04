use std::{
    collections::{HashMap, HashSet},
    fs,
    path::PathBuf,
    sync::Arc,
};

use anyhow::{anyhow, bail, Result};
use indicatif::ProgressStyle;
use opencv::core::{MatTraitConst, MatTraitConstManual, ModifyInplace};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};
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

#[inline]
fn bgr_to_code(bgr: &opencv::core::Vec3b) -> u32 {
    let b = bgr[0] as u32;
    let g = bgr[1] as u32;
    let r = bgr[2] as u32;
    (b << 16) | (g << 8) | r
}

#[inline]
fn code_to_bgr(code: u32) -> opencv::core::Vec3b {
    let b = ((code >> 16) & 255) as u8;
    let g = ((code >> 8) & 255) as u8;
    let r = (code & 255) as u8;
    opencv::core::Vec3b::from([b, g, r])
}

#[inline]
fn src_range_1d(d: i32, scale: f32, src_len: i32) -> (i32, i32) {
    let s0 = (d as f32) * scale;
    let s1 = ((d + 1) as f32) * scale;
    let mut a = s0.floor() as i32;
    let mut b = s1.ceil() as i32;
    a = a.clamp(0, src_len);
    b = b.clamp(0, src_len);
    if b < a {
        b = a;
    }
    (a, b)
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
            let img = opencv::imgcodecs::imread(input_path, opencv::imgcodecs::IMREAD_COLOR)?;

            if img.typ() != opencv::core::CV_8UC3 {
                bail!("Label must be CV_8UC3 (8-bit 3-channel BGR)");
            }

            tracing::info!("Read image {} complete", file_name);

            // unsafe {
            //     img.modify_inplace(|src, dst| -> Result<()> {
            //         opencv::imgproc::cvt_color(src, dst, opencv::imgproc::COLOR_BGR2RGB, 0)
            //             .map_err(|e| anyhow!("Failed to convert image BGR to RGB, {}", e))
            //     })?;
            // }

            // tracing::info!("Image {} BGR to RGB convert complete", file_name);

            if !img.is_continuous() {
                bail!("Mat is not continuous; cannot safely index via data_typed slice");
            }

            let img_data: &[opencv::core::Vec3b] = img.data_typed()?;
            let palette_set = img_data
                .par_chunks(16 * 1024)
                .map(|chunk| {
                    let mut local = HashSet::with_capacity(32);
                    for bgr in chunk {
                        local.insert(bgr_to_code(bgr));
                    }
                    local
                })
                .reduce(HashSet::new, |mut a, b| {
                    a.extend(b);
                    a
                });

            let palette: Vec<u32> = palette_set.into_iter().collect();

            let mut code2id: HashMap<u32, usize> = HashMap::with_capacity(palette.len() * 2);
            for (i, &c) in palette.iter().enumerate() {
                code2id.insert(c, i);
            }

            let original_size = img.size()?;

            let new_size = opencv::core::Size::new(
                (f64::from(original_size.width) * f64::from(ratio)).round() as i32,
                (f64::from(original_size.height) * f64::from(ratio)).round() as i32,
            );

            tracing::info!(
                "Label {} palette size = {}, src_size={:?}, dst_size={:?}",
                file_name,
                palette.len(),
                original_size,
                new_size
            );

            let src_w = original_size.width;
            let src_h = original_size.height;
            let out_w = new_size.width;
            let out_h = new_size.height;

            let scale_x = (src_w as f32) / (out_w as f32);
            let scale_y = (src_h as f32) / (out_h as f32);

            let x_ranges: Vec<(i32, i32)> = (0..out_w)
                .map(|dx| src_range_1d(dx, scale_x, src_w))
                .collect();
            let y_ranges: Vec<(i32, i32)> = (0..out_h)
                .map(|dy| src_range_1d(dy, scale_y, src_h))
                .collect();

            let k = palette.len();
            let mut out_buf =
                vec![opencv::core::Vec3b::default(); (out_w as usize) * (out_h as usize)];

            out_buf
                .par_chunks_mut(out_w as usize)
                .enumerate()
                .for_each(|(dy_u, row_out)| {
                    let (y0, y1) = y_ranges[dy_u];
                    let mut counts = vec![0u32; k];
                    let mut touched: Vec<usize> = Vec::with_capacity(256);

                    for dx in 0..out_w {
                        let (x0, x1) = x_ranges[dx as usize];
                        for &id in &touched {
                            counts[id] = 0;
                        }
                        touched.clear();
                        for sy in y0..y1 {
                            let row_base = (sy as usize) * (src_w as usize);
                            for sx in x0..x1 {
                                let idx = row_base + (sx as usize);
                                let p = &img_data[idx];
                                let code = bgr_to_code(p);
                                if let Some(&cid) = code2id.get(&code) {
                                    if counts[cid] == 0 {
                                        touched.push(cid);
                                    }
                                    counts[cid] += 1;
                                }
                            }
                        }

                        let mut best_id = 0usize;
                        let mut best_cnt = 0u32;
                        for &cid in &touched {
                            let c = counts[cid];
                            if c > best_cnt {
                                best_cnt = c;
                                best_id = cid;
                            }
                        }

                        row_out[dx as usize] = code_to_bgr(palette[best_id]);
                    }
                });

            let out = opencv::core::Mat::from_slice(&out_buf)?;
            let out = out.reshape(3, out_h)?;

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

            opencv::imgcodecs::imwrite(output_path, &out, &opencv::core::Vector::new())?;

            parent_span.pb_inc(1);

            Ok(())
        });
    }

    while let Some(result) = threads.join_next().await {
        result??;
    }

    Ok(())
}
