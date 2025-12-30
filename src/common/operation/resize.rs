use anyhow::{bail, Context, Result};
use opencv::{
    core::{Mat, Rect, Size},
    imgcodecs, imgproc,
    prelude::*,
};
use rayon::prelude::*;

/// Resize a multi-channel TIFF image using the given filter.
/// Parallelized with rayon by splitting the destination image into chunks.
///
/// NOTE:
/// - Async wrapper uses spawn_blocking (OpenCV + Rayon are CPU-bound)
/// - `dataset_path` is assumed to be a path to a single image
pub async fn resize_images(
    src_path: &str,
    save_path: &str,
    target_height: &i32,
    target_width: &i32,
    filter: &str,
) -> Result<Mat> {
    let in_path = src_path.to_owned();
    let out_path = save_path.to_owned();
    let h = *target_height;
    let w = *target_width;
    let filter = filter.to_string();

    tokio::task::spawn_blocking(move || resize_images_sync(&in_path, &out_path, h, w, &filter))
        .await
        .context("spawn_blocking failed")?
}

/// Synchronous core implementation
fn resize_images_sync(
    in_path: &str,
    out_path: &str,
    target_height: i32,
    target_width: i32,
    filter: &str,
) -> Result<Mat> {
    if target_height <= 0 || target_width <= 0 {
        bail!("target dimensions must be positive");
    }

    let interpolation = match filter {
        "nearest" => imgproc::INTER_NEAREST,
        "linear" => imgproc::INTER_LINEAR,
        "cubic" => imgproc::INTER_CUBIC,
        "lancozs" => imgproc::INTER_LANCZOS4,
        _ => bail!("unsupported filter: {}", filter),
    };

    // Read TIFF (preserve channels + depth)
    let src = imgcodecs::imread(in_path, imgcodecs::IMREAD_UNCHANGED)
        .with_context(|| format!("failed to read image: {}", in_path))?;

    if src.empty() {
        bail!("image is empty: {}", in_path);
    }

    let src_rows = src.rows();
    let src_cols = src.cols();

    let src_type = src.typ();
    let mut dst = Mat::zeros(target_height, target_width, src_type)?.to_mat()?;

    // Number of parallel chunks
    let num_chunks = rayon::current_num_threads().max(1);
    let chunk_height = (target_height as usize).div_ceil(num_chunks);

    // Lanczos kernel radius (used for safe padding)
    let kernel_radius = if interpolation == imgproc::INTER_LANCZOS4 {
        4.0
    } else {
        1.0
    };

    let results: Vec<(i32, Mat)> = (0..num_chunks)
        .into_par_iter()
        .filter_map(|i| {
            let dy0 = (i * chunk_height) as i32;
            let dy1 = ((i + 1) * chunk_height).min(target_height as usize) as i32;
            if dy0 >= dy1 {
                return None;
            }

            let ratio_y = target_height as f64 / src_rows as f64;

            // Map destination rows → source rows
            let mut sy0 = ((dy0 as f64) / ratio_y - kernel_radius).floor() as i32;
            let mut sy1 = ((dy1 as f64) / ratio_y + kernel_radius).ceil() as i32;

            sy0 = sy0.clamp(0, src_rows);
            sy1 = sy1.clamp(0, src_rows);

            let src_roi = Rect::new(0, sy0, src_cols, sy1 - sy0);
            let src_view = Mat::roi(&src, src_roi).ok()?;

            let mut chunk = Mat::default();
            imgproc::resize(
                &src_view,
                &mut chunk,
                Size::new(target_width, dy1 - dy0),
                0.0,
                0.0,
                interpolation,
            )
            .ok()?;

            Some((dy0, chunk))
        })
        .collect();

    // Stitch chunks back together
    for (dy0, chunk) in results {
        let h = chunk.rows();
        let roi = Rect::new(0, dy0, target_width, h);
        let mut dst_view = Mat::roi_mut(&mut dst, roi)?;
        chunk.copy_to(&mut dst_view)?;
    }

    // Save output
    // Empty params = default encoder options; TIFF will preserve depth/channels.
    imgcodecs::imwrite(out_path, &dst, &opencv::core::Vector::new())
        .with_context(|| format!("failed to write image: {}", out_path))?;

    Ok(dst)
}
