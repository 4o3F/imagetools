use anyhow::{anyhow, bail, Context, Result};
use indicatif::ProgressStyle;
use opencv::{
    core::{
        Mat, MatTrait, MatTraitConst, MatTraitConstManual, MatTraitManual, ModifyInplace, Vec3b,
        Vector, CV_8UC3,
    },
    imgcodecs::{self, imread, imwrite},
    imgproc::{self},
};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

use std::{collections::HashMap, fs, path::PathBuf, sync::Arc};
use tokio::{sync::Semaphore, task::JoinSet};
use tracing::{info_span, Span};
use tracing_indicatif::span_ext::IndicatifSpanExt;

use crate::THREAD_POOL;

pub async fn remap_color(
    original_color: &str,
    new_color: &str,
    dataset_path: &String,
) -> Result<()> {
    let mut original_color_vec: Vec<u8> = vec![];
    for splited in original_color.split(',') {
        let splited = splited
            .parse::<u8>()
            .context("Parsing original color RGB")?;
        original_color_vec.push(splited);
    }

    let mut new_color_vec: Vec<u8> = vec![];
    for splited in new_color.split(',') {
        let splited = splited.parse::<u8>().context("Parsing target color RGB")?;
        new_color_vec.push(splited);
    }

    if original_color_vec.len() != 3 || new_color_vec.len() != 3 {
        bail!("Malformed color RGB, please use R,G,B format");
    }

    let mut entries: Vec<PathBuf> = Vec::new();
    let dataset_path = PathBuf::from(dataset_path);
    if dataset_path.is_file() {
        entries.push(dataset_path.clone());
        fs::create_dir_all(
            dataset_path
                .parent()
                .ok_or(anyhow!("Failed to get dataset path parent dir"))?
                .join("output"),
        )?;
    } else {
        entries = fs::read_dir(&dataset_path)
            .context(format!("Failed to read dir {:?}", dataset_path))?
            .map(|x| x.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()?;
        fs::create_dir_all(dataset_path.join("output"))?;
    }

    let mut threads = JoinSet::new();
    let semaphore = Arc::new(Semaphore::new(
        (*THREAD_POOL
            .read()
            .map_err(|_| anyhow!("THREAD_POOL lock poisoned"))?)
        .into(),
    ));

    let header_span = info_span!("remap_color_threads");
    header_span.pb_set_style(
        &ProgressStyle::with_template("{spinner} Processing {msg}\n{wide_bar} {pos}/{len}")
            .map_err(|e| anyhow!("Tracing progress template generate failed {e}"))?,
    );
    header_span.pb_set_length(entries.len() as u64);
    header_span.pb_set_message("starting...");

    let header_span_enter = header_span.enter();

    for entry in entries {
        let entry = entry.clone();
        let sem = semaphore.clone();
        let original_color = original_color_vec.clone();
        let new_color = new_color_vec.clone();

        let header_span = header_span.clone();

        let permit = sem
            .clone()
            .acquire_owned()
            .await
            .context("Semaphore closed")?;

        threads.spawn_blocking(move || -> Result<()> {
            let _permit = permit;
            let file_name = entry
                .file_name()
                .ok_or(anyhow!("Failed to get file name"))?
                .to_string_lossy()
                .into_owned();

            let mut img = imread(
                entry.to_str().ok_or(anyhow!("Failed to get entry path"))?,
                imgcodecs::IMREAD_COLOR,
            )?;
            if img.channels() != 3 {
                bail!("Image {} is not RGB image", file_name);
            }

            let rows = img.rows();
            let cols = img.cols();

            for row_index in 0..rows {
                let mut row = img.row_mut(row_index)?;
                for col_index in 0..cols {
                    let pixel = row.at_2d_mut::<Vec3b>(0, col_index)?;
                    if pixel[0] == original_color[0]
                        && pixel[1] == original_color[1]
                        && pixel[2] == original_color[2]
                    {
                        pixel[0] = new_color[0];
                        pixel[1] = new_color[1];
                        pixel[2] = new_color[2];
                    }
                }
            }

            imwrite(
                entry
                    .parent()
                    .ok_or(anyhow!("Failed to get {:?} parent dir", entry))?
                    .join("output")
                    .join(&file_name)
                    .to_str()
                    .ok_or(anyhow!("Failed to convert path to string"))?,
                &img,
                &Vector::new(),
            )?;

            tracing::info!("{} finished", file_name);
            header_span.pb_inc(1);
            Ok(())
        });
    }

    while let Some(res) = threads.join_next().await {
        if let Err(err) = res {
            bail!(err)
        }
    }
    drop(header_span_enter);
    tracing::info!("All done!");
    Ok(())
}

pub async fn remap_background_color(
    valid_colors: &str,
    new_color: &str,
    dataset_path: &String,
) -> Result<()> {
    let mut valid_color_vec: Vec<Vec<u8>> = vec![];
    for valid_color in valid_colors.split(";") {
        let mut color_vec = vec![];
        for splited in valid_color.split(',') {
            let splited = splited
                .parse::<u8>()
                .context("Malformed original color RGB, please use R,G,B format")?;
            color_vec.push(splited);
        }

        valid_color_vec.push(color_vec);
    }

    let mut new_color_vec: Vec<u8> = vec![];
    for splited in new_color.split(',') {
        let splited = splited
            .parse::<u8>()
            .context("Malformed new color RGB, please use R,G,B format")?;
        new_color_vec.push(splited);
    }

    if valid_color_vec.iter().any(|x| x.len() != 3) || new_color_vec.len() != 3 {
        bail!("Malformed color RGB, please use R,G,B format");
    }

    let mut entries: Vec<PathBuf> = Vec::new();
    let dataset_path = PathBuf::from(dataset_path);
    if dataset_path.is_file() {
        entries.push(dataset_path.clone());
        fs::create_dir_all(
            dataset_path
                .parent()
                .ok_or(anyhow!("Failed to get dataset path parent dir"))?
                .join("output"),
        )?;
    } else {
        entries = fs::read_dir(dataset_path.clone())?
            .map(|x| x.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()?;
        fs::create_dir_all(dataset_path.join("output"))?;
    }

    let mut threads = JoinSet::new();
    let semaphore = Arc::new(Semaphore::new(
        (*THREAD_POOL
            .read()
            .map_err(|_| anyhow!("THREAD_POOL lock poisoned"))?)
        .into(),
    ));

    let header_span = info_span!("remap_background_color_threads");
    header_span.pb_set_style(
        &ProgressStyle::with_template("{spinner} Processing {msg}\n{wide_bar} {pos}/{len}")
            .map_err(|e| anyhow!("Tracing progress template generate failed {e}"))?,
    );
    header_span.pb_set_length(entries.len() as u64);
    header_span.pb_set_message("starting...");

    let header_span_enter = header_span.enter();

    for entry in entries {
        let entry = entry.clone();
        let valid_colors = valid_color_vec.clone();
        let new_color = new_color_vec.clone();

        let permit = semaphore.clone().acquire_owned().await?;

        threads.spawn_blocking(move || -> Result<()> {
            let _permit = permit;
            let file_name = entry
                .file_name()
                .ok_or(anyhow!("Failed to get file name"))?
                .to_string_lossy()
                .into_owned();
            let mut img = imread(
                entry.to_str().ok_or(anyhow!("Failed to get entry path"))?,
                imgcodecs::IMREAD_COLOR,
            )?;
            if img.channels() != 3 {
                bail!("Image {} is not RGB", file_name);
            }

            let rows = img.rows();
            let cols = img.cols();

            for row_index in 0..rows {
                let mut row = img.row_mut(row_index)?;
                for col_index in 0..cols {
                    let pixel = row.at_2d_mut::<Vec3b>(0, col_index)?;

                    if !valid_colors.contains(&vec![pixel[0], pixel[1], pixel[2]]) {
                        pixel[0] = new_color[0];
                        pixel[1] = new_color[1];
                        pixel[2] = new_color[2];
                    }
                }
            }
            imwrite(
                entry
                    .parent()
                    .ok_or(anyhow!("Failed to get {:?} parent dir", entry))?
                    .join("output")
                    .join(&file_name)
                    .to_str()
                    .ok_or(anyhow!("Failed to convert path to string"))?,
                &img,
                &Vector::new(),
            )?;

            tracing::info!("{} finished", file_name);
            Ok(())
        });
    }

    while let Some(res) = threads.join_next().await {
        if let Err(err) = res {
            bail!("{:?}", err);
        }
    }
    drop(header_span_enter);
    tracing::info!("All done!");
    Ok(())
}

pub async fn class2rgb(dataset_path: &str, rgb_list: &str) -> Result<()> {
    let mut entries: Vec<PathBuf> = Vec::new();
    let dataset_path = PathBuf::from(dataset_path);
    if dataset_path.is_file() {
        entries.push(dataset_path.clone());
        fs::create_dir_all(
            dataset_path
                .parent()
                .ok_or(anyhow!("Failed to get dataset path parent dir"))?
                .join("output"),
        )?;
    } else {
        entries = fs::read_dir(dataset_path.clone())?
            .map(|x| x.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()?;
        fs::create_dir_all(dataset_path.join("output"))?;
    }

    let mut transform_map = HashMap::<u8, Vec3b>::new();
    {
        // Split RGB list
        for (class_id, rgb) in rgb_list.split(";").enumerate() {
            let mut rgb_vec: Vec<u8> = vec![];
            for splited in rgb.split(',') {
                let splited = splited
                    .parse::<u8>()
                    .context("Malformed original color RGB, please use R,G,B format")?;
                rgb_vec.push(splited);
            }

            transform_map.insert(
                class_id as u8,
                Vec3b::from_array([rgb_vec[0], rgb_vec[1], rgb_vec[2]]),
            );
        }
    }

    let mut transform_platte = vec![
        Vec3b::default();
        *transform_map
            .keys()
            .max()
            .ok_or(anyhow!("Empty transform map"))? as usize
            + 1
    ];

    for (k, v) in transform_map {
        transform_platte[k as usize] = v;
    }

    let transform_platte = Arc::new(transform_platte);

    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(
        (*THREAD_POOL
            .read()
            .map_err(|_| anyhow!("THREAD_POOL lock poison"))?)
        .into(),
    ));

    let header_span = info_span!("class2rgb_threads");
    header_span.pb_set_style(&ProgressStyle::with_template(
        "{spinner} Processing {msg}\n{wide_bar} {pos}/{len}",
    )?);
    header_span.pb_set_length(entries.len() as u64);

    let header_span_enter = header_span.enter();

    tracing::info!("Process {} files", entries.len());

    for entry in entries {
        if !entry.is_file() {
            continue;
        }
        let header_span = header_span.clone();
        let transform_platte = Arc::clone(&transform_platte);

        let permit = sem.clone().acquire_owned().await?;

        threads.spawn_blocking(move || -> Result<()> {
            let _permit = permit;

            let file_name = entry
                .file_name()
                .ok_or(anyhow!("Failed to get file name"))?
                .to_string_lossy()
                .into_owned();

            let task_span = info_span!(parent: &header_span, "Image Processing");
            task_span.pb_set_style(
                &ProgressStyle::with_template("{spinner} Processing {msg}\n{wide_bar} {pos}/{len}")
                    .map_err(|e| anyhow!("Tracing progress template generate failed {e}"))?,
            );
            task_span.pb_set_message(&file_name);

            let _guard = task_span.enter();

            let img = imread(
                entry.to_str().ok_or(anyhow!("Failed to get entry path"))?,
                imgcodecs::IMREAD_GRAYSCALE,
            )?;

            if img.empty() {
                bail!("Empty image read");
            }
            let rows = img.rows();
            let cols = img.cols();

            let mut output = Mat::new_rows_cols_with_default(
                rows,
                cols,
                CV_8UC3,
                opencv::core::Scalar::all(0.0),
            )?;

            let src_slice: &[u8] = img.data_typed()?;
            let dst_slice: &mut [Vec3b] = output.data_typed_mut()?;

            let chunk = 1_000_000usize;
            task_span.pb_set_length((rows * cols) as u64 / (chunk as u64));

            dst_slice
                .par_chunks_mut(chunk)
                .zip(src_slice.par_chunks(chunk))
                .for_each(|(out_c, in_c)| {
                    for (o, &id) in out_c.iter_mut().zip(in_c.iter()) {
                        *o = transform_platte[id as usize];
                    }
                    task_span.pb_inc(1);
                });

            unsafe {
                output.modify_inplace(|input, output| {
                    opencv::imgproc::cvt_color(input, output, imgproc::COLOR_RGB2BGR, 0)
                        .context("Cvt RGB to BGR error")
                })?;
            }

            imwrite(
                entry
                    .parent()
                    .ok_or(anyhow!("Failed to get {:?} parent dir", entry))?
                    .join("output")
                    .join(&file_name)
                    .to_str()
                    .ok_or(anyhow!("Failed to convert path to string"))?,
                &output,
                &Vector::new(),
            )?;
            tracing::info!("{} finished", file_name);
            Span::current().pb_set_message(&file_name);
            header_span.pb_inc(1);
            Ok(())
        });
    }
    while let Some(result) = threads.join_next().await {
        result??;
    }
    std::mem::drop(header_span_enter);
    tracing::info!("All done");
    tracing::info!(
        "Saved to {}",
        dataset_path
            .to_str()
            .ok_or(anyhow!("Failed to convert dataset_path to string"))?
    );
    Ok(())
}

pub async fn rgb2class(dataset_path: &str, rgb_list: &str) -> Result<()> {
    let mut entries: Vec<PathBuf> = Vec::new();
    let dataset_path = PathBuf::from(dataset_path);
    if dataset_path.is_file() {
        entries.push(dataset_path.clone());
        fs::create_dir_all(
            dataset_path
                .parent()
                .ok_or(anyhow!("Failed to get dataset path parent dir"))?
                .join("output"),
        )?;
    } else {
        entries = fs::read_dir(dataset_path.clone())?
            .map(|x| x.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()?;
        fs::create_dir_all(dataset_path.join("output"))?;
    }

    let mut transform_map = HashMap::<[u8; 3], u8>::new();

    // Split RGB list
    for (class_id, rgb) in rgb_list.split(";").enumerate() {
        let mut rgb_vec: Vec<u8> = vec![];
        for splited in rgb.split(',') {
            let splited = splited
                .parse::<u8>()
                .context("Malformed color RGB, please use R,G,B format")?;
            rgb_vec.push(splited);
        }

        transform_map.insert([rgb_vec[0], rgb_vec[1], rgb_vec[2]], class_id as u8);
    }

    let transform_map = Arc::new(transform_map);

    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(
        (*THREAD_POOL
            .read()
            .map_err(|_| anyhow!("THREAD_POOL lock poison"))?)
        .into(),
    ));

    tracing::trace!("Processing {} images", entries.len());
    let header_span = info_span!("rgb2class_threads");
    header_span.pb_set_style(&ProgressStyle::with_template(
        "{spinner} Processing {msg}\n{wide_bar} {pos}/{len}",
    )?);
    header_span.pb_set_length(entries.len() as u64);
    header_span.pb_set_message("Processing");

    let header_span_enter = header_span.enter();

    for entry in entries {
        if !entry.is_file() {
            continue;
        }
        let permit = sem.clone().acquire_owned().await?;
        let transform_map = Arc::clone(&transform_map);
        let header_span = header_span.clone();
        threads.spawn_blocking(move || -> Result<()> {
            let _permit = permit;
            let file_name = entry
                .file_name()
                .ok_or(anyhow!("Failed to get file name"))?
                .to_string_lossy()
                .into_owned();
            let mut img = imread(
                entry.to_str().ok_or(anyhow!("Failed to get entry path"))?,
                imgcodecs::IMREAD_COLOR,
            )?;
            unsafe {
                img.modify_inplace(|input, output| {
                    opencv::imgproc::cvt_color(input, output, imgproc::COLOR_BGR2RGB, 0)
                        .context("Cvt color to RGB error")
                })?;
            }
            let transform_map = transform_map.clone();

            for (_, data) in img.iter_mut::<Vec3b>()? {
                if transform_map.contains_key(&[data[0], data[1], data[2]]) {
                    let new_color =
                        transform_map
                            .get(&[data[0], data[1], data[2]])
                            .ok_or(anyhow!(
                                "Unknown RGB color {} {} {}",
                                data[0],
                                data[1],
                                data[2]
                            ))?;
                    data[0] = *new_color;
                    data[1] = *new_color;
                    data[2] = *new_color;
                } else {
                    bail!(
                        "Image {} Color {}, {}, {} not found in RGB list",
                        file_name,
                        data[0],
                        data[1],
                        data[2]
                    )
                }
            }

            unsafe {
                img.modify_inplace(|input, output| {
                    imgproc::cvt_color(input, output, imgproc::COLOR_RGB2GRAY, 0)
                        .context("Cvt color to gray error")
                })?;
            }

            imwrite(
                entry
                    .parent()
                    .ok_or(anyhow!("Failed to get {:?} parent dir", entry))?
                    .join("output")
                    .join(&file_name)
                    .to_str()
                    .ok_or(anyhow!("Failed to convert path to string"))?,
                &img,
                &Vector::new(),
            )?;
            header_span.pb_inc(1);
            Ok(())
        });
    }
    while let Some(result) = threads.join_next().await {
        if let Err(err) = result {
            bail!("{:?}", err);
        }
    }

    std::mem::drop(header_span_enter);
    std::mem::drop(header_span);

    tracing::info!("All done");
    tracing::info!(
        "Saved to {}/output/",
        dataset_path
            .to_str()
            .ok_or(anyhow!("Failed to convert path to string"))?
    );
    Ok(())
}
