use std::{
    collections::HashSet,
    fs::{self},
    path::PathBuf,
    sync::{Arc, RwLock},
};

use anyhow::{anyhow, bail, Context, Result};
use indicatif::ProgressStyle;
use parking_lot::Mutex;
use rayon::prelude::*;
use rayon_progress::ProgressAdaptor;
use tokio::{sync::Semaphore, task::JoinSet};

use opencv::{boxed_ref::BoxedRef, core, imgcodecs, imgproc::COLOR_BGR2RGB, prelude::*};
use tracing::{info_span, Instrument, Span};
use tracing_indicatif::span_ext::IndicatifSpanExt;
use tracing_unwrap::{OptionExt, ResultExt};

use crate::THREAD_POOL;

pub async fn split_images(dataset_path: &String, target_height: &u32, target_width: &u32) {
    let mut entries: Vec<PathBuf> = Vec::new();
    let dataset_path = PathBuf::from(dataset_path);
    if dataset_path.is_file() {
        entries.push(dataset_path.clone());
        fs::create_dir_all(format!(
            "{}/output/",
            dataset_path.parent().unwrap().to_str().unwrap()
        ))
        .expect_or_log("Failed to create directory");
    } else {
        entries = fs::read_dir(dataset_path.clone())
            .unwrap()
            .map(|x| x.unwrap().path())
            .collect();

        fs::create_dir_all(format!("{}/output/", dataset_path.to_str().unwrap()))
            .expect_or_log("Failed to create directory");
    }
    let mut threads = JoinSet::new();

    for entry in entries {
        if entry.is_dir() {
            continue;
        }
        let target_height = *target_height;
        let target_width = *target_width;
        let entry_path = entry
            .to_str()
            .expect_or_log("Failed to convert path to string")
            .to_owned();

        threads.spawn(async move {
            let file_name = entry
                .file_name()
                .expect_or_log("Failed to get file name")
                .to_str()
                .expect_or_log("Failed to get file name")
                .to_owned();
            tracing::info!("Image {} processing...", file_name);

            let file_stem = entry
                .file_stem()
                .expect_or_log("Failed to get file stem")
                .to_str()
                .expect_or_log("Failed to convert file stem to string")
                .to_owned();

            let file_extension = entry
                .extension()
                .expect_or_log("Failed to get file extension")
                .to_str()
                .expect_or_log("Failed to convert file extension to string")
                .to_owned();

            let img = imgcodecs::imread(&entry_path, imgcodecs::IMREAD_UNCHANGED)
                .expect_or_log(format!("Failed to read image: {}", entry_path).as_str());
            if img.empty() {
                tracing::error!("Failed to read image: {} image is empty", entry_path);
                return ();
            }
            let size = img.size().expect_or_log("Failed to get image size");
            tracing::trace!("Image {} size {:?}", file_name, size);

            let (width, height) = (size.width as u32, size.height as u32);

            let y_count = height / target_height;
            let x_count = width / target_width;

            tracing::trace!(
                "Image {} vertical_count {} horizontal_count {}",
                file_name,
                y_count,
                x_count
            );

            // LTR
            'outer: for x_index in 0..x_count {
                'inner: for y_index in 0..y_count {
                    let start_x = x_index * target_width;
                    let start_y = y_index * target_height;

                    if start_x + target_width > width {
                        continue 'outer;
                    }
                    if start_y + target_height > height {
                        continue 'inner;
                    }

                    let cropped_img = core::Mat::roi(
                        &img,
                        core::Rect::new(
                            start_x as i32,
                            start_y as i32,
                            target_width as i32,
                            target_height as i32,
                        ),
                    )
                    .expect_or_log("Failed to crop image");
                    let path = format!(
                        "{}/output/{}_LTR_x{}_y{}.{}",
                        entry.parent().unwrap().to_str().unwrap(),
                        file_stem,
                        x_index,
                        y_index,
                        file_extension
                    );
                    let result =
                        imgcodecs::imwrite(path.as_str(), &cropped_img, &core::Vector::new())
                            .expect_or_log(
                                format!(
                                    "Failed to save image: {} opencv imwrite internal error",
                                    path
                                )
                                .as_str(),
                            );
                    if !result {
                        tracing::error!("Failed to save image {}", path);
                    }
                }
            }
            tracing::info!("Image LTR {} done", file_name);

            // RTL
            'outer: for x_index in 0..x_count {
                'inner: for y_index in 0..y_count {
                    let start_y = height as i32 - (y_index * target_height) as i32;
                    let start_x = width as i32 - (x_index * target_width) as i32;

                    if start_x < 0
                        || start_x as u32 >= width
                        || start_x as u32 + target_width > width
                    {
                        continue 'outer;
                    }
                    if start_y < 0
                        || start_y as u32 >= height
                        || start_y as u32 + target_height > height
                    {
                        continue 'inner;
                    }

                    let cropped_img = core::Mat::roi(
                        &img,
                        core::Rect::new(
                            start_x,
                            start_y,
                            target_width as i32,
                            target_height as i32,
                        ),
                    )
                    .expect_or_log("Failed to crop image");
                    let path = format!(
                        "{}/output/{}_RTL_x{}_y{}.{}",
                        entry.parent().unwrap().to_str().unwrap(),
                        file_stem,
                        x_index,
                        y_index,
                        file_extension
                    );
                    let result =
                        imgcodecs::imwrite(path.as_str(), &cropped_img, &core::Vector::new())
                            .expect_or_log(
                                format!(
                                    "Failed to save image: {} opencv imwrite internal error",
                                    path
                                )
                                .as_str(),
                            );
                    if !result {
                        tracing::error!("Failed to save image {}", path);
                    }
                }
            }
            tracing::info!("Image RTL {} done", file_name);

            tracing::info!("Image {} done", file_name);
        });
    }

    while let Some(result) = threads.join_next().await {
        match result {
            Ok(_) => {}
            Err(e) => {
                tracing::error!("{}", e);
            }
        }
    }

    tracing::info!("Image split done");
}

pub async fn split_images_with_bias(
    dataset_path: &String,
    bias_step: &u32,
    target_height: &u32,
    target_width: &u32,
) {
    let entries = fs::read_dir(dataset_path).unwrap();
    let mut threads = JoinSet::new();

    fs::create_dir_all(format!("{}/../output/", dataset_path)).unwrap();

    for entry in entries {
        let entry = entry.unwrap();
        if entry.path().is_dir() {
            continue;
        }
        let dataset_path = dataset_path.clone();
        let bias_step = *bias_step;
        let target_height = *target_height;
        let target_width = *target_width;
        let entry_path = entry.path().to_str().unwrap().to_string();

        threads.spawn(async move {
            tracing::info!(
                "Image {} processing...",
                entry.file_name().to_str().unwrap()
            );

            // 读取图片
            let img = imgcodecs::imread(&entry_path, imgcodecs::IMREAD_UNCHANGED).unwrap();
            let size = img.size().unwrap();
            let (width, height) = (size.width as u32, size.height as u32);
            let y_count = height / target_height;
            let x_count = width / target_width;
            let bias_max = x_count.max(y_count);

            // LTR
            for bias in 0..bias_max {
                'outer: for x_index in 0..x_count {
                    'inner: for y_index in 0..y_count {
                        let start_y = y_index * target_height + bias * bias_step;
                        let start_x = x_index * target_width + bias * bias_step;

                        if start_x + target_width > width {
                            continue 'outer;
                        }
                        if start_y + target_height > height {
                            continue 'inner;
                        }

                        let cropped_img = core::Mat::roi(
                            &img,
                            core::Rect::new(
                                start_x as i32,
                                start_y as i32,
                                target_width as i32,
                                target_height as i32,
                            ),
                        )
                        .unwrap();
                        imgcodecs::imwrite(
                            format!(
                                "{}/../output/{}_LTR_bias{}_x{}_y{}.{}",
                                dataset_path,
                                entry.path().file_stem().unwrap().to_str().unwrap(),
                                bias,
                                x_index,
                                y_index,
                                entry.path().extension().unwrap().to_str().unwrap()
                            )
                            .as_str(),
                            &cropped_img,
                            &core::Vector::new(),
                        )
                        .unwrap();
                    }
                }
                tracing::info!(
                    "Image LTR {} bias {} done",
                    entry.file_name().to_str().unwrap(),
                    bias
                );
            }

            // RTL
            for bias in 0..bias_max {
                'outer: for x_index in 0..x_count {
                    'inner: for y_index in 0..y_count {
                        let start_y =
                            height as i32 - (y_index * target_height + bias * bias_step) as i32;
                        let start_x =
                            width as i32 - (x_index * target_width + bias * bias_step) as i32;

                        if start_x < 0
                            || start_x as u32 >= width
                            || start_x as u32 + target_width > width
                        {
                            continue 'outer;
                        }
                        if start_y < 0
                            || start_y as u32 >= height
                            || start_y as u32 + target_height > height
                        {
                            continue 'inner;
                        }

                        let cropped_img = core::Mat::roi(
                            &img,
                            core::Rect::new(
                                start_x,
                                start_y,
                                target_width as i32,
                                target_height as i32,
                            ),
                        )
                        .unwrap();
                        imgcodecs::imwrite(
                            format!(
                                "{}/../output/{}_RTL_bias{}_x{}_y{}.{}",
                                dataset_path,
                                entry.path().file_stem().unwrap().to_str().unwrap(),
                                bias,
                                x_index,
                                y_index,
                                entry.path().extension().unwrap().to_str().unwrap()
                            )
                            .as_str(),
                            &cropped_img,
                            &core::Vector::new(),
                        )
                        .unwrap();
                    }
                }
                tracing::info!(
                    "Image RTL {} bias {} done",
                    entry.file_name().to_str().unwrap(),
                    bias
                );
            }

            tracing::info!("Image {} done", entry.file_name().to_str().unwrap());
        });
    }
    while threads.join_next().await.is_some() {}
}

pub fn check_valid_pixel_count(
    img: &BoxedRef<'_, Mat>,
    rgb_list: &[core::Vec3b],
    mode: bool,
) -> (bool, f64) {
    let count = Arc::new(Mutex::new(0));
    let size = img.size().unwrap();
    let (width, height) = (size.width, size.height);

    (0..height).for_each(|row_index| {
        let row = img.row(row_index).unwrap();
        let mut row_count = 0;
        for col_index in 0..width {
            let pixel = row.at_2d::<core::Vec3b>(0, col_index).unwrap();
            match mode {
                true => {
                    if rgb_list.contains(&pixel) {
                        row_count += 1;
                    }
                }
                false => {
                    if !rgb_list.contains(&pixel) {
                        row_count += 1;
                    }
                }
            }
        }

        let mut count = count.lock();
        *count += row_count;
    });

    let count = *count.lock();

    let ratio = f64::from(count) / f64::from(width * height);
    // if ratio > 0.01 {
    //     tracing::info!("Valid ratio {}", ratio);
    // }
    (ratio > 0.01, ratio)
}

pub async fn process_dataset_with_rgblist(
    dataset_path: &String,
    rgb_list_str: &str,
    valid_rgb_mode: bool,
) {
    let rgb_list: Arc<RwLock<Vec<core::Vec3b>>> = Arc::new(RwLock::new(Vec::new()));
    for rgb in rgb_list_str.split(";") {
        let rgb_list = Arc::clone(&rgb_list);
        let mut rgb_list = rgb_list.write().unwrap();

        let rgb_vec: Vec<u8> = rgb.split(',').map(|s| s.parse::<u8>().unwrap()).collect();

        rgb_list.push(core::Vec3b::from([rgb_vec[0], rgb_vec[1], rgb_vec[2]]));
    }

    tracing::info!(
        "RGB list length: {} Mode: {}",
        rgb_list.read().unwrap().len(),
        match valid_rgb_mode {
            true => "Valid",
            false => "Invalid",
        }
    );

    let image_entries: Vec<PathBuf>;
    let label_entries: Vec<PathBuf>;

    let image_output_path: String;
    let label_output_path: String;

    let image_path = PathBuf::from(dataset_path).join("images");
    let label_path = PathBuf::from(dataset_path).join("labels");
    tracing::debug!(
        "Image path: {}, label path: {}",
        image_path.display(),
        label_path.display()
    );
    tracing::debug!(
        "Image path is dir: {}, label path is dir: {}",
        image_path.is_dir(),
        label_path.is_dir()
    );
    if !(label_path.is_dir() && image_path.is_dir()) {
        tracing::error!("Image and label path should be both directories and exist.");
        return ();
    }

    if image_path.is_dir() {
        image_entries = fs::read_dir(&image_path)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();

        image_output_path = format!("{}/filtered_images/", image_path.to_str().unwrap());
    } else {
        image_entries = vec![image_path.clone()];
        image_output_path = format!(
            "{}/filtered/images/",
            image_path.parent().unwrap().to_str().unwrap()
        );
    }

    if label_path.is_dir() {
        label_entries = fs::read_dir(&label_path)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();

        label_output_path = format!("{}/filtered_labels/", label_path.to_str().unwrap());
    } else {
        label_entries = vec![label_path.clone()];
        label_output_path = format!(
            "{}/filtered/labels/",
            label_path.parent().unwrap().to_str().unwrap()
        );
    }

    let sem = Arc::new(Semaphore::new(
        (*THREAD_POOL.read().expect_or_log("Get pool error")).into(),
    ));
    let valid_id = Arc::new(RwLock::new(Vec::<String>::new()));
    let mut threads = tokio::task::JoinSet::new();

    match fs::create_dir_all(&image_output_path) {
        Ok(_) => {
            tracing::info!("Image output directory created");
        }
        Err(e) => {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                tracing::info!("Image output directory already exists");
            } else {
                tracing::error!("Failed to create directory: {}", e);
                return ();
            }
        }
    }

    match fs::create_dir_all(&label_output_path) {
        Ok(_) => {
            tracing::info!("Label output directory created");
        }
        Err(e) => {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                tracing::info!("Label output directory already exists");
            } else {
                tracing::error!("Failed to create directory: {}", e);
                return ();
            }
        }
    }

    tracing::info!("Processing labels");
    let header_span = info_span!("filter_dataset_with_rgblist_label_thread");
    header_span.pb_set_style(
        &ProgressStyle::with_template("{spinner} Processing {msg}\n{wide_bar} {pos}/{len}")
            .unwrap(),
    );
    header_span.pb_set_length(label_entries.len() as u64);
    header_span.pb_set_message("Processing");
    let _header_guard = header_span.enter();

    // for path in fs::read_dir(&label_output_path).unwrap() {
    //     let path = path.unwrap().path();
    //     valid_id
    //         .write()
    //         .unwrap()
    //         .push(path.file_stem().unwrap().to_str().unwrap().to_string());
    // }

    for path in label_entries {
        let sem = Arc::clone(&sem);
        let valid_id = Arc::clone(&valid_id);
        let rgb_list = Arc::clone(&rgb_list);
        let label_output_path = label_output_path.clone();

        threads.spawn(
            async move {
                let _permit = sem.acquire().await.unwrap();
                let mut img =
                    imgcodecs::imread(&path.to_str().unwrap(), imgcodecs::IMREAD_COLOR).unwrap();
                if img.channels() != 3 {
                    tracing::error!(
                        "Image {} is not RGB format (has {}), skipping!",
                        path.file_name().unwrap().to_str().unwrap(),
                        img.channels()
                    );
                    return;
                }
                unsafe {
                    img.modify_inplace(|input, output| {
                        opencv::imgproc::cvt_color(input, output, COLOR_BGR2RGB, 0)
                            .expect_or_log("Cvt BGR to RGB error")
                    });
                }
                let img = BoxedRef::from(img);
                let rgb_list = rgb_list.read().unwrap();
                let (valid, _) = check_valid_pixel_count(&img, &rgb_list, valid_rgb_mode);
                if valid {
                    valid_id
                        .write()
                        .unwrap()
                        .push(path.file_stem().unwrap().to_str().unwrap().to_string());
                    imgcodecs::imwrite(
                        &format!(
                            "{}//{}.{}",
                            label_output_path,
                            path.file_stem().unwrap().to_str().unwrap(),
                            path.extension().unwrap().to_str().unwrap(),
                        ),
                        &img,
                        &core::Vector::new(),
                    )
                    .unwrap();
                }
                Span::current()
                    .pb_set_message(&format!("{}", path.file_name().unwrap().to_str().unwrap()));
                Span::current().pb_inc(1);
            }
            .in_current_span(),
        );
    }

    while threads.join_next().await.is_some() {}
    drop(_header_guard);
    drop(header_span);
    tracing::info!(
        "Labels filter done, total {} valid labels",
        valid_id.read().unwrap().len()
    );

    tracing::info!("Processing images");
    let header_span = info_span!("filter_dataset_with_rgblist_image_thread");
    header_span.pb_set_style(
        &ProgressStyle::with_template("{spinner} Processing {msg}\n{wide_bar} {pos}/{len}")
            .unwrap(),
    );
    header_span.pb_set_length(image_entries.len() as u64);
    header_span.pb_set_message("Processing");
    let _header_guard = header_span.enter();
    for path in image_entries {
        let sem = Arc::clone(&sem);
        let valid_id = Arc::clone(&valid_id);
        let image_output_path = image_output_path.clone();

        threads.spawn(
            async move {
                let _permit = sem.acquire().await.unwrap();
                let img = imgcodecs::imread(&path.to_str().unwrap(), imgcodecs::IMREAD_UNCHANGED)
                    .unwrap();

                let img = BoxedRef::from(img);
                if valid_id
                    .read()
                    .unwrap()
                    .contains(&path.file_stem().unwrap().to_str().unwrap().to_string())
                {
                    imgcodecs::imwrite(
                        &format!(
                            "{}//{}.{}",
                            image_output_path,
                            path.file_stem().unwrap().to_str().unwrap(),
                            path.extension().unwrap().to_str().unwrap(),
                        ),
                        &img,
                        &core::Vector::new(),
                    )
                    .unwrap();
                }
                Span::current()
                    .pb_set_message(&format!("{}", path.file_name().unwrap().to_str().unwrap()));
                Span::current().pb_inc(1);
            }
            .in_current_span(),
        );
    }
    while threads.join_next().await.is_some() {}
    drop(_header_guard);
    drop(header_span);
    tracing::info!("Images filter done");
}

pub async fn split_images_with_rgb_filter(
    images_path: &String,
    target_height: &u32,
    target_width: &u32,
    rgb_list_str: &str,
    valid_rgb_mode: bool,
) {
    let rgb_list: Arc<RwLock<Vec<core::Vec3b>>> = Arc::new(RwLock::new(Vec::new()));
    for rgb in rgb_list_str.split(";") {
        let rgb_list = Arc::clone(&rgb_list);
        let mut rgb_list = rgb_list.write().unwrap();

        let rgb_vec: Vec<u8> = rgb.split(',').map(|s| s.parse::<u8>().unwrap()).collect();

        rgb_list.push(core::Vec3b::from([rgb_vec[0], rgb_vec[1], rgb_vec[2]]));
    }

    tracing::info!(
        "RGB list length: {} Mode: {}",
        rgb_list.read().unwrap().len(),
        match valid_rgb_mode {
            true => "Valid",
            false => "Invalid",
        }
    );

    let label_entries: Vec<PathBuf>;

    let label_output_path: String;

    let label_path = PathBuf::from(images_path.as_str());

    if label_path.is_dir() {
        label_entries = fs::read_dir(&label_path)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();

        label_output_path = format!("{}/output/", label_path.to_str().unwrap());
    } else {
        label_entries = vec![label_path.clone()];
        label_output_path = format!(
            "{}/output/labels/",
            label_path.parent().unwrap().to_str().unwrap()
        );
    }

    let sem = Arc::new(Semaphore::new(
        (*THREAD_POOL.read().expect_or_log("Get pool error")).into(),
    ));
    tracing::info!("sem available permits: {}", sem.available_permits());
    let mut threads = tokio::task::JoinSet::new();

    match fs::create_dir_all(&label_output_path) {
        Ok(_) => {
            tracing::info!("Label output directory created");
        }
        Err(e) => {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                tracing::info!("Label output directory already exists");
            } else {
                tracing::error!("Failed to create directory: {}", e);
                return ();
            }
        }
    }

    // Label Processing
    let mut label_extension = None;
    for entry in label_entries {
        if !entry.is_file() {
            continue;
        }

        if label_extension.is_none() {
            let extension = entry
                .extension()
                .unwrap()
                .to_os_string()
                .into_string()
                .unwrap();
            label_extension = Some(extension);
        }

        let permit = Arc::clone(&sem);
        let target_width = *target_width;
        let target_height = *target_height;
        let rgb_list = Arc::clone(&rgb_list);
        let label_extension = label_extension.clone();

        let label_output_path = label_output_path.clone();
        threads.spawn(async move {
            let _ = permit.acquire().await.unwrap();
            tracing::info!(
                "Processing {}",
                entry.file_name().unwrap().to_str().unwrap()
            );
            let label_id = entry.file_stem().unwrap().to_str().unwrap().to_string();
            let mut img =
                imgcodecs::imread(entry.to_str().unwrap(), imgcodecs::IMREAD_COLOR).unwrap();

            if img.channels() != 3 {
                tracing::error!(
                    "Image {} is not RGB format (has {}), skipping!",
                    entry.file_name().unwrap().to_str().unwrap(),
                    img.channels()
                );
                return;
            }
            unsafe {
                img.modify_inplace(|input, output| {
                    opencv::imgproc::cvt_color(input, output, COLOR_BGR2RGB, 0)
                        .expect_or_log("Cvt BGR to RGB error")
                });
            }
            let size = img.size().unwrap();
            let (width, height) = (size.width, size.height);
            let y_count = height / target_height as i32;
            let x_count = width / target_width as i32;
            // let mut labels_map = HashMap::<String, Mat>::new();

            // Crop horizontally from left
            let row_iter = ProgressAdaptor::new(0..y_count);
            let row_progress = row_iter.items_processed();
            row_iter.for_each(|row_index| {
                for col_index in 0..x_count {
                    let label_id = format!("{}_LTR_x{}_y{}", label_id, col_index, row_index);
                    let cropped = core::Mat::roi(
                        &img,
                        core::Rect::new(
                            col_index * target_width as i32,
                            row_index * target_height as i32,
                            target_width as i32,
                            target_height as i32,
                        ),
                    )
                    .unwrap();

                    let rgb_list = rgb_list.read().unwrap();
                    if check_valid_pixel_count(&cropped, &rgb_list, valid_rgb_mode).0 {
                        imgcodecs::imwrite(
                            &format!(
                                "{}/{}.{}",
                                label_output_path,
                                label_id,
                                label_extension.as_ref().unwrap()
                            ),
                            &cropped,
                            &core::Vector::new(),
                        )
                        .unwrap();
                    }
                }
                if row_progress.get() != 0 && row_progress.get() % 10 == 0 {
                    tracing::info!(
                        "Label {} LTR Row {} / {} done",
                        label_id,
                        row_progress.get(),
                        y_count
                    );
                }
            });

            tracing::info!("Label {} LTR iteration done", label_id);

            // Crop horizontally from right
            let row_iter = ProgressAdaptor::new(0..y_count);
            let row_progress = row_iter.items_processed();
            row_iter.for_each(|row_index| {
                for col_index in 0..x_count {
                    let label_id = format!("{}_RTL_x{}_y{}", label_id, col_index, row_index);
                    let cropped = core::Mat::roi(
                        &img,
                        core::Rect::new(
                            width - (col_index + 1) * target_width as i32,
                            height - (row_index + 1) * target_height as i32,
                            target_width as i32,
                            target_height as i32,
                        ),
                    )
                    .unwrap();
                    let rgb_list = rgb_list.read().unwrap();
                    if check_valid_pixel_count(&cropped, &rgb_list, valid_rgb_mode).0 {
                        imgcodecs::imwrite(
                            &format!(
                                "{}/{}.{}",
                                label_output_path,
                                label_id,
                                label_extension.as_ref().unwrap()
                            ),
                            &cropped,
                            &core::Vector::new(),
                        )
                        .unwrap();
                    }
                }
                if row_progress.get() != 0 && row_progress.get() % 10 == 0 {
                    tracing::info!(
                        "Label {} RTL Row {} / {} done",
                        label_id,
                        row_progress.get(),
                        y_count
                    );
                }
            });
            tracing::info!("Label {} RTL iteration done", label_id);

            tracing::info!("Label {} process done", label_id);
        });
    }

    while threads.join_next().await.is_some() {}
}

pub async fn split_images_with_label_filter(
    images_path: &String,
    labels_path: &String,
    target_height: &u32,
    target_width: &u32,
) -> Result<()> {
    let mut valid_name_set = HashSet::<String>::new();

    for entry in fs::read_dir(labels_path)? {
        let path = entry?.path();
        if let Some(name) = path.file_stem().map(|s| s.to_string_lossy()) {
            valid_name_set.insert(name.into_owned());
        } else {
            bail!("Failed to read file stem, illigal path");
        }
    }
    let valid_name_set = Arc::new(valid_name_set);

    let image_entries: Vec<PathBuf>;
    let images_output_path: String;
    let images_path = PathBuf::from(images_path.as_str());

    if images_path.is_dir() {
        image_entries = fs::read_dir(&images_path)
            .context(format!("Failed to read directory: {:?}", images_path))?
            .map(|e| e.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()?;
        images_output_path = images_path.join("output").to_string_lossy().into_owned();
    } else {
        image_entries = vec![images_path.clone()];
        images_output_path = images_path
            .parent()
            .ok_or(anyhow!("Failed to get parent dir"))?
            .join("output")
            .join("labels")
            .to_string_lossy()
            .into_owned();
    }

    let sem = Arc::new(Semaphore::new(
        (*THREAD_POOL.read().expect_or_log("Get pool error")).into(),
    ));
    tracing::info!("sem available permits: {}", sem.available_permits());
    let mut threads = tokio::task::JoinSet::new();

    match fs::create_dir_all(&images_output_path) {
        Ok(_) => {
            tracing::info!("Image output directory created");
        }
        Err(e) => {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                tracing::info!("Image output directory already exists");
            } else {
                bail!("Failed to create directory: {}", e);
            }
        }
    }

    let header_span = info_span!("split_images_threads");
    header_span.pb_set_style(
        &ProgressStyle::with_template("{spinner} Processing {msg}\n{wide_bar} {pos}/{len}")
            .map_err(|e| anyhow!("Tracing progress template generate failed {e}"))?,
    );
    header_span.pb_set_length(image_entries.len() as u64);
    header_span.pb_set_message("starting...");

    let header_span_enter = header_span.enter();

    // Image Processing
    let mut image_extension = None;
    for entry in image_entries {
        if !entry.is_file() {
            continue;
        }

        if image_extension.is_none() {
            let extension: String = entry
                .extension()
                .ok_or(anyhow!("Failed to get file extension"))?
                .to_os_string()
                .into_string()
                .map_err(|e| anyhow!("Failed to convert os string {:?}", e))?;
            image_extension = Some(extension);
        }

        let target_width = *target_width;
        let target_height = *target_height;
        let valid_name_set = Arc::clone(&valid_name_set);
        let image_extension = image_extension.clone();
        let images_output_path = images_output_path.clone();

        let header_span = header_span.clone();

        threads.spawn_blocking(move || -> anyhow::Result<()> {
            let file_name = entry
                .file_name()
                .ok_or(anyhow!("Failed to get file name"))?
                .to_string_lossy()
                .into_owned();
            let task_span = info_span!(parent: &header_span, "Image Processing", file_name);
            task_span.pb_set_style(
                &ProgressStyle::with_template("{spinner} Processing {msg}\n{wide_bar} {pos}/{len}")
                    .map_err(|e| anyhow!("Tracing progress template generate failed {e}"))?,
            );
            task_span.pb_set_message(&file_name);

            let _guard = task_span.enter();

            tracing::info!("Processing {}", file_name);
            let image_id = match entry.file_stem().map(|s| s.to_string_lossy()) {
                Some(id) => id.into_owned(),
                None => bail!(
                    "Failed to get image id for file {}",
                    entry.as_path().to_string_lossy().into_owned()
                ),
            };
            let img = imgcodecs::imread(
                entry.to_str().ok_or(anyhow!("Failed to get entry path"))?,
                imgcodecs::IMREAD_UNCHANGED,
            )?;

            let size = img.size()?;
            let (width, height) = (size.width, size.height);
            let y_count = height / target_height as i32;
            let x_count = width / target_width as i32;

            task_span.pb_set_length((y_count * x_count * 2) as u64);

            // Crop horizontally from left
            for row_index in 0..y_count {
                for col_index in 0..x_count {
                    let image_id = format!("{}_LTR_x{}_y{}", image_id, col_index, row_index);
                    task_span.pb_inc(1);
                    if !valid_name_set.contains(&image_id) {
                        continue;
                    }
                    let cropped = core::Mat::roi(
                        &img,
                        core::Rect::new(
                            col_index * target_width as i32,
                            row_index * target_height as i32,
                            target_width as i32,
                            target_height as i32,
                        ),
                    )
                    .map_err(|e| anyhow!("Crop ROI error, {e}"))?;

                    imgcodecs::imwrite(
                        &format!(
                            "{}/{}.{}",
                            images_output_path,
                            image_id,
                            image_extension.as_ref().unwrap()
                        ),
                        &cropped,
                        &core::Vector::new(),
                    )
                    .map_err(|e| anyhow!("Image write failed, {e}"))?;

                    if row_index != 0 && row_index % 10 == 0 {
                        tracing::info!(
                            "Image {} LTR Row {} / {} done",
                            image_id,
                            row_index,
                            y_count
                        );
                    }
                }
            }

            tracing::info!("Label {} LTR iteration done", image_id);

            // Crop horizontally from right
            for row_index in 0..y_count {
                for col_index in 0..x_count {
                    let image_id = format!("{}_RTL_x{}_y{}", image_id, col_index, row_index);
                    task_span.pb_inc(1);
                    if !valid_name_set.contains(&image_id) {
                        continue;
                    }
                    let cropped = core::Mat::roi(
                        &img,
                        core::Rect::new(
                            width - (col_index + 1) * target_width as i32,
                            height - (row_index + 1) * target_height as i32,
                            target_width as i32,
                            target_height as i32,
                        ),
                    )
                    .map_err(|e| anyhow!("Crop ROI error, {e}"))?;

                    imgcodecs::imwrite(
                        &format!(
                            "{}/{}.{}",
                            images_output_path,
                            image_id,
                            image_extension.as_ref().unwrap()
                        ),
                        &cropped,
                        &core::Vector::new(),
                    )
                    .map_err(|e| anyhow!("Image write failed, {e}"))?;
                }
                if row_index != 0 && row_index % 10 == 0 {
                    tracing::info!(
                        "Image {} RTL Row {} / {} done",
                        image_id,
                        row_index,
                        y_count
                    );
                }
            }
            tracing::info!("Image {} RTL iteration done", image_id);
            tracing::info!("Image {} process done", image_id);

            header_span.pb_inc(1);

            Ok(())
        });
    }

    while threads.join_next().await.is_some() {}

    drop(header_span_enter);
    Ok(())
}

pub async fn stich_images(splited_images: &String, target_height: &i32, target_width: &i32) {
    let entries = fs::read_dir(splited_images).unwrap();
    let mut size: Option<(i32, i32)> = None;

    let mut result_mat = Mat::new_rows_cols_with_default(
        *target_height,
        *target_width,
        core::CV_8UC3,
        opencv::core::Scalar::all(0.),
    )
    .unwrap();
    let re = regex::Regex::new(r"^(.*)_(LTR|RTL)_x(\d*)_y(\d*)\.(jpg|tif)")
        .expect_or_log("Failed to compile regex");

    for entry in entries {
        let entry = entry.unwrap();
        let file_name = entry.file_name();
        let file_name = file_name.to_str().unwrap();

        // Image name, direction, x, y
        let mut info = Vec::new();
        while let Some(m) = re.captures(file_name) {
            info.push(m.get(1).unwrap().as_str());
            info.push(m.get(2).unwrap().as_str());
            info.push(m.get(3).unwrap().as_str());
            info.push(m.get(4).unwrap().as_str());
            break;
        }

        // tracing::trace!("Captures {:?}", re.captures(file_name));

        if info.len() != 4 {
            tracing::error!("Failed to parse image name {}", file_name);
            return;
        }

        let img =
            imgcodecs::imread(entry.path().to_str().unwrap(), imgcodecs::IMREAD_UNCHANGED).unwrap();

        let current_img_size = img.size().unwrap();
        if size.is_none() {
            size = Some((current_img_size.width, current_img_size.height));
        } else {
            if current_img_size.width != size.unwrap().0
                || current_img_size.height != size.unwrap().1
            {
                tracing::error!(
                    "Image {} size is not consistent",
                    entry.file_name().to_str().unwrap()
                );
                return;
            }
        }

        if info[1] == "RTL" {
            let x = target_width - ((info[2].parse::<i32>().unwrap() + 1) * size.unwrap().0);
            let y = target_height - ((info[3].parse::<i32>().unwrap() + 1) * size.unwrap().1);
            tracing::trace!("RTL x {} y {}", x, y);
            let mut roi = Mat::roi_mut(
                &mut result_mat,
                core::Rect::new(x, y, size.unwrap().0, size.unwrap().1),
            )
            .expect_or_log("Failed to create roi");
            img.copy_to(&mut roi).expect_or_log("Failed to copy image");
        } else if info[1] == "LTR" {
            let mut roi = Mat::roi_mut(
                &mut result_mat,
                core::Rect::new(
                    info[2].parse::<i32>().unwrap() * size.unwrap().0,
                    info[3].parse::<i32>().unwrap() * size.unwrap().1,
                    size.unwrap().0,
                    size.unwrap().1,
                ),
            )
            .expect_or_log("Failed to create roi");
            img.copy_to(&mut roi).expect_or_log("Failed to copy image");
        } else {
            tracing::error!("Failed to parse image direction {}", info[1]);
            return;
        }

        tracing::info!("Image {} processed", entry.file_name().to_str().unwrap());
    }

    imgcodecs::imwrite(
        format!("{}/stiched.png", splited_images).as_str(),
        &result_mat,
        &core::Vector::new(),
    )
    .expect_or_log("Failed to save image");
}
