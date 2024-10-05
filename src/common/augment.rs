use std::{
    fs::{self},
    path::PathBuf,
    sync::{Arc, RwLock},
};

use tokio::{sync::Semaphore, task::JoinSet};

use opencv::{core, imgcodecs, prelude::*};
use tracing_unwrap::{OptionExt, ResultExt};

pub async fn split_images(dataset_path: &String, target_height: &u32, target_width: &u32) {
    let entries = fs::read_dir(dataset_path).expect_or_log("Failed to read directory");
    let mut threads = JoinSet::new();

    fs::create_dir_all(format!("{}\\..\\output\\", dataset_path))
        .expect_or_log("Failed to create directory");

    for entry in entries {
        let entry = entry.expect_or_log("Failed to iterate entries");
        if entry.path().is_dir() {
            continue;
        }
        let dataset_path = dataset_path.clone();
        let target_height = *target_height;
        let target_width = *target_width;
        let entry_path = entry
            .path()
            .to_str()
            .expect_or_log("Failed to convert path to string")
            .to_owned();

        threads.spawn(async move {
            let file_name = entry
                .file_name()
                .to_str()
                .expect_or_log("Failed to get file name")
                .to_owned();
            tracing::info!("Image {} processing...", file_name);

            let file_stem = entry
                .path()
                .file_stem()
                .expect_or_log("Failed to get file stem")
                .to_str()
                .expect_or_log("Failed to convert file stem to string")
                .to_owned();

            let file_extension = entry
                .path()
                .extension()
                .expect_or_log("Failed to get file extension")
                .to_str()
                .expect_or_log("Failed to convert file extension to string")
                .to_owned();

            // 读取图片
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
                        "{}\\..\\output\\{}_LTR_x{}_y{}.{}",
                        dataset_path, file_stem, x_index, y_index, file_extension
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
            tracing::info!("Image LTR {} done", entry.file_name().to_str().unwrap());

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
                        "{}\\..\\output\\{}_RTL_x{}_y{}.{}",
                        dataset_path, file_stem, x_index, y_index, file_extension
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
            tracing::info!("Image RTL {} done", file_name,);

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

    fs::create_dir_all(format!("{}\\..\\output\\", dataset_path)).unwrap();

    for entry in entries {
        let entry = entry.unwrap();
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
                                "{}\\..\\output\\{}_LTR_bias{}_x{}_y{}.{}",
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
                                "{}\\..\\output\\{}_RTL_bias{}_x{}_y{}.png",
                                dataset_path,
                                entry.file_name().to_str().unwrap().replace(".png", ""),
                                bias,
                                x_index,
                                y_index
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

pub fn check_valid_pixel_count(img: &Mat, rgb_list: &[core::Vec3b], mode: bool) -> bool {
    let mut count = 0;
    let size = img.size().unwrap();
    let (width, height) = (size.width, size.height);

    for y in 0..height {
        for x in 0..width {
            let pixel = img.at_2d::<core::Vec3b>(y, x).unwrap();
            match mode {
                true => {
                    if rgb_list.contains(&pixel) {
                        count += 1;
                    }
                }
                false => {
                    if !rgb_list.contains(&pixel) {
                        count += 1;
                    }
                }
            }
        }
    }

    let ratio = (count as f32) / ((width * height) as f32);
    // if ratio > 0.01 {
    //     tracing::info!("Valid ratio {}", ratio);
    // }
    ratio > 0.01
}
pub async fn split_images_with_filter(
    image_path: &String,
    label_path: &String,
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

    let image_entries: Vec<PathBuf>;
    let label_entries: Vec<PathBuf>;

    let image_output_path: String;
    let label_output_path: String;

    let image_path = PathBuf::from(image_path.as_str());
    let label_path = PathBuf::from(label_path.as_str());
    if label_path.is_dir() ^ image_path.is_dir() {
        tracing::error!("Image and label path should be both directories or both files");
        return ();
    }

    if image_path.is_dir() {
        image_entries = fs::read_dir(&image_path)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();

        image_output_path = format!("{}\\output\\", image_path.to_str().unwrap());
    } else {
        image_entries = vec![image_path.clone()];
        image_output_path = format!(
            "{}\\..\\output\\images\\",
            image_path.parent().unwrap().to_str().unwrap()
        );
    }

    if label_path.is_dir() {
        label_entries = fs::read_dir(&label_path)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();

        label_output_path = format!("{}\\output\\", label_path.to_str().unwrap());
    } else {
        label_entries = vec![label_path.clone()];
        label_output_path = format!(
            "{}\\..\\output\\labels\\",
            label_path.parent().unwrap().to_str().unwrap()
        );
    }

    let sem = Arc::new(Semaphore::new(5));
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
        let valid_id = Arc::clone(&valid_id);
        let target_width = *target_width;
        let target_height = *target_height;
        let rgb_list = Arc::clone(&rgb_list);
        let label_extension = label_extension.clone();

        let label_output_path = label_output_path.clone();
        threads.spawn(async move {
            let _ = permit.acquire().await.unwrap();
            let label_id = entry.file_stem().unwrap().to_str().unwrap().to_string();
            let img =
                imgcodecs::imread(entry.to_str().unwrap(), imgcodecs::IMREAD_UNCHANGED).unwrap();

            let size = img.size().unwrap();
            let (width, height) = (size.width, size.height);
            let y_count = height / target_height as i32;
            let x_count = width / target_width as i32;
            // let mut labels_map = HashMap::<String, Mat>::new();

            // Crop horizontally from left
            for x_index in 0..x_count {
                for y_index in 0..y_count {
                    let label_id = format!("{}_lt2rb_{}_{}", label_id, x_index, y_index);
                    let cropped = core::Mat::roi(
                        &img,
                        core::Rect::new(
                            x_index * target_width as i32,
                            y_index * target_height as i32,
                            target_width as i32,
                            target_height as i32,
                        ),
                    )
                    .unwrap()
                    .clone_pointee();

                    let rgb_list = rgb_list.read().unwrap();
                    if check_valid_pixel_count(&cropped, &rgb_list, valid_rgb_mode) {
                        imgcodecs::imwrite(
                            &format!(
                                "{}\\{}.{}",
                                label_output_path,
                                label_id,
                                label_extension.as_ref().unwrap()
                            ),
                            &cropped,
                            &core::Vector::new(),
                        )
                        .unwrap();
                        valid_id.write().unwrap().push(label_id.clone());
                    }
                    tracing::info!("Label {} processed", label_id);
                }
            }

            tracing::info!("Label {} LTR iteration done", label_id);

            // Crop horizontally from right
            for x_index in 0..x_count {
                for y_index in 0..y_count {
                    let label_id = format!("{}_rb2lt_{}_{}", label_id, x_index, y_index);
                    let cropped = core::Mat::roi(
                        &img,
                        core::Rect::new(
                            width - (x_index + 1) * target_width as i32,
                            height - (y_index + 1) * target_height as i32,
                            target_width as i32,
                            target_height as i32,
                        ),
                    )
                    .unwrap()
                    .clone_pointee();
                    let rgb_list = rgb_list.read().unwrap();
                    if check_valid_pixel_count(&cropped, &rgb_list, valid_rgb_mode) {
                        imgcodecs::imwrite(
                            &format!(
                                "{}\\{}.{}",
                                label_output_path,
                                label_id,
                                label_extension.as_ref().unwrap()
                            ),
                            &cropped,
                            &core::Vector::new(),
                        )
                        .unwrap();
                        valid_id.write().unwrap().push(label_id.clone());
                    }
                    tracing::info!("Label {} processed", label_id);
                }
            }

            tracing::info!("Label {} RTL iteration done", label_id);

            tracing::info!("Label {} process done", label_id);
        });
    }

    while threads.join_next().await.is_some() {}

    tracing::info!("Labels process done");

    let mut image_extension = None;
    for entry in image_entries {
        if !entry.is_file() {
            continue;
        }

        if image_extension.is_none() {
            let extension = entry
                .extension()
                .unwrap()
                .to_os_string()
                .into_string()
                .unwrap();
            image_extension = Some(extension);
        }

        let permit = Arc::clone(&sem);
        // let cropped_images = Arc::clone(&cropped_images);
        let valid_id = Arc::clone(&valid_id);
        let target_height = *target_height;
        let target_width = *target_width;
        let image_extension = image_extension.clone();
        let image_output_path = image_output_path.clone();
        threads.spawn(async move {
            let _ = permit.acquire().await.unwrap();
            let img_id = entry.file_stem().unwrap().to_str().unwrap().to_string();
            let img =
                imgcodecs::imread(entry.to_str().unwrap(), imgcodecs::IMREAD_UNCHANGED).unwrap();

            let size = img.size().unwrap();
            let (width, height) = (size.width, size.height);
            let y_count = height / target_height as i32;
            let x_count = width / target_width as i32;

            // Crop horizontally from left
            for x_index in 0..x_count {
                for y_index in 0..y_count {
                    let img_id = format!("{}_lt2rb_{}_{}", img_id, x_index, y_index);
                    if !valid_id.read().unwrap().contains(&img_id) {
                        continue;
                    }
                    let cropped = core::Mat::roi(
                        &img,
                        core::Rect::new(
                            x_index * target_width as i32,
                            y_index * target_height as i32,
                            target_width as i32,
                            target_height as i32,
                        ),
                    )
                    .unwrap()
                    .clone_pointee();
                    imgcodecs::imwrite(
                        &format!(
                            "{}\\{}.{}",
                            image_output_path,
                            img_id,
                            image_extension.as_ref().unwrap()
                        ),
                        &cropped,
                        &core::Vector::new(),
                    )
                    .unwrap();
                    tracing::info!("Image {} processed", img_id);
                }
            }

            tracing::info!("Image {} RTL iteration done", img_id);

            // Crop horizontally from right
            for x_index in 0..x_count {
                for y_index in 0..y_count {
                    let img_id = format!("{}_rb2lt_{}_{}", img_id, x_index, y_index);
                    if !valid_id.read().unwrap().contains(&img_id) {
                        continue;
                    }
                    let cropped = core::Mat::roi(
                        &img,
                        core::Rect::new(
                            width - (x_index + 1) * target_width as i32,
                            height - (y_index + 1) * target_height as i32,
                            target_width as i32,
                            target_height as i32,
                        ),
                    )
                    .unwrap()
                    .clone_pointee();
                    imgcodecs::imwrite(
                        &format!(
                            "{}\\{}.{}",
                            image_output_path,
                            img_id,
                            image_extension.as_ref().unwrap()
                        ),
                        &cropped,
                        &core::Vector::new(),
                    )
                    .unwrap();
                    tracing::info!("Image {} processed", img_id);
                }
            }

            tracing::info!("Image {} LTR iteration done", img_id);

            // cropped_images.lock().unwrap().extend(imgs_map);
            tracing::info!("Image {} process done", img_id);
        });
    }

    while threads.join_next().await.is_some() {}
}
