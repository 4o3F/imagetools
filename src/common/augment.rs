use std::{
    fs,
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

            let vertical_count = height / target_height;
            let horizontal_count = width / target_width;

            tracing::trace!(
                "Image {} vertical_count {} horizontal_count {}",
                file_name,
                vertical_count,
                horizontal_count
            );

            // LTR
            'outer: for horizontal_index in 0..horizontal_count {
                'inner: for vertical_index in 0..vertical_count {
                    let start_y = horizontal_index * target_height;
                    let start_x = vertical_index * target_width;

                    if start_x + target_width > width {
                        continue 'inner;
                    }
                    if start_y + target_height > height {
                        continue 'outer;
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
                        "{}\\..\\output\\{}_LTR_h{}_v{}.{}",
                        dataset_path, file_stem, horizontal_index, vertical_index, file_extension
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
            'outer: for horizontal_index in 0..horizontal_count {
                'inner: for vertical_index in 0..vertical_count {
                    let start_y = height as i32 - (horizontal_index * target_height) as i32;
                    let start_x = width as i32 - (vertical_index * target_width) as i32;

                    if start_x < 0
                        || start_x as u32 >= width
                        || start_x as u32 + target_width > width
                    {
                        continue 'inner;
                    }
                    if start_y < 0
                        || start_y as u32 >= height
                        || start_y as u32 + target_height > height
                    {
                        continue 'outer;
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
                        "{}\\..\\output\\{}_RTL_h{}_v{}.{}",
                        dataset_path, file_stem, horizontal_index, vertical_index, file_extension
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
            let vertical_count = height / target_height;
            let horizontal_count = width / target_width;
            let bias_max = horizontal_count.max(vertical_count);

            // LTR
            for bias in 0..bias_max {
                'outer: for horizontal_index in 0..horizontal_count {
                    'inner: for vertical_index in 0..vertical_count {
                        let start_y = horizontal_index * target_height + bias * bias_step;
                        let start_x = vertical_index * target_width + bias * bias_step;

                        if start_x + target_width > width {
                            continue 'inner;
                        }
                        if start_y + target_height > height {
                            continue 'outer;
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
                                "{}\\..\\output\\{}_LTR_bias{}_h{}_v{}.{}",
                                dataset_path,
                                entry.path().file_stem().unwrap().to_str().unwrap(),
                                bias,
                                horizontal_index,
                                vertical_index,
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
                'outer: for horizontal_index in 0..horizontal_count {
                    'inner: for vertical_index in 0..vertical_count {
                        let start_y = height as i32
                            - (horizontal_index * target_height + bias * bias_step) as i32;
                        let start_x = width as i32
                            - (vertical_index * target_width + bias * bias_step) as i32;

                        if start_x < 0
                            || start_x as u32 >= width
                            || start_x as u32 + target_width > width
                        {
                            continue 'inner;
                        }
                        if start_y < 0
                            || start_y as u32 >= height
                            || start_y as u32 + target_height > height
                        {
                            continue 'outer;
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
                                "{}\\..\\output\\{}_RTL_bias{}_h{}_v{}.png",
                                dataset_path,
                                entry.file_name().to_str().unwrap().replace(".png", ""),
                                bias,
                                horizontal_index,
                                vertical_index
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

pub fn check_valid_pixel_count(img: &Mat, valid_rgb_list: &[core::Vec3b]) -> bool {
    let mut count = 0;
    let size = img.size().unwrap();
    let (width, height) = (size.width, size.height);

    for y in 0..height {
        for x in 0..width {
            let pixel = img.at_2d::<core::Vec3b>(y, x).unwrap();
            if valid_rgb_list.contains(&pixel) {
                count += 1;
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
    valid_rgb_list_str: &str,
) {
    let valid_rgb_list: Arc<RwLock<Vec<core::Vec3b>>> = Arc::new(RwLock::new(Vec::new()));
    for rgb in valid_rgb_list_str.split(";") {
        let valid_rgb_list = Arc::clone(&valid_rgb_list);
        let mut valid_rgb_list = valid_rgb_list.write().unwrap();

        let rgb_vec: Vec<u8> = rgb.split(',').map(|s| s.parse::<u8>().unwrap()).collect();

        valid_rgb_list.push(core::Vec3b::from([rgb_vec[0], rgb_vec[1], rgb_vec[2]]));
    }

    tracing::info!(
        "Valid rgb list length: {}",
        valid_rgb_list.read().unwrap().len()
    );

    let image_entries = fs::read_dir(image_path).unwrap();
    let label_entries = fs::read_dir(label_path).unwrap();
    let sem = Arc::new(Semaphore::new(3));
    // let cropped_images = Arc::new(Mutex::new(HashMap::<String, Mat>::new()));
    // let cropped_labels = Arc::new(Mutex::new(HashMap::<String, Mat>::new()));
    let invalid_id = Arc::new(RwLock::new(Vec::<String>::new()));
    let mut threads = tokio::task::JoinSet::new();

    fs::create_dir_all(format!("{}\\output\\", image_path)).unwrap();
    fs::create_dir_all(format!("{}\\output\\", label_path)).unwrap();

    let mut label_extension = None;
    for entry in label_entries {
        let entry = entry.unwrap();

        if !entry.path().is_file() {
            continue;
        }

        if label_extension.is_none() {
            let extension = entry
                .path()
                .extension()
                .unwrap()
                .to_os_string()
                .into_string()
                .unwrap();
            label_extension = Some(extension);
        }

        let permit = Arc::clone(&sem);
        // let cropped_labels = Arc::clone(&cropped_labels);
        // let cropped_images = Arc::clone(&cropped_images);
        let invalid_id = Arc::clone(&invalid_id);
        let target_width = *target_width;
        let target_height = *target_height;
        let valid_rgb_list = Arc::clone(&valid_rgb_list);
        let label_path = label_path.clone();
        let label_extension = label_extension.clone();

        threads.spawn(async move {
            let _ = permit.acquire().await.unwrap();
            let label_id = entry
                .path()
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            let img =
                imgcodecs::imread(entry.path().to_str().unwrap(), imgcodecs::IMREAD_UNCHANGED)
                    .unwrap();

            let vertical_count = img.rows() / target_height as i32;
            let horizontal_count = img.cols() / target_width as i32;
            // let mut labels_map = HashMap::<String, Mat>::new();

            // Crop horizontally from left
            for horizontal_index in 0..horizontal_count {
                for vertical_index in 0..vertical_count {
                    let label_id =
                        format!("{}_rb2lt_{}_{}", label_id, horizontal_index, vertical_index);
                    let cropped = core::Mat::roi(
                        &img,
                        core::Rect::new(
                            horizontal_index * target_width as i32,
                            vertical_index * target_height as i32,
                            target_width as i32,
                            target_height as i32,
                        ),
                    )
                    .unwrap()
                    .clone_pointee();

                    let valid_rgb_list = valid_rgb_list.read().unwrap();
                    if check_valid_pixel_count(&cropped, &valid_rgb_list) {
                        imgcodecs::imwrite(
                            &format!(
                                "{}\\output\\{}.{}",
                                label_path,
                                label_id,
                                label_extension.as_ref().unwrap()
                            ),
                            &cropped,
                            &core::Vector::new(),
                        )
                        .unwrap();
                    } else {
                        invalid_id.write().unwrap().push(label_id.clone());
                    }
                    tracing::info!("Label {} processed", label_id);
                    // labels_map.insert(
                    //     format!("{}_lt2rb_{}_{}", label_id, horizontal_index, vertical_index),
                    //     cropped,
                    // );
                }
            }

            tracing::info!("Label {} LTR iteration done", label_id);

            // Crop horizontally from right
            for horizontal_index in 0..horizontal_count {
                for vertical_index in 0..vertical_count {
                    let label_id =
                        format!("{}_rb2lt_{}_{}", label_id, horizontal_index, vertical_index);
                    let cropped = core::Mat::roi(
                        &img,
                        core::Rect::new(
                            img.cols() - (horizontal_index + 1) * target_width as i32,
                            img.rows() - (vertical_index + 1) * target_height as i32,
                            target_width as i32,
                            target_height as i32,
                        ),
                    )
                    .unwrap()
                    .clone_pointee();
                    let valid_rgb_list = valid_rgb_list.read().unwrap();
                    if check_valid_pixel_count(&cropped, &valid_rgb_list) {
                        imgcodecs::imwrite(
                            &format!(
                                "{}\\output\\{}.{}",
                                label_path,
                                label_id,
                                label_extension.as_ref().unwrap()
                            ),
                            &cropped,
                            &core::Vector::new(),
                        )
                        .unwrap();
                    } else {
                        invalid_id.write().unwrap().push(label_id.clone());
                    }
                    tracing::info!("Label {} processed", label_id);
                }
            }

            tracing::info!("Label {} RTL iteration done", label_id);

            // let mut useless_img_id = Vec::<String>::new();
            // let valid_rgb_list = valid_rgb_list.read().unwrap();
            // for (label_id, label) in labels_map.iter() {
            //     if check_valid_pixel_count(label, &valid_rgb_list) {
            //         continue;
            //     }
            //     useless_img_id.push(label_id.clone());
            // }

            // // let mut cropped_labels = cropped_labels.lock().unwrap();
            // // let mut cropped_images = cropped_images.lock().unwrap();
            // let mut invalid_id = invalid_id
            //     .write()
            //     .expect_or_log("Failed to lock invalid_id vec");
            // for img_id in useless_img_id {
            //     labels_map.remove(&img_id);
            //     // cropped_images.remove(&img_id);
            //     invalid_id.push(img_id);
            // }

            // tracing::info!("Label {} save start", label_id);
            // for (label_id, label) in labels_map.iter() {
            //     imgcodecs::imwrite(
            //         &format!(
            //             "{}\\output\\{}.{}",
            //             label_path,
            //             label_id,
            //             label_extension.as_ref().unwrap()
            //         ),
            //         label,
            //         &core::Vector::new(),
            //     )
            //     .unwrap();
            // }
            // cropped_labels.extend(labels_map);
            tracing::info!("Label {} process done", label_id);
        });
    }

    while threads.join_next().await.is_some() {}

    tracing::info!("Labels process done");

    let mut image_extension = None;
    for entry in image_entries {
        let entry = entry.unwrap();

        if !entry.path().is_file() {
            continue;
        }

        if image_extension.is_none() {
            let extension = entry
                .path()
                .extension()
                .unwrap()
                .to_os_string()
                .into_string()
                .unwrap();
            image_extension = Some(extension);
        }

        let permit = Arc::clone(&sem);
        // let cropped_images = Arc::clone(&cropped_images);
        let invalid_id = Arc::clone(&invalid_id);
        let target_height = *target_height;
        let target_width = *target_width;
        let image_path = image_path.clone();
        let image_extension = image_extension.clone();

        threads.spawn(async move {
            let _ = permit.acquire().await.unwrap();
            let img_id = entry
                .path()
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            let img =
                imgcodecs::imread(entry.path().to_str().unwrap(), imgcodecs::IMREAD_UNCHANGED)
                    .unwrap();

            let vertical_count = img.rows() / target_height as i32;
            let horizontal_count = img.cols() / target_width as i32;
            // let mut imgs_map = HashMap::<String, Mat>::new();

            // Crop horizontally from left
            for horizontal_index in 0..horizontal_count {
                for vertical_index in 0..vertical_count {
                    let img_id =
                        format!("{}_lt2rb_{}_{}", img_id, horizontal_index, vertical_index);
                    if invalid_id.read().unwrap().contains(&img_id) {
                        continue;
                    }
                    let cropped = core::Mat::roi(
                        &img,
                        core::Rect::new(
                            horizontal_index * target_width as i32,
                            vertical_index * target_height as i32,
                            target_width as i32,
                            target_height as i32,
                        ),
                    )
                    .unwrap()
                    .clone_pointee();
                    // imgs_map.insert(
                    //     img_id,
                    //     cropped,
                    // );
                    imgcodecs::imwrite(
                        &format!(
                            "{}\\output\\{}.{}",
                            image_path,
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
            for horizontal_index in 0..horizontal_count {
                for vertical_index in 0..vertical_count {
                    let img_id =
                        format!("{}_rb2lt_{}_{}", img_id, horizontal_index, vertical_index);
                    if invalid_id.read().unwrap().contains(&img_id) {
                        continue;
                    }
                    let cropped = core::Mat::roi(
                        &img,
                        core::Rect::new(
                            img.cols() - (horizontal_index + 1) * target_width as i32,
                            img.rows() - (vertical_index + 1) * target_height as i32,
                            target_width as i32,
                            target_height as i32,
                        ),
                    )
                    .unwrap()
                    .clone_pointee();
                    imgcodecs::imwrite(
                        &format!(
                            "{}\\output\\{}.{}",
                            image_path,
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

    // let cropped_labels = cropped_labels.lock().unwrap();
    // for (label_id, label) in cropped_labels.iter() {
    //     imgcodecs::imwrite(
    //         &format!(
    //             "{}\\output\\{}.{}",
    //             label_path,
    //             label_id,
    //             label_extension.as_ref().unwrap()
    //         ),
    //         label,
    //         &core::Vector::new(),
    //     )
    //     .unwrap();
    //     tracing::info!("Label {} saved", label_id);
    // }

    // let cropped_images = cropped_images.lock().unwrap();
    // for (img_id, img) in cropped_images.iter() {
    //     imgcodecs::imwrite(
    //         &format!(
    //             "{}\\output\\{}.{}",
    //             image_path,
    //             img_id,
    //             image_extension.as_ref().unwrap()
    //         ),
    //         img,
    //         &core::Vector::new(),
    //     )
    //     .unwrap();
    //     tracing::info!("Image {} saved", img_id);
    // }
}
