use std::{
    collections::HashMap,
    fs,
    sync::{Arc, Mutex, RwLock},
};

use image::{Rgb, RgbImage};
use tokio::{sync::Semaphore, task::JoinSet};

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
        if entry
            .path()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .ends_with(".png")
        {
            let dataset_path = dataset_path.clone();
            let bias_step = bias_step.clone();
            let target_height = target_height.clone();
            let target_width = target_width.clone();
            threads.spawn(async move {
                log::info!(
                    "Image {} processing...",
                    entry.path().file_name().unwrap().to_str().unwrap()
                );
                let mut reader = image::io::Reader::open(entry.path()).unwrap();
                reader.no_limits();
                let img = reader.decode().unwrap();
                // if !entry
                //     .path()
                //     .file_name()
                //     .unwrap()
                //     .to_str()
                //     .unwrap()
                //     .starts_with("13")
                // {
                //     return ();
                // }
                // This only useful for cropping the left side black bar
                // if path.contains("images") && !path.starts_with("11") && !path.starts_with("12") {
                //     img = img.crop_imm(18, 0, img.width() - 18, img.height());
                // }
                let vertical_count = img.height() / target_height;
                let horizontal_count = img.width() / target_width;
                let bias_max = horizontal_count.max(vertical_count);
                for bias in 0..bias_max {
                    'outer: for horizontal_index in 0..horizontal_count {
                        'inner: for vertical_index in 0..vertical_count {
                            // These two should always be larger than 0
                            let start_y = horizontal_index * target_height + bias * bias_step;
                            let start_x = vertical_index * target_width + bias * bias_step;
                            if start_x + target_width > img.width() {
                                continue 'inner;
                            }
                            if start_y + target_height > img.height() {
                                continue 'outer;
                            }
                            let cropped_img = img.crop_imm(
                                start_x as u32,
                                start_y as u32,
                                target_width,
                                target_height,
                            );
                            cropped_img
                                .save(format!(
                                    "{}\\..\\output\\{}_LTR_bias{}_h{}_v{}.png",
                                    dataset_path,
                                    entry
                                        .path()
                                        .file_name()
                                        .unwrap()
                                        .to_str()
                                        .unwrap()
                                        .replace(".png", ""),
                                    bias,
                                    horizontal_index,
                                    vertical_index
                                ))
                                .unwrap();
                        }
                    }
                    log::info!(
                        "Image LTR {} bias {} done",
                        entry.path().file_name().unwrap().to_str().unwrap(),
                        bias
                    );
                }

                for bias in 0..bias_max {
                    'outer: for horizontal_index in 0..horizontal_count {
                        'inner: for vertical_index in 0..vertical_count {
                            let start_y = img.height() as i32
                                - (horizontal_index * target_height + bias * bias_step) as i32;
                            let start_x = img.width() as i32
                                - (vertical_index * target_width + bias * bias_step) as i32;
                            // This should never be greater than original image width
                            if start_x < 0 {
                                continue 'inner;
                            }
                            if start_y < 0 {
                                continue 'outer;
                            }
                            let cropped_img = img.crop_imm(
                                start_x as u32,
                                start_y as u32,
                                target_width,
                                target_height,
                            );
                            cropped_img
                                .save(format!(
                                    "{}\\..\\output\\{}_RTL_bias{}_h{}_v{}.png",
                                    dataset_path,
                                    entry
                                        .path()
                                        .file_name()
                                        .unwrap()
                                        .to_str()
                                        .unwrap()
                                        .replace(".png", ""),
                                    bias,
                                    horizontal_index,
                                    vertical_index
                                ))
                                .unwrap();
                        }
                    }
                    log::info!(
                        "Image RTL {} bias {} done",
                        entry.path().file_name().unwrap().to_str().unwrap(),
                        bias
                    );
                }

                log::info!(
                    "Image {} done",
                    entry.path().file_name().unwrap().to_str().unwrap()
                );
            });
        }
    }
    while threads.join_next().await.is_some() {}
}

pub fn check_valid_pixel_count(img: &RgbImage, valid_rgb_list: &Vec<Rgb<u8>>) -> bool {
    let mut count = 0;
    img.enumerate_pixels().for_each(|(_, _, pixel)| {
        if valid_rgb_list.contains(pixel) {
            count += 1;
        }
    });
    if (count as f32) / ((img.width() * img.height()) as f32) > 0.01 {
        println!(
            "Valid ratio {}",
            (count as f32) / ((img.width() * img.height()) as f32)
        );
    }
    (count as f32) / ((img.width() * img.height()) as f32) > 0.01
}

pub async fn split_images_with_filter(
    dataset_path: &String,
    target_height: &u32,
    target_width: &u32,
    valid_rgb_list_str: &String,
) {
    let valid_rgb_list: Arc<RwLock<Vec<Rgb<u8>>>> = Arc::new(RwLock::new(Vec::new()));
    for rgb in valid_rgb_list_str.split(";").into_iter() {
        let valid_rgb_list = Arc::clone(&valid_rgb_list);
        let mut valid_rgb_list = valid_rgb_list.try_write().unwrap();

        let mut rgb_vec: Vec<u8> = vec![];
        for splited in rgb.split(',') {
            let splited = splited.parse::<u8>().unwrap();
            rgb_vec.push(splited);
        }

        let rgb = Rgb([rgb_vec[0], rgb_vec[1], rgb_vec[2]]);
        valid_rgb_list.push(rgb);
    }

    let image_entries = fs::read_dir(format!("{}\\images", dataset_path)).unwrap();
    let label_entries = fs::read_dir(format!("{}\\labels", dataset_path)).unwrap();
    let sem = Arc::new(Semaphore::new(3));
    let cropped_images = Arc::new(Mutex::new(HashMap::<String, RgbImage>::new()));
    let cropped_labels = Arc::new(Mutex::new(HashMap::<String, RgbImage>::new()));
    let mut threads = JoinSet::new();
    for entry in image_entries {
        let entry = entry.unwrap();
        if entry
            .path()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .ends_with(".png")
        {
            let permit = Arc::clone(&sem);
            let cropped_images = Arc::clone(&cropped_images);
            let target_height = target_height.clone();
            let target_width = target_width.clone();
            threads.spawn(async move {
                let _ = permit.acquire().await.unwrap();
                let img_id = entry
                    .path()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .replace(".png", "");
                let img = image::open(entry.path()).unwrap();
                // This is for cropping the left side black bar
                // let img = img.crop_imm(18, 0, img.width(), img.height());

                let vertical_count = img.height() / target_height;
                let horizontal_count = img.width() / target_width;
                let mut imgs_map = HashMap::<String, RgbImage>::new();
                // crop horizentally from left
                for horizontal_index in 0..horizontal_count {
                    // crop vertically from top
                    for vertical_index in 0..vertical_count {
                        imgs_map.insert(
                            format!("{}_lt2rb_{}_{}", img_id, horizontal_index, vertical_index),
                            img.crop_imm(
                                horizontal_index * target_width,
                                vertical_index * target_height,
                                target_width,
                                target_height,
                            )
                            .into_rgb8(),
                        );
                    }
                }
                // crop horizentally from right
                for horizontal_index in 0..horizontal_count {
                    // crop vertically from bottom
                    for vertical_index in 0..vertical_count {
                        imgs_map.insert(
                            format!("{}_rb2lt_{}_{}", img_id, horizontal_index, vertical_index),
                            img.crop_imm(
                                img.width() - (horizontal_index + 1) * target_width,
                                img.height() - (vertical_index + 1) * target_height,
                                target_width,
                                target_height,
                            )
                            .into_rgb8(),
                        );
                    }
                }

                cropped_images.lock().unwrap().extend(imgs_map);
                println!("Image {} process done", img_id)
            });
        }
    }
    while threads.join_next().await.is_some() {}
    for entry in label_entries {
        let entry = entry.unwrap();
        if entry
            .path()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .ends_with(".png")
        {
            let permit = Arc::clone(&sem);
            let cropped_labels = Arc::clone(&cropped_labels);
            let cropped_images = Arc::clone(&cropped_images);

            let target_width = target_width.clone();
            let target_height = target_height.clone();

            let valid_rgb_list = Arc::clone(&valid_rgb_list);
            threads.spawn(async move {
                let _ = permit.acquire().await.unwrap();
                let label_id = entry
                    .path()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .replace(".png", "");
                let img = image::open(entry.path()).unwrap();
                let vertical_count = img.height() / target_height;
                let horizontal_count = img.width() / target_width;
                let mut labels_map = HashMap::<String, RgbImage>::new();
                // crop horizentally from left
                for horizontal_index in 0..horizontal_count {
                    // crop vertically from top
                    for vertical_index in 0..vertical_count {
                        labels_map.insert(
                            format!("{}_lt2rb_{}_{}", label_id, horizontal_index, vertical_index),
                            img.crop_imm(
                                horizontal_index * target_width,
                                vertical_index * target_height,
                                target_width,
                                target_height,
                            )
                            .into_rgb8(),
                        );
                    }
                }
                // crop horizentally from right
                for horizontal_index in 0..horizontal_count {
                    // crop vertically from bottom
                    for vertical_index in 0..vertical_count {
                        labels_map.insert(
                            format!("{}_rb2lt_{}_{}", label_id, horizontal_index, vertical_index),
                            img.crop_imm(
                                img.width() - (horizontal_index + 1) * target_width,
                                img.height() - (vertical_index + 1) * target_height,
                                target_width,
                                target_height,
                            )
                            .into_rgb8(),
                        );
                    }
                }

                let mut useless_img_id = Vec::<String>::new();
                let valid_rgb_list = valid_rgb_list.try_read().unwrap();
                for (label_id, label) in labels_map.iter() {
                    if check_valid_pixel_count(label, &valid_rgb_list) {
                        continue;
                    }
                    useless_img_id.push(label_id.clone());
                }

                let mut cropped_labels = cropped_labels.lock().unwrap();
                let mut cropped_images = cropped_images.lock().unwrap();
                for img_id in useless_img_id {
                    labels_map.remove(&img_id);
                    cropped_images.remove(&img_id);
                }
                cropped_labels.extend(labels_map);
                println!("Label {} process done", label_id)
            });
        }
    }

    while threads.join_next().await.is_some() {}

    println!("Labels process done");

    let cropped_labels = cropped_labels.lock().unwrap();
    for (label_id, label) in cropped_labels.iter() {
        label
            .save(format!(
                "{}\\labels\\output\\{}.png",
                dataset_path, label_id
            ))
            .unwrap();
        println!("Label {} saved", label_id);
    }
    let cropped_images = cropped_images.lock().unwrap();
    for (img_id, img) in cropped_images.iter() {
        img.save(format!("{}\\images\\output\\{}.png", dataset_path, img_id))
            .unwrap();
        println!("Image {} saved", img_id)
    }
}
