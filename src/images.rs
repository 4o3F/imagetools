use image::{imageops::FilterType, Rgb, RgbImage};
use std::{
    collections::HashMap,
    fs,
    sync::{Arc, Mutex},
};
use tokio::{sync::Semaphore, task::JoinSet};
pub fn check_valid_pixel_count(img: &RgbImage) -> bool {
    let valid_rgb = [
        // Rgb([0, 0, 0]),
        Rgb([0, 255, 0]),
        Rgb([0, 255, 255]),
        Rgb([0, 0, 255]),
        Rgb([255, 255, 255]),
        Rgb([255, 0, 255]),
        Rgb([255, 255, 0]),
    ];
    let mut count = 0;
    img.enumerate_pixels().for_each(|(_, _, pixel)| {
        if valid_rgb.contains(pixel) {
            count += 1;
        }
    });
    println!(
        "Valid pixels {}, total pixel {}, ratio {}",
        count,
        img.width() * img.height(),
        (count as f32) / ((img.width() * img.height()) as f32)
    );
    (count as f32) / ((img.width() * img.height()) as f32) > 0.01
}

fn calc_normalize_result(img: &RgbImage, x: usize, y: usize) -> Rgb<u8> {
    let taps: &[(isize, isize)] = &[(0, -1), (-1, 0), (1, 0), (0, 1)];

    let central_color = img.get_pixel(x as u32, y as u32);
    let mut target_color = *central_color;
    for (x_offset, y_offset) in taps.iter() {
        let x = x as isize + x_offset;
        let y = y as isize + y_offset;
        if x < 0 || y < 0 || x >= img.width() as isize || y >= img.height() as isize {
            continue;
        }
        let pixel = img.get_pixel(x as u32, y as u32);

        for i in 0..3 {
            if (pixel[i] as i32 - central_color[i] as i32).abs() < 50 {
                target_color[i] = pixel[i];
            }
        }
    }

    if target_color[0] > 128 {
        target_color[0] = 255;
    } else {
        target_color[0] = 0;
    }

    if target_color[1] > 128 {
        target_color[1] = 255;
    } else {
        target_color[1] = 0;
    }

    if target_color[2] > 128 {
        target_color[2] = 255;
    } else {
        target_color[2] = 0;
    }

    target_color
}

#[allow(dead_code)]
pub async fn normalize_color() {
    let path = "E:\\卫星数据\\labels";
    let entries = fs::read_dir(path).unwrap();
    let mut threads = JoinSet::new();
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
            threads.spawn(async move {
                println!(
                    "Image {} processing...",
                    entry.path().file_name().unwrap().to_str().unwrap()
                );
                let img = image::open(entry.path()).unwrap();
                let img = img.resize_exact(7300, 6908, FilterType::Nearest);
                let img = img.crop_imm(18, 0, img.width() - 18, img.height());
                let mut img = img.into_rgb8();
                for x in 0..img.width() {
                    for y in 0..img.height() {
                        let result = calc_normalize_result(&img, x as usize, y as usize);
                        img.put_pixel(x, y, result);
                    }
                    // println!("{}", x)
                }

                img.save(format!(
                    "{}\\output\\{}",
                    path,
                    entry.path().file_name().unwrap().to_str().unwrap()
                ))
                .unwrap();
                println!(
                    "Image {} done",
                    entry.path().file_name().unwrap().to_str().unwrap()
                );
            });
        }
    }
    while threads.join_next().await.is_some() {}
}

#[allow(dead_code)]
pub async fn split_images() {
    let path = "E:\\卫星数据";
    let image_entries = fs::read_dir(format!("{}\\images", path)).unwrap();
    let label_entries = fs::read_dir(format!("{}\\labels", path)).unwrap();
    let sem = Arc::new(Semaphore::new(3));
    let cropped_images = Arc::new(Mutex::new(HashMap::<String, RgbImage>::new()));
    let cropped_labels = Arc::new(Mutex::new(HashMap::<String, RgbImage>::new()));
    let mut threads = JoinSet::new();
    let length = 768;
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
                let img = img.crop_imm(18, 0, img.width(), img.height());

                let vertical_count = img.height() / length;
                let horizontal_count = img.width() / length;
                let mut imgs_map = HashMap::<String, RgbImage>::new();
                // crop horizentally from left
                for horizontal_index in 0..horizontal_count {
                    // crop vertically from top
                    for vertical_index in 0..vertical_count {
                        imgs_map.insert(
                            format!("{}_lt2rb_{}_{}", img_id, horizontal_index, vertical_index),
                            img.crop_imm(
                                horizontal_index * length,
                                vertical_index * length,
                                length,
                                length,
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
                                img.width() - (horizontal_index + 1) * length,
                                img.height() - (vertical_index + 1) * length,
                                length,
                                length,
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
    println!("Images process done");
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
                let vertical_count = img.height() / length;
                let horizontal_count = img.width() / length;
                let mut labels_map = HashMap::<String, RgbImage>::new();
                // crop horizentally from left
                for horizontal_index in 0..horizontal_count {
                    // crop vertically from top
                    for vertical_index in 0..vertical_count {
                        labels_map.insert(
                            format!("{}_lt2rb_{}_{}", label_id, horizontal_index, vertical_index),
                            img.crop_imm(
                                horizontal_index * length,
                                vertical_index * length,
                                length,
                                length,
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
                                img.width() - (horizontal_index + 1) * length,
                                img.height() - (vertical_index + 1) * length,
                                length,
                                length,
                            )
                            .into_rgb8(),
                        );
                    }
                }

                let mut useless_img_id = Vec::<String>::new();
                for (label_id, label) in labels_map.iter() {
                    if check_valid_pixel_count(label) {
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
            .save(format!("{}\\labels\\output\\{}.png", path, label_id))
            .unwrap();
        println!("Label {} saved", label_id);
    }
    let cropped_images = cropped_images.lock().unwrap();
    for (img_id, img) in cropped_images.iter() {
        img.save(format!("{}\\images\\output\\{}.png", path, img_id))
            .unwrap();
        println!("Image {} saved", img_id)
    }
}

#[allow(dead_code)]
async fn crop_images() {
    let path = "D:\\Documents\\Competitions\\ChallengeCup\\NWPU_YRCC2\\datasets\\labels";
    let entries = fs::read_dir(path).unwrap();
    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(10));
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
            let permit = Arc::clone(&sem);
            threads.spawn(async move {
                let _permit = permit.acquire().await.unwrap();
                let img = image::open(entry.path()).unwrap();
                let img_1 = img.crop_imm(0, 0, 640, 640);
                let img_2 = img.crop_imm(480, 0, 640, 640);
                let img_3 = img.crop_imm(960, 0, 640, 640);
                img_1
                    .save(format!(
                        "{}\\output\\{}",
                        path,
                        entry
                            .path()
                            .file_name()
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .replace(".png", "_1.png")
                    ))
                    .unwrap();
                img_2
                    .save(format!(
                        "{}\\output\\{}",
                        path,
                        entry
                            .path()
                            .file_name()
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .replace(".png", "_2.png")
                    ))
                    .unwrap();
                img_3
                    .save(format!(
                        "{}\\output\\{}",
                        path,
                        entry
                            .path()
                            .file_name()
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .replace(".png", "_3.png")
                    ))
                    .unwrap();
            });
        }
    }
    while threads.join_next().await.is_some() {}
}

#[allow(dead_code)]
async fn resize_label() {
    let path = "D:\\Documents\\Competitions\\ChallengeCup\\NWPU_YRCC2\\datasets\\labels";
    let entries = fs::read_dir(path).unwrap();
    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(10));

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
            // print!("Image {} started\n", entry.path().file_name().unwrap().to_str().unwrap());
            let permit = Arc::clone(&sem);

            threads.spawn(async move {
                let _permit = permit.acquire().await.unwrap();

                let img = image::open(entry.path()).unwrap();
                let img_resized = img.resize_exact(1024, 1024, FilterType::Nearest);
                img_resized
                    .save(format!(
                        "{}\\output\\{}",
                        path,
                        entry.path().file_name().unwrap().to_str().unwrap() // .replace(".jpg", ".png")
                    ))
                    .unwrap();

                // print!("Image {} done\n", entry.path().file_name().unwrap().to_str().unwrap());
            });
        }
    }
    while threads.join_next().await.is_some() {}
}
