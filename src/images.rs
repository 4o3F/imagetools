use image::{imageops::FilterType, Rgb, RgbImage};
use std::{
    collections::HashMap,
    fs, path,
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
    if (count as f32) / ((img.width() * img.height()) as f32) > 0.01 {
        println!(
            "Valid ratio {}",
            (count as f32) / ((img.width() * img.height()) as f32)
        );
    }
    (count as f32) / ((img.width() * img.height()) as f32) > 0.01
}

#[allow(dead_code)]
pub async fn valid_image_list() {
    let path = "E:\\卫星数据\\labels\\label_outputs";
    let entries = fs::read_dir(path).unwrap();
    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(10));
    let result = Arc::new(Mutex::new(Vec::<String>::new()));
    for entry in entries {
        let entry = entry.unwrap();
        if entry
            .path()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .ends_with(".txt")
        {
            let permit = Arc::clone(&sem);
            let result = Arc::clone(&result);
            threads.spawn(async move {
                let _permit = permit.acquire().await.unwrap();
                result.lock().unwrap().push(format!(
                    "./images/{}\n",
                    entry
                        .path()
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_string().replace(".txt", ".png")
                ));
            });
        }
    }

    while threads.join_next().await.is_some() {}

    let mut data = result.lock().unwrap();
    use rand::seq::SliceRandom;
    data.shuffle(&mut rand::thread_rng());
    let train_count = (data.len() as f32 * 0.8) as i32;
    let train_data = data[0..train_count as usize].to_vec();
    let valid_data = data[train_count as usize..].to_vec();

    fs::write(format!("{}\\..\\val.txt", path), valid_data.concat()).unwrap();
    fs::write(format!("{}\\..\\train.txt", path), train_data.concat()).unwrap();
}

#[allow(dead_code)]
pub async fn flip_image() {
    let path = "E:\\卫星数据\\images\\output";
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
            && entry
                .path()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("13")
        {
            let permit = Arc::clone(&sem);
            threads.spawn(async move {
                let _permit = permit.acquire().await.unwrap();
                let img = image::open(entry.path()).unwrap();
                let img_flippedh = img.fliph();
                img_flippedh
                    .save(format!(
                        "{}\\{}",
                        path,
                        entry
                            .path()
                            .file_name()
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .to_string()
                            .replace(".png", "_flipped_h.png")
                    ))
                    .unwrap();

                let img_flippedv = img.flipv();
                img_flippedv
                    .save(format!(
                        "{}\\{}",
                        path,
                        entry
                            .path()
                            .file_name()
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .to_string()
                            .replace(".png", "_flipped_v.png")
                    ))
                    .unwrap();
            });
        }
        while threads.join_next().await.is_some() {}
    }
}

const satellite_rgb: [Rgb<u8>; 5] = [
    Rgb([0, 0, 0]),       // 黑
    Rgb([0, 255, 0]),     // 绿
    Rgb([0, 255, 255]),   // 浅蓝
    Rgb([0, 0, 255]),     // 蓝
    Rgb([255, 255, 255]), // 白
    // Rgb([255, 0, 255]),   // 紫
    // Rgb([255, 255, 0]),   // 黄
];

fn calc_normalize_result(img: &RgbImage, x: usize, y: usize) -> Rgb<u8> {
    let source = img.get_pixel(x as u32, y as u32);
    let mut similarities = satellite_rgb
        .iter()
        .map(|&rgb| {
            let source = lab::rgb_bytes_to_labs(&source.0);
            let target = lab::rgb_bytes_to_labs(&rgb.0);
            let source = source.get(0).unwrap();
            let target = target.get(0).unwrap();
            let delta = (source.l - target.l).powi(2)
                + (source.a - target.a).powi(2)
                + (source.b - target.b).powi(2);
            delta.sqrt()
        })
        .collect::<Vec<f32>>();
    similarities[0] = similarities.get(0).unwrap() * 1.2;
    let min_index = similarities
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    let result = satellite_rgb.get(min_index).unwrap();
    *result
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
                // let img = img.resize_exact(7300, 6908, FilterType::Nearest);
                let img = img.resize_exact(8816,9306, FilterType::Nearest);
                // let img = img.crop_imm(18, 0, img.width() - 18, img.height());
                let mut img = img.into_rgb8();
                for x in 0..img.width() {
                    for y in 0..img.height() {
                        if x % 1000 == 0 && y % 1000 == 0 {
                            println!("{} {}", x, y)
                        }
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
                let mut img = image::open(entry.path()).unwrap();
                // This only useful for cropping the left side black bar
                if path.ends_with("images") && !path.starts_with("11") && !path.starts_with("12") {
                    img = img.crop_imm(18, 0, img.width() - 18, img.height());
                }
                let vertical_count = img.height() / 768;
                let horizontal_count = img.width() / 768;
                for bias in 0..3 {
                    'outer: for horizontal_index in 0..horizontal_count {
                        'inner: for vertical_index in 0..vertical_count {
                            let start_y = (horizontal_index * 768 + bias * 300) as i32;
                            let start_x = (vertical_index * 768 + bias * 300) as i32;
                            if start_x < 0 || start_x + 768 > img.width() as i32 {
                                continue 'inner;
                            }
                            if start_y < 0 || start_y + 768 > img.height() as i32 {
                                continue 'outer;
                            }
                            let cropped_img =
                                img.crop_imm(start_x as u32, start_y as u32, 768, 768);
                            cropped_img
                                .save(format!(
                                    "{}\\output\\{}_LTR_bias{}_h{}_v{}.png",
                                    path,
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
                    println!(
                        "Image LTR {} bias {} done",
                        entry.path().file_name().unwrap().to_str().unwrap(),
                        bias
                    );
                }

                for bias in 0..4 {
                    'outer: for horizontal_index in 0..horizontal_count {
                        'inner: for vertical_index in 0..vertical_count {
                            let start_y =
                                img.height() as i32 - (horizontal_index * 768 + bias * 300) as i32;
                            let start_x =
                                img.width() as i32 - (vertical_index * 768 + bias * 300) as i32;
                            if start_x < 0 || start_x + 768 > img.width() as i32 {
                                continue 'inner;
                            }
                            if start_y < 0 || start_y + 768 > img.height() as i32 {
                                continue 'outer;
                            }
                            let cropped_img =
                                img.crop_imm(start_x as u32, start_y as u32, 768, 768);
                            cropped_img
                                .save(format!(
                                    "{}\\output\\{}_RTL_bias{}_h{}_v{}.png",
                                    path,
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
                    println!(
                        "Image RTL {} bias {} done",
                        entry.path().file_name().unwrap().to_str().unwrap(),
                        bias
                    );
                }

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
pub async fn split_images_with_filter() {
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
pub async fn resize_images() {
    // let path = "D:\\Documents\\Competitions\\ChallengeCup\\NWPU_YRCC2\\datasets\\labels";
    let path = "E:\\YRCC_ori\\yolo\\images";
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
                let img_resized = img.resize_exact(768, 768, FilterType::Lanczos3);
                img_resized
                    .save(format!(
                        "{}\\..\\resize_output\\{}",
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
