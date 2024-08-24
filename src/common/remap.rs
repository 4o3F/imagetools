use std::{
    collections::HashMap,
    fs,
    ops::Deref,
    sync::{Arc, RwLock},
};

use image::{Luma, Rgb, RgbImage};
use tokio::{sync::Semaphore, task::JoinSet};

pub fn remap_color(
    original_color: &String,
    new_color: &String,
    image_path: &String,
    save_path: &String,
) {
    let original_color_rgb: Rgb<u8>;
    let new_color_rgb: Rgb<u8>;

    let mut original_color_vec: Vec<u8> = vec![];
    for splited in original_color.split(',') {
        let splited = splited.parse::<u8>().unwrap();
        original_color_vec.push(splited);
    }

    let mut new_color_vec: Vec<u8> = vec![];
    for splited in new_color.split(',') {
        let splited = splited.parse::<u8>().unwrap();
        new_color_vec.push(splited);
    }

    if original_color_vec.len() != 3 {
        log::error!("Malformed original color RGB, please use R,G,B format");
        return;
    } else {
        original_color_rgb = Rgb([
            original_color_vec[0],
            original_color_vec[1],
            original_color_vec[2],
        ]);
    }

    if new_color_vec.len() != 3 {
        log::error!("Malformed new color RGB, please use R,G,B format");
        return;
    } else {
        new_color_rgb = Rgb([new_color_vec[0], new_color_vec[1], new_color_vec[2]]);
    }

    let img = image::open(&image_path).unwrap();
    let mut img = img.into_rgb8();
    for (_, _, pixel) in img.enumerate_pixels_mut() {
        if *pixel == original_color_rgb {
            *pixel = new_color_rgb
        }
    }
    img.save(save_path).unwrap();

    log::info!("{} color remap done!", image_path);
}

pub async fn remap_color_dir(
    original_color: &String,
    new_color: &String,
    path: &String,
    save_path: &String,
) {
    let entries = fs::read_dir(path).unwrap();
    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(10));
    let original_color = Arc::new(original_color.deref().to_string());
    let new_color = Arc::new(new_color.deref().to_string());
    let path = Arc::new(path.deref().to_string());
    let save_path = Arc::new(save_path.deref().to_string());

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
            let original_color = Arc::clone(&original_color);
            let new_color = Arc::clone(&new_color);
            let path = Arc::clone(&path);
            let save_path = Arc::clone(&save_path);

            threads.spawn(async move {
                let _permit = permit.acquire().await.unwrap();
                remap_color(
                    &original_color,
                    &new_color,
                    &format!(
                        "{}\\{}",
                        path,
                        entry.path().file_name().unwrap().to_str().unwrap()
                    ),
                    &format!(
                        "{}\\{}",
                        save_path,
                        entry.path().file_name().unwrap().to_str().unwrap()
                    ),
                );
            });
        }
    }

    while threads.join_next().await.is_some() {}

    log::info!("All color remap done!");
}

pub async fn class2rgb(dataset_path: &String, rgb_list: &String) {
    let entries = fs::read_dir(dataset_path).unwrap();

    let transform_map = Arc::new(RwLock::new(HashMap::<Luma<u8>, Rgb<u8>>::new()));
    {
        // Split RGB list
        let mut class_id = 0u8;
        for rgb in rgb_list.split(";").into_iter() {
            let mut rgb_vec: Vec<u8> = vec![];
            for splited in rgb.split(',') {
                let splited = splited.parse::<u8>().unwrap();
                rgb_vec.push(splited);
            }

            let rgb = Rgb([rgb_vec[0], rgb_vec[1], rgb_vec[2]]);
            let gray = Luma([class_id]);
            transform_map.write().unwrap().insert(gray, rgb);
            class_id += 1;
        }
    }
    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(10));
    for entry in entries {
        let entry = entry.unwrap();
        let sem = Arc::clone(&sem);
        let transform_map = Arc::clone(&transform_map);
        let dataset_path = dataset_path.clone();
        threads.spawn(async move {
            let _ = sem.acquire().await.unwrap();
            let img = image::open(entry.path()).unwrap();
            let original_img = img.into_luma8();
            let mut mapped_img = RgbImage::new(original_img.width(), original_img.height());
            for ((original_x, original_y, original_pixel), (mapped_x, mapped_y, mapped_pixel)) in
                original_img
                    .enumerate_pixels()
                    .zip(mapped_img.enumerate_pixels_mut())
            {
                if original_x != mapped_x || original_y != mapped_y {
                    log::error!("Pixel coordinate mismatch");
                    return ();
                }
                let Luma([g]) = original_pixel;
                let transform_map = transform_map.read().unwrap();
                let new_color = transform_map.get(&Luma([*g])).unwrap();
                let Rgb([r, g, b]) = mapped_pixel;
                *r = new_color.0[0];
                *g = new_color.0[1];
                *b = new_color.0[2];
            }
            mapped_img
                .save(format!(
                    "{}\\..\\output\\{}",
                    dataset_path,
                    entry.file_name().into_string().unwrap()
                ))
                .unwrap();
            log::info!("{} finished", entry.file_name().into_string().unwrap());
        });
    }
    while threads.join_next().await.is_some() {}
}
