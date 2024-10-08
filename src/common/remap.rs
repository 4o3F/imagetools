use image::Rgb;
use opencv::{
    core::{Mat, MatTrait, MatTraitConst, Vec3b, Vector, CV_8U, CV_8UC3},
    imgcodecs::{self, imread, imwrite},
};

use std::{
    collections::HashMap,
    fs,
    ops::Deref,
    path::PathBuf,
    sync::{Arc, RwLock},
};
use tokio::{sync::Semaphore, task::JoinSet};
use tracing_unwrap::ResultExt;

use crate::THREAD_POOL;

pub fn remap_color(original_color: &str, new_color: &str, image_path: &String, save_path: &String) {
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

    let original_color_rgb: Rgb<u8> = if original_color_vec.len() != 3 {
        tracing::error!("Malformed original color RGB, please use R,G,B format");
        return;
    } else {
        Rgb([
            original_color_vec[0],
            original_color_vec[1],
            original_color_vec[2],
        ])
    };

    let new_color_rgb: Rgb<u8> = if new_color_vec.len() != 3 {
        tracing::error!("Malformed new color RGB, please use R,G,B format");
        return;
    } else {
        Rgb([new_color_vec[0], new_color_vec[1], new_color_vec[2]])
    };

    let img = image::open(image_path).unwrap();
    let mut img = img.into_rgb8();
    for (_, _, pixel) in img.enumerate_pixels_mut() {
        if *pixel == original_color_rgb {
            *pixel = new_color_rgb
        }
    }
    img.save(save_path).unwrap();

    tracing::info!("{} color remap done!", image_path);
}

pub async fn remap_color_dir(
    original_color: &String,
    new_color: &String,
    path: &String,
    save_path: &String,
) {
    let entries = fs::read_dir(path).unwrap();
    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(
        (*THREAD_POOL.read().expect_or_log("Get pool error")).into(),
    ));
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

    tracing::info!("All color remap done!");
}

pub fn remap_background_color(
    valid_colors: &String,
    new_color: &str,
    image_path: &String,
    save_path: &String,
) {
    let mut original_color_vec: Vec<Vec<u8>> = vec![];
    for valid_color in valid_colors.split(";") {
        let mut color_vec = vec![];
        for splited in valid_color.split(',') {
            let splited = splited
                .parse::<u8>()
                .expect_or_log("Malformed original color RGB, please use R,G,B format");
            color_vec.push(splited);
        }

        original_color_vec.push(color_vec);
    }

    let mut new_color_vec: Vec<u8> = vec![];
    for splited in new_color.split(',') {
        let splited = splited
            .parse::<u8>()
            .expect_or_log("Malformed new color RGB, please use R,G,B format");
        new_color_vec.push(splited);
    }

    if original_color_vec.iter().any(|x| x.len() != 3) || new_color_vec.len() != 3 {
        tracing::error!("Malformed color RGB, please use R,G,B format");
        return;
    }

    let new_color_rgb = [new_color_vec[0], new_color_vec[1], new_color_vec[2]];

    let mut img = imread(&image_path, imgcodecs::IMREAD_COLOR).expect_or_log("Open image error");
    tracing::info!("Image loaded");
    if img.depth() != CV_8U {
        tracing::error!("Image depth is not 8U, not supported");
        return;
    }
    let cols = img.cols();
    let rows = img.rows();

    for y in 0..rows {
        for x in 0..cols {
            let pixel = img.at_2d_mut::<Vec3b>(y, x).unwrap();
            if !original_color_vec.contains(&vec![pixel.0[0], pixel.0[1], pixel.0[2]]) {
                pixel.0 = new_color_rgb;
            }
        }
        tracing::trace!("Y {} done", y);
    }

    imwrite(save_path, &img, &Vector::new()).expect_or_log("Save image error");

    tracing::info!("{} background color remap done!", image_path);
}

pub async fn class2rgb(dataset_path: &String, rgb_list: &str) {
    let mut entries: Vec<PathBuf> = Vec::new();
    let dataset_path = PathBuf::from(dataset_path.as_str());
    if dataset_path.is_file() {
        entries.push(dataset_path.clone());
        fs::create_dir_all(format!(
            "{}\\output\\",
            dataset_path.parent().unwrap().to_str().unwrap()
        ))
        .expect_or_log("Failed to create directory");
    } else {
        entries = fs::read_dir(dataset_path.clone())
            .unwrap()
            .map(|x| x.unwrap().path())
            .collect();
        fs::create_dir_all(format!("{}\\output\\", dataset_path.to_str().unwrap()))
            .expect_or_log("Failed to create directory");
    }

    let transform_map = Arc::new(RwLock::new(HashMap::<u8, Vec3b>::new()));
    {
        // Split RGB list
        for (class_id, rgb) in rgb_list.split(";").enumerate() {
            let mut rgb_vec: Vec<u8> = vec![];
            for splited in rgb.split(',') {
                let splited = splited.parse::<u8>().unwrap();
                rgb_vec.push(splited);
            }

            transform_map.write().unwrap().insert(
                class_id as u8,
                Vec3b::from_array([rgb_vec[0], rgb_vec[1], rgb_vec[2]]),
            );
        }
    }
    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(
        (*THREAD_POOL.read().expect_or_log("Get pool error")).into(),
    ));
    for entry in entries {
        let sem = Arc::clone(&sem);
        let transform_map = Arc::clone(&transform_map);
        threads.spawn(async move {
            let _ = sem.acquire().await.unwrap();
            let img = imread(&entry.to_str().unwrap(), imgcodecs::IMREAD_COLOR).unwrap();

            let rows = img.rows();
            let mut lut: Vec<Vec3b> = Vec::with_capacity(rows as usize);

            for _ in 0..256 {
                lut.push(Vec3b::from_array([0, 0, 0]));
            }
            transform_map.read().unwrap().iter().for_each(|(k, v)| {
                lut[*k as usize] = *v;
            });

            let lut = Mat::from_slice(&lut).unwrap();

            let mut result = Mat::default();
            opencv::core::lut(&img, &lut, &mut result).unwrap();

            tracing::trace!(
                "Write to {}",
                format!(
                    "{}\\output\\{}",
                    entry.parent().unwrap().to_str().unwrap(),
                    entry.file_name().unwrap().to_str().unwrap()
                )
            );
            imwrite(
                format!(
                    "{}\\output\\{}",
                    entry.parent().unwrap().to_str().unwrap(),
                    entry.file_name().unwrap().to_str().unwrap()
                )
                .as_str(),
                &result,
                &Vector::new(),
            )
            .unwrap();
            tracing::info!("{} finished", entry.file_name().unwrap().to_str().unwrap());
        });
    }
    while threads.join_next().await.is_some() {}
    tracing::info!("All done");
    tracing::info!("Saved to {}\\output\\", dataset_path.to_str().unwrap());
}

pub async fn rgb2class(dataset_path: &String, rgb_list: &str) {
    let mut entries: Vec<PathBuf> = Vec::new();
    let dataset_path = PathBuf::from(dataset_path.as_str());
    if dataset_path.is_file() {
        entries.push(dataset_path.clone());
        fs::create_dir_all(format!(
            "{}\\output\\",
            dataset_path.parent().unwrap().to_str().unwrap()
        ))
        .expect_or_log("Failed to create directory");
    } else {
        entries = fs::read_dir(dataset_path.clone())
            .unwrap()
            .map(|x| x.unwrap().path())
            .filter(|x| x.is_file())
            .collect();
        fs::create_dir_all(format!("{}\\output\\", dataset_path.to_str().unwrap()))
            .expect_or_log("Failed to create directory");
    }

    let transform_map = Arc::new(RwLock::new(HashMap::<[u8; 3], u8>::new()));
    {
        // Split RGB list
        for (class_id, rgb) in rgb_list.split(";").enumerate() {
            let mut rgb_vec: Vec<u8> = vec![];
            for splited in rgb.split(',') {
                let splited = splited.parse::<u8>().unwrap();
                rgb_vec.push(splited);
            }

            transform_map
                .write()
                .unwrap()
                .insert([rgb_vec[0], rgb_vec[1], rgb_vec[2]], class_id as u8);
        }
    }

    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(
        (*THREAD_POOL.read().expect_or_log("Get pool error")).into(),
    ));
    for entry in entries {
        let sem = Arc::clone(&sem);
        let transform_map = Arc::clone(&transform_map);

        threads.spawn(async move {
            let _ = sem.acquire().await.unwrap();
            let img = imread(&entry.to_str().unwrap(), imgcodecs::IMREAD_COLOR).unwrap();

            let mut lut =
                Mat::new_rows_cols_with_default(1, 256, CV_8UC3, opencv::core::Scalar::all(0.))
                    .expect_or_log("Create LUT error");

            for i in 0..=255u8 {
                let lut_value = lut
                    .at_2d_mut::<Vec3b>(0, i.into())
                    .expect_or_log("Get LUT value error");

                *lut_value = Vec3b::from_array([i, i, i]);
            }
            transform_map.read().unwrap().iter().for_each(|(k, v)| {
                let lut_value = lut
                    .at_2d_mut::<Vec3b>(0, *v as i32)
                    .expect_or_log("Get LUT value error");
                *lut_value = Vec3b::from_array([k[0], k[1], k[2]]);
            });
            let mut result = Mat::default();
            opencv::core::lut(&img, &lut, &mut result).unwrap();

            tracing::trace!(
                "Write to {}",
                format!(
                    "{}\\output\\{}",
                    entry.parent().unwrap().to_str().unwrap(),
                    entry.file_name().unwrap().to_str().unwrap()
                )
            );
            imwrite(
                format!(
                    "{}\\output\\{}",
                    entry.parent().unwrap().to_str().unwrap(),
                    entry.file_name().unwrap().to_str().unwrap()
                )
                .as_str(),
                &result,
                &Vector::new(),
            )
            .unwrap();

            tracing::info!("{} finished", entry.file_name().unwrap().to_str().unwrap());
        });
    }
    while let Some(result) = threads.join_next().await {
        match result {
            Ok(()) => {}
            Err(e) => {
                tracing::error!("Error {}", e);
                threads.abort_all();
                break;
            }
        }
    }
    tracing::info!("All done");
    tracing::info!("Saved to {}\\output\\", dataset_path.to_str().unwrap());
}
