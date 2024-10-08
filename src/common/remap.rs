use opencv::{
    core::{Mat, MatTrait, MatTraitConst, Vec3b, Vector},
    imgcodecs::{self, imread, imwrite},
    imgproc::COLOR_BGR2GRAY,
};
use rayon::prelude::*;
use rayon_progress::ProgressAdaptor;

use parking_lot::RwLock;
use std::{collections::HashMap, fs, path::PathBuf, sync::Arc};
use tokio::{sync::Semaphore, task::JoinSet};
use tracing_unwrap::{OptionExt, ResultExt};

use crate::THREAD_POOL;

pub async fn remap_color(original_color: &str, new_color: &str, dataset_path: &String) {
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

    if original_color_vec.len() != 3 || new_color_vec.len() != 3 {
        tracing::error!("Malformed color RGB, please use R,G,B format");
        return;
    }

    let mut entries: Vec<PathBuf> = Vec::new();
    let dataset_path = PathBuf::from(dataset_path);
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

    let mut threads = JoinSet::new();
    let semaphore = Arc::new(Semaphore::new(
        (*THREAD_POOL.read().expect_or_log("Get pool error")).into(),
    ));
    for entry in entries {
        let entry = entry.clone();
        let sem = semaphore.clone();
        let original_color = original_color_vec.clone();
        let new_color = new_color_vec.clone();

        threads.spawn(async move {
            let _ = sem.acquire().await.unwrap();
            let img = imread(&entry.to_str().unwrap(), imgcodecs::IMREAD_COLOR).unwrap();
            if img.channels() != 3 {
                tracing::error!(
                    "Image {} is not RGB",
                    entry.file_name().unwrap().to_str().unwrap()
                );
                return Err(());
            }

            let rows = img.rows();
            let cols = img.cols();
            let row_iter = ProgressAdaptor::new(0..rows);
            let row_progress = row_iter.items_processed();
            let img = Arc::new(RwLock::new(img));

            row_iter.for_each(|row_index| {
                let mut row = img.read().row(row_index).unwrap().clone_pointee();
                for col_index in 0..cols {
                    let pixel = row.at_2d_mut::<Vec3b>(0, col_index).unwrap();
                    if pixel[0] == original_color[0]
                        && pixel[1] == original_color[1]
                        && pixel[2] == original_color[2]
                    {
                        pixel[0] = new_color[0];
                        pixel[1] = new_color[1];
                        pixel[2] = new_color[2];
                    }
                }

                let mut original_row = img.write();
                let mut original_row = original_row.row_mut(row_index).unwrap();

                row.copy_to(&mut original_row)
                    .expect_or_log("Copy row error");

                tracing::trace!("Row {} done", row_progress.get());
                if row_progress.get() != 0 && row_progress.get() % 100 == 0 {
                    tracing::info!("Row {} done", row_progress.get());
                }
            });

            imwrite(
                format!(
                    "{}\\output\\{}",
                    entry.parent().unwrap().to_str().unwrap(),
                    entry.file_name().unwrap().to_str().unwrap()
                )
                .as_str(),
                &*img.read(),
                &Vector::new(),
            )
            .expect_or_log("Failed to save image");

            tracing::info!("{} finished", entry.file_name().unwrap().to_str().unwrap());
            Ok(())
        });
    }

    while let Some(res) = threads.join_next().await {
        if let Err(err) = res {
            tracing::error!("{:?}", err);
            break;
        }
    }

    tracing::info!("All done!");
}
pub async fn remap_background_color(valid_colors: &str, new_color: &str, dataset_path: &String) {
    let mut valid_color_vec: Vec<Vec<u8>> = vec![];
    for valid_color in valid_colors.split(";") {
        let mut color_vec = vec![];
        for splited in valid_color.split(',') {
            let splited = splited
                .parse::<u8>()
                .expect_or_log("Malformed original color RGB, please use R,G,B format");
            color_vec.push(splited);
        }

        valid_color_vec.push(color_vec);
    }

    let mut new_color_vec: Vec<u8> = vec![];
    for splited in new_color.split(',') {
        let splited = splited.parse::<u8>().unwrap();
        new_color_vec.push(splited);
    }

    if valid_color_vec.iter().any(|x| x.len() != 3) || new_color_vec.len() != 3 {
        tracing::error!("Malformed color RGB, please use R,G,B format");
        return;
    }

    let mut entries: Vec<PathBuf> = Vec::new();
    let dataset_path = PathBuf::from(dataset_path);
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

    let mut threads = JoinSet::new();
    let semaphore = Arc::new(Semaphore::new(
        (*THREAD_POOL.read().expect_or_log("Get pool error")).into(),
    ));
    for entry in entries {
        let entry = entry.clone();
        let sem = semaphore.clone();
        let valid_colors = valid_color_vec.clone();
        let new_color = new_color_vec.clone();

        threads.spawn(async move {
            let _ = sem.acquire().await.unwrap();
            let img = imread(&entry.to_str().unwrap(), imgcodecs::IMREAD_COLOR).unwrap();
            if img.channels() != 3 {
                tracing::error!(
                    "Image {} is not RGB",
                    entry.file_name().unwrap().to_str().unwrap()
                );
                return Err(());
            }

            let rows = img.rows();
            let cols = img.cols();
            let row_iter = ProgressAdaptor::new(0..rows);
            let row_progress = row_iter.items_processed();
            let img = Arc::new(RwLock::new(img));

            row_iter.for_each(|row_index| {
                let mut row = img.read().row(row_index).unwrap().clone_pointee();
                for col_index in 0..cols {
                    let pixel = row.at_2d_mut::<Vec3b>(0, col_index).unwrap();

                    if !valid_colors.contains(&vec![pixel[0], pixel[1], pixel[2]]) {
                        pixel[0] = new_color[0];
                        pixel[1] = new_color[1];
                        pixel[2] = new_color[2];
                    }
                }

                let mut original_row = img.write();
                let mut original_row = original_row.row_mut(row_index).unwrap();

                row.copy_to(&mut original_row)
                    .expect_or_log("Copy row error");

                tracing::trace!("Row {} done", row_progress.get());
                if row_progress.get() != 0 && row_progress.get() % 100 == 0 {
                    tracing::info!("Row {} done", row_progress.get());
                }
            });

            imwrite(
                format!(
                    "{}\\output\\{}",
                    entry.parent().unwrap().to_str().unwrap(),
                    entry.file_name().unwrap().to_str().unwrap()
                )
                .as_str(),
                &*img.read(),
                &Vector::new(),
            )
            .expect_or_log("Failed to save image");

            tracing::info!("{} finished", entry.file_name().unwrap().to_str().unwrap());
            Ok(())
        });
    }

    while let Some(res) = threads.join_next().await {
        if let Err(err) = res {
            tracing::error!("{:?}", err);
            break;
        }
    }

    tracing::info!("All done!");
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

            transform_map.write().insert(
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
            transform_map.read().iter().for_each(|(k, v)| {
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
                .insert([rgb_vec[0], rgb_vec[1], rgb_vec[2]], class_id as u8);
        }
    }

    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(
        (*THREAD_POOL.read().expect_or_log("Get pool error")).into(),
    ));

    tracing::trace!("Processing {} images", entries.len());
    for entry in entries {
        let sem = Arc::clone(&sem);
        let transform_map = Arc::clone(&transform_map);

        threads.spawn(async move {
            let _ = sem.acquire().await.unwrap();
            let img = imread(&entry.to_str().unwrap(), imgcodecs::IMREAD_COLOR).unwrap();
            let rows = img.rows();
            let cols = img.cols();
            let row_iter = ProgressAdaptor::new(0..rows);
            let row_progress = row_iter.items_processed();
            let img = Arc::new(RwLock::new(img));

            row_iter.for_each(|row_index| {
                let mut row = img.read().row(row_index).unwrap().clone_pointee();
                for col_index in 0..cols {
                    let pixel = row.at_2d_mut::<Vec3b>(0, col_index).unwrap();
                    if transform_map
                        .read()
                        .contains_key(&[pixel[0], pixel[1], pixel[2]])
                    {
                        let new_color = transform_map.read();
                        let new_color = new_color.get(&[pixel[0], pixel[1], pixel[2]]).unwrap();
                        pixel[0] = *new_color;
                        pixel[1] = *new_color;
                        pixel[2] = *new_color;
                    }
                }

                let mut original_row = img.write();
                let mut original_row = original_row.row_mut(row_index).unwrap();

                row.copy_to(&mut original_row)
                    .expect_or_log("Copy row error");

                tracing::trace!("Row {} done", row_progress.get());
                if row_progress.get() != 0 && row_progress.get() % 100 == 0 {
                    tracing::info!("Row {} done", row_progress.get());
                }
            });

            let mut gray_result = Mat::default();
            opencv::imgproc::cvt_color(&*img.read(), &mut gray_result, COLOR_BGR2GRAY, 0)
                .expect_or_log("Cvt color to gray error");

            imwrite(
                format!(
                    "{}\\output\\{}",
                    entry.parent().unwrap().to_str().unwrap(),
                    entry.file_name().unwrap().to_str().unwrap()
                )
                .as_str(),
                &gray_result,
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
    tracing::info!(
        "Saved to {}\\output\\",
        dataset_path
            .parent()
            .expect_or_log("Get parent error")
            .to_str()
            .unwrap()
    );
}
