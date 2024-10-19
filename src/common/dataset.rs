use opencv::{
    core::{self, MatTraitConst},
    imgcodecs::{self, imread, IMREAD_GRAYSCALE},
};
use parking_lot::RwLock;
use rayon::prelude::*;
use rayon_progress::ProgressAdaptor;
use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tokio::{sync::Semaphore, task::JoinSet};
use tracing_unwrap::{OptionExt, ResultExt};

use crate::THREAD_POOL;

// This will generate CSV format dataset list for huggingface dataset lib
pub async fn generate_dataset_csv(dataset_path: &String, train_ratio: &f32) {
    let dataset_path = PathBuf::from(dataset_path);
    if !(dataset_path.join("images").is_dir() && dataset_path.join("labels").is_dir()) {
        tracing::error!(
            "Invalid dataset path: {}, should contain images and labels folders",
            dataset_path.display()
        );
        return;
    }

    let mut images = fs::read_dir(dataset_path.join("images").clone())
        .unwrap()
        .into_iter()
        .map(|x| x.unwrap().path())
        .filter(|x| x.is_file())
        .collect::<Vec<PathBuf>>();

    images.sort_unstable_by(|a, b| a.file_name().cmp(&b.file_name()));

    let mut labels = fs::read_dir(dataset_path.join("labels").clone())
        .unwrap()
        .into_iter()
        .map(|x| x.unwrap().path())
        .filter(|x| x.is_file())
        .collect::<Vec<PathBuf>>();

    labels.sort_unstable_by(|a, b| a.file_name().cmp(&b.file_name()));

    let mut data = Vec::<String>::new();

    for (image, label) in images.iter().zip(labels.iter()) {
        if image.file_stem() != label.file_stem() {
            return tracing::error!(
                "Image and label should have same name, but encountered {}, {}",
                image.display(),
                label.display()
            );
        }

        data.push(format!(
            "{},{}",
            fs::canonicalize(image).unwrap().display(),
            fs::canonicalize(label).unwrap().display()
        ));
    }

    use rand::seq::SliceRandom;
    data.shuffle(&mut rand::thread_rng());

    let train_count = (data.len() as f32 * train_ratio) as i32;
    let mut train_data = data[0..train_count as usize].to_vec();
    let mut val_data = data[train_count as usize..].to_vec();

    train_data.insert(0, "image,label".to_string());
    val_data.insert(0, "image,label".to_string());

    fs::write(dataset_path.join("train.csv"), train_data.join("\n"))
        .expect_or_log("Failed to write");
    fs::write(dataset_path.join("val.csv"), val_data.join("\n")).expect_or_log("Failed to write");
    tracing::info!(
        "Saved to train: {} val: {}",
        dataset_path.join("train.csv").display(),
        dataset_path.join("val.csv").display()
    );
}

pub async fn generate_dataset_list(dataset_path: &String, train_ratio: &f32) {
    // Check dataset_path contain images and labels folder

    let dataset_path = PathBuf::from(dataset_path);
    if !(dataset_path.join("images").is_dir() && dataset_path.join("labels").is_dir()) {
        tracing::error!(
            "Invalid dataset path: {}, should contain images and labels folders",
            dataset_path.display()
        );
        return;
    }

    let dataset_path = dataset_path.join("images");
    let entries = fs::read_dir(dataset_path.clone())
        .unwrap()
        .into_iter()
        .map(|x| x.unwrap().path())
        .collect::<Vec<PathBuf>>();

    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(
        (*THREAD_POOL.read().expect_or_log("Get pool error")).into(),
    ));
    let result = Arc::new(Mutex::new(Vec::<String>::new()));
    for entry in entries {
        let permit = Arc::clone(&sem);
        let result = Arc::clone(&result);
        threads.spawn(async move {
            let _permit = permit.acquire().await.unwrap();
            result.lock().unwrap().push(format!(
                "{}\n",
                entry.file_name().unwrap().to_str().unwrap()
            ));
        });
    }

    while threads.join_next().await.is_some() {}

    let mut data = result.lock().unwrap();
    use rand::seq::SliceRandom;
    data.shuffle(&mut rand::thread_rng());
    let train_count = (data.len() as f32 * train_ratio) as i32;
    let train_data = data[0..train_count as usize].to_vec();
    let valid_data = data[train_count as usize..].to_vec();

    let dataset_path = dataset_path.to_str().unwrap();
    fs::write(
        format!("{}\\..\\val.txt", dataset_path),
        valid_data.concat(),
    )
    .unwrap();
    fs::write(
        format!("{}\\..\\train.txt", dataset_path),
        train_data.concat(),
    )
    .unwrap();
    tracing::info!("Train dataset length: {}", train_data.len());
    tracing::info!("Saved to {}\\..\\train.txt", dataset_path);
    tracing::info!("Valid dataset length: {}", valid_data.len());
    tracing::info!("Saved to {}\\..\\val.txt", dataset_path);
    tracing::info!("Dataset split done");
}

// TODO: rewrite this to make it suitable for all datasets
pub async fn split_dataset(dataset_path: &String, train_ratio: &f32) {
    // Check dataset_path contain images and labels folder

    let dataset_path = PathBuf::from(dataset_path);
    if !(dataset_path.join("images").is_dir() && dataset_path.join("labels").is_dir()) {
        tracing::error!(
            "Invalid dataset path: {}, should contain images and labels folders",
            dataset_path.display()
        );
        return;
    }

    // Create dir
    fs::create_dir_all(dataset_path.join("images").join("train"))
        .expect_or_log("Failed to create directory");
    fs::create_dir_all(dataset_path.join("images").join("val"))
        .expect_or_log("Failed to create directory");

    fs::create_dir_all(dataset_path.join("labels").join("train"))
        .expect_or_log("Failed to create directory");
    fs::create_dir_all(dataset_path.join("labels").join("val"))
        .expect_or_log("Failed to create directory");

    let dataset_path = dataset_path.join("images");
    let entries = fs::read_dir(dataset_path.clone())
        .unwrap()
        .into_iter()
        .map(|x| x.unwrap().path())
        .collect::<Vec<PathBuf>>();

    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(
        (*THREAD_POOL.read().expect_or_log("Get pool error")).into(),
    ));
    let result = Arc::new(Mutex::new(Vec::<PathBuf>::new()));
    for entry in entries {
        let permit = Arc::clone(&sem);
        let result = Arc::clone(&result);
        threads.spawn(async move {
            let _permit = permit.acquire().await.unwrap();
            result.lock().unwrap().push(entry);
        });
    }

    while threads.join_next().await.is_some() {}

    let mut data = result.lock().unwrap();
    use rand::seq::SliceRandom;
    data.shuffle(&mut rand::thread_rng());
    let train_count = (data.len() as f32 * train_ratio) as i32;
    let train_data = data[0..train_count as usize].to_vec();
    let valid_data = data[train_count as usize..].to_vec();

    for entry in train_data {
        if !entry.is_file() {
            continue;
        }
        tracing::info!(
            "Renaming {} to {}",
            &entry.display(),
            &dataset_path
                .join("train")
                .join(entry.file_name().unwrap())
                .display()
        );
        fs::rename(
            &entry,
            dataset_path.join("train").join(entry.file_name().unwrap()),
        )
        .unwrap();

        tracing::info!(
            "Renaming {} to {}",
            &entry
                .to_str()
                .unwrap()
                .to_string()
                .replace("images", "labels")
                .replace(".tif", ".png"),
            &dataset_path
                .parent()
                .unwrap()
                .join("labels")
                .join("train")
                .join(
                    entry
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_string()
                        .replace(".tif", ".png"),
                )
                .display(),
        );
        fs::rename(
            &entry
                .to_str()
                .unwrap()
                .to_string()
                .replace("images", "labels")
                .replace(".tif", ".png"),
            dataset_path
                .parent()
                .unwrap()
                .join("labels")
                .join("train")
                .join(
                    entry
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_string()
                        .replace(".tif", ".png"),
                ),
        )
        .unwrap();
    }

    for entry in valid_data {
        if !entry.is_file() {
            continue;
        }
        tracing::info!(
            "Renaming {} to {}",
            &entry.display(),
            &dataset_path
                .join("val")
                .join(entry.file_name().unwrap())
                .display()
        );
        fs::rename(
            &entry,
            dataset_path.join("val").join(entry.file_name().unwrap()),
        )
        .unwrap();

        tracing::info!(
            "Renaming {} to {}",
            &entry
                .to_str()
                .unwrap()
                .to_string()
                .replace("images", "labels")
                .replace(".tif", ".png"),
            &dataset_path
                .parent()
                .unwrap()
                .join("labels")
                .join("val")
                .join(
                    entry
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_string()
                        .replace(".tif", ".png"),
                )
                .display()
        );

        fs::rename(
            &entry
                .to_str()
                .unwrap()
                .to_string()
                .replace("images", "labels")
                .replace(".tif", ".png"),
            dataset_path
                .parent()
                .unwrap()
                .join("labels")
                .join("val")
                .join(
                    entry
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_string()
                        .replace(".tif", ".png"),
                ),
        )
        .unwrap();
    }
}

pub async fn count_classes(dataset_path: &String) {
    let mut entries: Vec<PathBuf> = Vec::new();
    let dataset_path = PathBuf::from(dataset_path);
    if dataset_path.is_file() {
        entries.push(dataset_path.clone());
    } else {
        entries = fs::read_dir(dataset_path.clone())
            .unwrap()
            .map(|x| x.unwrap().path())
            .collect();
    }

    let type_map = Arc::new(Mutex::new(HashMap::<u8, i32>::new()));
    let sem = Arc::new(Semaphore::new(
        (*THREAD_POOL.read().expect_or_log("Get pool error")).into(),
    ));
    let mut threads = JoinSet::new();
    for entry in entries {
        if entry.is_dir() {
            continue;
        }
        let type_map = Arc::clone(&type_map);
        let sem = Arc::clone(&sem);
        threads.spawn(async move {
            let _ = sem.acquire().await.unwrap();
            let img = imread(entry.to_str().unwrap(), IMREAD_GRAYSCALE).unwrap();
            let row = img.rows();
            let cols = img.cols();
            let img = Arc::new(RwLock::new(img));

            let row_iter = ProgressAdaptor::new(0..row);
            row_iter.for_each(|row_index| {
                let mut row_type_map = HashMap::<u8, i32>::new();
                let row = img.read().row(row_index).unwrap().clone_pointee();
                for col_index in 0..cols {
                    let pixel = row.at_2d::<u8>(0, col_index).unwrap();
                    row_type_map
                        .entry(*pixel)
                        .and_modify(|x| *x += 1)
                        .or_insert(1);
                }

                let mut entry = type_map.lock().unwrap();
                for (class_id, count) in row_type_map.iter() {
                    let total_count = entry.entry(*class_id).or_insert(0);
                    *total_count += count;
                }
            });
            tracing::info!("Image {} done", entry.to_str().unwrap());
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
    tracing::info!("Dataset counted");

    let type_map = type_map.lock().unwrap();
    tracing::info!("Classes counts: {:?}", type_map);

    let total_pixel = type_map.values().sum::<i32>();

    let mut weight_map = HashMap::<u8, f64>::new();
    let class_count = type_map.len() as f64;
    for (class_id, count) in type_map.iter() {
        let class_weight: f64 = f64::from(total_pixel) / (f64::from(*count) * class_count);
        weight_map.insert(*class_id, class_weight);
    }

    tracing::info!("Inverse class weights: {:?}", weight_map);
}

pub async fn calc_mean_std(dataset_path: &String) {
    let entries = fs::read_dir(dataset_path).expect_or_log("Failed to read directory");
    let mut threads = JoinSet::new();

    let mean_map = Arc::new(Mutex::new(HashMap::<usize, Vec<f64>>::new()));
    let std_map = Arc::new(Mutex::new(HashMap::<usize, Vec<f64>>::new()));

    let min_value = Arc::new(Mutex::new(f64::MAX));
    let max_value = Arc::new(Mutex::new(f64::MIN));

    let sem = Arc::new(Semaphore::new(
        (*THREAD_POOL.read().expect_or_log("Get pool error")).into(),
    ));
    for entry in entries {
        let entry = entry.expect_or_log("Failed to iterate entries");
        if !entry.path().is_file() {
            continue;
        }
        let mean_map = Arc::clone(&mean_map);
        let std_map = Arc::clone(&std_map);

        let global_min_value = Arc::clone(&min_value);
        let global_max_value = Arc::clone(&max_value);

        let sem = Arc::clone(&sem);
        threads.spawn(async move {
            let _permit = sem
                .acquire()
                .await
                .expect_or_log("Failed to acquire semaphore");
            let image = imgcodecs::imread(
                entry
                    .path()
                    .to_str()
                    .expect_or_log("Failed to convert path to string"),
                imgcodecs::IMREAD_UNCHANGED,
            )
            .expect_or_log("Failed to read image");

            let mut mean = core::Vector::<f64>::new();
            let mut stddev = core::Vector::<f64>::new();
            if image.empty() {
                tracing::error!("Image {} is empty", entry.path().display());
                return;
            }
            core::mean_std_dev(&image, &mut mean, &mut stddev, &core::no_array())
                .expect_or_log("Failed to calculate mean & std dev of mat");
            let mut min_value = 0.0;
            let mut max_value = 0.0;
            core::min_max_loc(
                &image,
                Some(&mut min_value),
                Some(&mut max_value),
                None,
                None,
                &core::no_array(),
            )
            .expect_or_log("Failed to find min & max value of mat");

            let mut mean_map = mean_map.lock().expect_or_log("Failed to lock mean map");
            let mut std_map = std_map.lock().expect_or_log("Failed to lock std map");
            for i in 1..=mean.len() {
                if !mean_map.contains_key(&i) {
                    mean_map.insert(i, Vec::<f64>::new());
                    std_map.insert(i, Vec::<f64>::new());
                }
                mean_map
                    .get_mut(&i)
                    .unwrap()
                    .push(mean.get(i - 1).expect_or_log("Failed to get mean value"));
                std_map.get_mut(&i).unwrap().push(
                    stddev
                        .get(i - 1)
                        .expect_or_log("Failed to get stddev value"),
                );
            }

            let mut global_min_value = global_min_value
                .lock()
                .expect_or_log("Failed to lock min value");
            let mut global_max_value = global_max_value
                .lock()
                .expect_or_log("Failed to lock max value");

            *global_min_value = f64::min(*global_min_value, min_value);
            *global_max_value = f64::max(*global_max_value, max_value);

            tracing::info!("Image {} done", entry.path().display());
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

    let mean_map = mean_map.lock().expect_or_log("Failed to lock mean map");
    let std_map = std_map.lock().expect_or_log("Failed to lock std map");

    assert_eq!(mean_map.len(), std_map.len());

    let mut mean_vec: Vec<f64> = vec![0.0; mean_map.len()];
    for (i, mean) in mean_map.iter() {
        let mean = mean.iter().sum::<f64>() / mean.len() as f64;
        *mean_vec
            .get_mut(*i - 1)
            .expect_or_log("Failed to get mean value") = mean;
    }

    let mut std_vec: Vec<f64> = vec![0.0; std_map.len()];
    for (i, std) in std_map.iter() {
        let std = std.iter().sum::<f64>() / std.len() as f64;
        *std_vec
            .get_mut(*i - 1)
            .expect_or_log("Failed to get std value") = std;
    }

    let min_value = min_value.lock().expect_or_log("Failed to lock min value");
    let max_value = max_value.lock().expect_or_log("Failed to lock max value");

    tracing::info!("Mean: {:?}", mean_vec);
    tracing::info!("Std: {:?}", std_vec);
    tracing::info!("Min: {:?}", *min_value);
    tracing::info!("Max: {:?}", *max_value);
    tracing::info!("Dataset calculated");
}
