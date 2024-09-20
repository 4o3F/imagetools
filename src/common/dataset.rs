use std::{
    collections::HashMap,
    fs,
    sync::{Arc, Mutex},
};

use opencv::{
    core::{self, MatTraitConst},
    imgcodecs,
};
use tokio::{sync::Semaphore, task::JoinSet};
use tracing_unwrap::{OptionExt, ResultExt};

pub async fn split_dataset(dataset_path: &String, train_ratio: &f32) {
    let entries = fs::read_dir(dataset_path).unwrap();
    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(10));
    let result = Arc::new(Mutex::new(Vec::<String>::new()));
    for entry in entries {
        let entry = entry.unwrap();
        let permit = Arc::clone(&sem);
        let result = Arc::clone(&result);
        threads.spawn(async move {
            let _permit = permit.acquire().await.unwrap();
            result.lock().unwrap().push(format!(
                "{}\n",
                entry.path().file_name().unwrap().to_str().unwrap()
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

pub async fn count_classes(dataset_path: &String) {
    let entries = fs::read_dir(dataset_path).unwrap();

    let type_map = Arc::new(Mutex::new(HashMap::<u8, i32>::new()));
    let sem = Arc::new(Semaphore::new(5));
    let mut threads = JoinSet::new();
    for entry in entries {
        let entry = entry.unwrap();
        let type_map = Arc::clone(&type_map);
        let sem = Arc::clone(&sem);
        threads.spawn(async move {
            let _ = sem.acquire().await.unwrap();
            let image = image::open(entry.path()).unwrap();
            let image = image.as_luma8().unwrap();
            tracing::info!("Loaded image: {}", entry.path().display());
            let mut current_img_type_map = HashMap::<u8, i32>::new();
            for (_, _, pixel) in image.enumerate_pixels() {
                if !current_img_type_map.contains_key(&pixel[0]) {
                    current_img_type_map.insert(pixel[0], 1);
                } else {
                    current_img_type_map
                        .insert(pixel[0], current_img_type_map.get(&pixel[0]).unwrap() + 1);
                }
            }

            let mut type_map = type_map.lock().unwrap();
            for (class_id, count) in current_img_type_map.iter() {
                let total_count = type_map.entry(*class_id).or_insert(0);
                *total_count += count;
            }

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

    let sem = Arc::new(Semaphore::new(1));
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
