use std::{
    collections::HashMap,
    fs,
    sync::{Arc, Mutex},
};

use opencv::{core::{self, MatTraitConst}, imgcodecs};
use tokio::{sync::Semaphore, task::JoinSet};

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
    log::info!("Train dataset length: {}", train_data.len());
    log::info!("Saved to {}\\..\\train.txt", dataset_path);
    log::info!("Valid dataset length: {}", valid_data.len());
    log::info!("Saved to {}\\..\\val.txt", dataset_path);
    log::info!("Dataset split done");
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
            log::info!("Loaded image: {}", entry.path().display());
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

            log::info!("Image {} done", entry.path().display());
        });
    }

    while let Some(result) = threads.join_next().await {
        match result {
            Ok(()) => {}
            Err(e) => {
                log::error!("Error {}", e);
                threads.abort_all();
                break;
            }
        }
    }
    log::info!("Dataset counted");

    let type_map = type_map.lock().unwrap();
    log::info!("Classes counts: {:?}", type_map);

    let total_pixel = type_map.values().sum::<i32>();

    let mut weight_map = HashMap::<u8, f64>::new();
    let class_count = type_map.len() as f64;
    for (class_id, count) in type_map.iter() {
        let class_weight: f64 = f64::from(total_pixel) / (f64::from(*count) * class_count);
        weight_map.insert(*class_id, class_weight);
    }

    log::info!("Inverse class weights: {:?}", weight_map);
}

pub async fn calc_mean_std(dataset_path: &String) {
    let entries = fs::read_dir(dataset_path).unwrap();
    let mut threads = JoinSet::new();
    let mean_map = Arc::new(Mutex::new(HashMap::<usize, Vec<f64>>::new()));
    let std_map = Arc::new(Mutex::new(HashMap::<usize, Vec<f64>>::new()));
    let sem = Arc::new(Semaphore::new(10));
    for entry in entries {
        let entry = entry.unwrap();
        if !entry.path().is_file() {
            continue;
        }
        let mean_map = Arc::clone(&mean_map);
        let std_map = Arc::clone(&std_map);
        let sem = Arc::clone(&sem);
        threads.spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let image =
                imgcodecs::imread(entry.path().to_str().unwrap(), imgcodecs::IMREAD_UNCHANGED)
                    .unwrap();

            let mut mean = core::Vector::<f64>::new();
            let mut stddev = core::Vector::<f64>::new();
            if image.empty() {
                log::error!("Image {} is empty", entry.path().display());
                return;
            }
            core::mean_std_dev(&image, &mut mean, &mut stddev, &core::no_array()).unwrap();

            let mut mean_map = mean_map.lock().unwrap();
            let mut std_map = std_map.lock().unwrap();
            for i in 1..=mean.len() {
                if !mean_map.contains_key(&i) {
                    mean_map.insert(i, Vec::<f64>::new());
                    std_map.insert(i, Vec::<f64>::new());
                }
                mean_map.get_mut(&i).unwrap().push(mean.get(i - 1).unwrap());
                std_map
                    .get_mut(&i)
                    .unwrap()
                    .push(stddev.get(i - 1).unwrap());
            }

            log::info!("Image {} done", entry.path().display());
        });
    }

    while let Some(result) = threads.join_next().await {
        match result {
            Ok(()) => {}
            Err(e) => {
                log::error!("Error {}", e);
                threads.abort_all();
                break;
            }
        }
    }

    let mean_map = mean_map.lock().unwrap();
    let std_map = std_map.lock().unwrap();

    assert_eq!(mean_map.len(), std_map.len());

    let mut mean_vec: Vec<f64> = vec![0.0; mean_map.len()];
    for (i, mean) in mean_map.iter() {
        let mean = mean.iter().sum::<f64>() / mean.len() as f64;
        *mean_vec.get_mut(*i - 1).unwrap() = mean;
    }

    let mut std_vec: Vec<f64> = vec![0.0; std_map.len()];
    for (i, std) in std_map.iter() {
        let std = std.iter().sum::<f64>() / std.len() as f64;
        *std_vec.get_mut(*i - 1).unwrap() = std;
    }

    log::info!("Mean: {:?}", mean_vec);
    log::info!("Std: {:?}", std_vec);
    log::info!("Dataset calculated");
}
