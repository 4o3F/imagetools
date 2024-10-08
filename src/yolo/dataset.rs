use std::{
    collections::HashMap,
    fs,
    sync::{Arc, Mutex},
};

use itertools::Itertools;
use tokio::{sync::Semaphore, task::JoinSet};
use tracing_unwrap::ResultExt;

use crate::THREAD_POOL;

pub async fn split_dataset(dataset_path: &String, train_ratio: &f32) {
    let entries = fs::read_dir(dataset_path).unwrap();
    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(
        (*THREAD_POOL.read().expect_or_log("Get pool error")).into(),
    ));
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
                        .to_string()
                        .replace(".txt", ".png")
                ));
            });
        }
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

pub async fn count_types(dataset_path: &String) {
    let entries = fs::read_dir(dataset_path).unwrap();
    let type_map = Arc::new(Mutex::new(HashMap::<u8, u32>::new()));
    let mut threads = JoinSet::new();
    for entry in entries {
        let entry = entry.unwrap();
        let type_map = Arc::clone(&type_map);
        threads.spawn(async move {
            let content = fs::read_to_string(entry.path()).unwrap();
            let mut current_type_map = HashMap::<u8, u32>::new();
            content.lines().for_each(|line| {
                let class_id = line
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse::<u8>()
                    .unwrap();
                let count = current_type_map.entry(class_id).or_insert(0);
                *count += 1;
            });
            let mut type_map = type_map.lock().unwrap();
            for (class_id, count) in current_type_map.iter() {
                let total_count = type_map.entry(*class_id).or_insert(0);
                *total_count += count;
            }
        });
    }
    while threads.join_next().await.is_some() {}
    let type_map = type_map.lock().unwrap();
    for key in type_map.keys().sorted() {
        let class_id = key;
        let count = type_map.get(class_id).unwrap();
        tracing::info!("Class {}: {}", class_id, count);
    }
}
