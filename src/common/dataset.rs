use std::{
    collections::HashMap,
    fs,
    sync::{Arc, Mutex},
};

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
                entry
                    .path()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string()
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

    let mut threads = JoinSet::new();
    for entry in entries {
        let entry = entry.unwrap();
        let type_map = Arc::clone(&type_map);

        threads.spawn(async move {
            let image = image::open(entry.path()).unwrap();
            let image = image.as_luma8().unwrap();
            log::info!("Loaded image: {}", entry.path().display());
            let mut current_img_type_map = HashMap::<u8, i32>::new();
            for (_, _, pixel) in image.enumerate_pixels() {
                if current_img_type_map.get(&pixel[0]).is_none() {
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

    while threads.join_next().await.is_some() {}
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
