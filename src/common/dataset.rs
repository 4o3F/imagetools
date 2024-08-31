use std::{fs, sync::{Arc, Mutex}};

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
