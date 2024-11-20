use indicatif::ProgressStyle;
use opencv::{
    core::{self, MatTraitConst, ModifyInplace, Vec3b},
    imgcodecs::{self, imread, IMREAD_COLOR, IMREAD_GRAYSCALE},
    imgproc::COLOR_BGR2RGB,
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
use tracing::{info_span, Instrument, Span};
use tracing_indicatif::span_ext::IndicatifSpanExt;
use tracing_unwrap::{OptionExt, ResultExt};

use crate::THREAD_POOL;

fn check_semantic_segmentation_dataset(dataset_path: &PathBuf) -> bool {
    if !(dataset_path.join("images").is_dir() && dataset_path.join("labels").is_dir()) {
        tracing::error!(
            "Invalid dataset path: {}, should contain images and labels folders",
            dataset_path.display()
        );
        false
    } else {
        true
    }
}

// This will generate CSV format dataset list for huggingface dataset lib
pub fn generate_dataset_csv(dataset_path: &String, train_ratio: &f32) {
    let dataset_path = PathBuf::from(dataset_path);
    if !check_semantic_segmentation_dataset(&dataset_path) {
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

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct DatasetItem {
    image: String,
    label: String,
}

pub fn generate_dataset_json(dataset_path: &String, train_ratio: &f32) {
    let dataset_path = PathBuf::from(dataset_path);
    if !check_semantic_segmentation_dataset(&dataset_path) {
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

    let mut data = Vec::<DatasetItem>::new();

    for (image, label) in images.iter().zip(labels.iter()) {
        if image.file_stem() != label.file_stem() {
            return tracing::error!(
                "Image and label should have same name, but encountered {}, {}",
                image.display(),
                label.display()
            );
        }

        data.push(DatasetItem {
            image: fs::canonicalize(image).unwrap().display().to_string(),
            label: fs::canonicalize(label).unwrap().display().to_string(),
        });
    }

    use rand::seq::SliceRandom;
    data.shuffle(&mut rand::thread_rng());

    let train_count = (data.len() as f32 * train_ratio) as i32;
    let train_data = data[0..train_count as usize].to_vec();
    let val_data = data[train_count as usize..].to_vec();

    fs::write(
        dataset_path.join("train.json"),
        serde_json::to_string(&train_data).expect_or_log("Failed to serialize"),
    )
    .expect_or_log("Failed to write");

    fs::write(
        dataset_path.join("val.json"),
        serde_json::to_string(&val_data).expect_or_log("Failed to serialize"),
    )
    .expect_or_log("Failed to write");
    tracing::info!(
        "Saved to train: {} val: {}",
        dataset_path.join("train.json").display(),
        dataset_path.join("val.json").display()
    );
}

pub fn combine_dataset_json(dataset_path: &Vec<String>, save_path: &String) {
    let save_path = PathBuf::from(save_path);
    if !save_path.is_dir() {
        fs::create_dir_all(save_path.clone()).expect_or_log("Failed to create directory");
    }
    let mut combined_train_datas = Vec::<DatasetItem>::new();
    let mut combined_val_datas = Vec::<DatasetItem>::new();
    for dataset_path in dataset_path {
        let dataset_path = PathBuf::from(dataset_path);
        if !(dataset_path.join("train.json").is_file() && dataset_path.join("val.json").is_file()) {
            tracing::error!(
                "Invalid dataset path: {}, should contain train.json and val.json",
                dataset_path.display()
            );
            return;
        }

        let train_data: Vec<DatasetItem> = serde_json::from_str(
            &fs::read_to_string(dataset_path.join("train.json")).expect_or_log("Failed to read"),
        )
        .expect("Failed to deserialize");

        let val_data: Vec<DatasetItem> = serde_json::from_str(
            &fs::read_to_string(dataset_path.join("val.json")).expect_or_log("Failed to read"),
        )
        .expect("Failed to deserialize");

        combined_train_datas.extend(train_data);
        combined_val_datas.extend(val_data);
    }

    let combined_train_datas =
        serde_json::to_string(&combined_train_datas).expect_or_log("Failed to serialize");

    let combined_val_datas =
        serde_json::to_string(&combined_val_datas).expect_or_log("Failed to serialize");

    fs::write(save_path.join("train.json"), combined_train_datas).expect_or_log("Failed to write");
    fs::write(save_path.join("val.json"), combined_val_datas).expect_or_log("Failed to write");
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

pub async fn count_rgb(dataset_path: &String, rgb_list: &String) {
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

    let count_map = Arc::new(Mutex::new(HashMap::<[u8; 3], u64>::new()));
    {
        // Split RGB list
        for rgb in rgb_list.split(";") {
            let mut rgb_vec: Vec<u8> = vec![];
            for splited in rgb.split(',') {
                let splited = splited.parse::<u8>().unwrap();
                rgb_vec.push(splited);
            }

            count_map
                .lock()
                .unwrap()
                .insert([rgb_vec[0], rgb_vec[1], rgb_vec[2]], 0);
        }
    }

    let sem = Arc::new(Semaphore::new(
        (*THREAD_POOL.read().expect_or_log("Get pool error")).into(),
    ));
    let mut threads = JoinSet::new();

    let header_span = info_span!("count_rgb_threads");
    header_span.pb_set_style(
        &ProgressStyle::with_template("{spinner} Processing {msg}\n{wide_bar} {pos}/{len}")
            .unwrap(),
    );
    header_span.pb_set_length(entries.len() as u64);
    header_span.pb_set_message("Processing");

    let header_span_enter = header_span.enter();

    for entry in entries {
        if !entry.is_file() {
            continue;
        }
        let count_map = Arc::clone(&count_map);
        let sem = Arc::clone(&sem);
        let header_span = header_span.clone();
        threads.spawn(
            async move {
                let _ = sem.acquire().await.unwrap();
                let mut img = imread(entry.to_str().unwrap(), IMREAD_COLOR).unwrap();
                unsafe {
                    img.modify_inplace(|input, output| {
                        opencv::imgproc::cvt_color(input, output, COLOR_BGR2RGB, 0)
                            .expect_or_log("Cvt grayscale to RGB error")
                    });
                }
                let row = img.rows();
                let cols = img.cols();
                let img = Arc::new(RwLock::new(img));

                (0..row).into_par_iter().for_each(|row_index| {
                    let mut row_type_map = HashMap::<[u8; 3], u64>::new();
                    let row = img.read().row(row_index).unwrap().clone_pointee();
                    for col_index in 0..cols {
                        let pixel = row.at_2d::<Vec3b>(0, col_index).unwrap();
                        row_type_map
                            .entry(pixel.0)
                            .and_modify(|x| *x = x.saturating_add(1))
                            .or_insert(1);
                    }

                    let mut entry = count_map.lock().unwrap();
                    for (rgb_color, count) in row_type_map.iter() {
                        let total_count = entry.entry(*rgb_color).or_insert(0);
                        *total_count = total_count.saturating_add(*count);
                    }
                });
                tracing::trace!("Image {} done", entry.to_str().unwrap());
                Span::current()
                    .pb_set_message(&format!("{}", entry.file_name().unwrap().to_str().unwrap()));
                Span::current().pb_inc(1);
            }
            .instrument(header_span),
        );
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
    drop(header_span_enter);
    drop(header_span);
    tracing::info!("Dataset counted");

    let type_map = count_map.lock().unwrap();
    for (rgb, count) in (*type_map).iter() {
        tracing::info!("{},{},{}: {}", rgb[0], rgb[1], rgb[2], count);
    }
    tracing::info!("Classes counts: {:?}", type_map);

    // let total_pixel = type_map.values().sum::<i64>();

    // let mut weight_map = HashMap::<[u8; 3], f64>::new();
    // let class_count = type_map.len() as f64;
    // for (class_id, count) in type_map.iter() {
    //     let class_weight: f64 = f64::from(total_pixel) / (f64::from(*count) * class_count);
    //     weight_map.insert(*class_id, class_weight);
    // }

    // tracing::info!("Inverse class weights: {:?}", weight_map);
}

pub async fn calc_mean_std(dataset_path: &String) {
    let entries = fs::read_dir(dataset_path).expect_or_log("Failed to read directory");
    let entries = entries
        .into_iter()
        .map(|x| x.expect_or_log("Failed to iterate entries").path())
        .collect::<Vec<PathBuf>>();
    // let entries = Arc::new(entries);
    let mut threads = JoinSet::new();

    let mean_map = Arc::new(Mutex::new(HashMap::<usize, Vec<f64>>::new()));
    let std_map = Arc::new(Mutex::new(HashMap::<usize, Vec<f64>>::new()));

    let min_value = Arc::new(Mutex::new(f64::MAX));
    let max_value = Arc::new(Mutex::new(f64::MIN));

    let header_span = info_span!("calc_mean_std_thread");
    header_span.pb_set_style(
        &ProgressStyle::with_template("{spinner} Processing {msg}\n{wide_bar} {pos}/{len}")
            .unwrap(),
    );
    header_span.pb_set_length(entries.len() as u64);

    let header_span_enter = header_span.enter();

    // let entries_iter = ProgressAdaptor::new(0..entries.len());
    // let entries_iter_progress = entries_iter.items_processed();
    // let log_interval = {
    //     let mut tmp = entries.len();
    //     tracing::debug!("Entries len: {}", entries.len());
    //     let mut result = 0;
    //     while tmp != 0 {
    //         result += 1;
    //         tmp /= 10;
    //     }
    //     10_usize.pow(result - 2)
    // };

    for entry in entries {
        if !entry.is_file() {
            return;
        }
        let mean_map = Arc::clone(&mean_map);
        let std_map = Arc::clone(&std_map);

        let global_min_value = Arc::clone(&min_value);
        let global_max_value = Arc::clone(&max_value);

        threads.spawn(
            async move {
                let image = imgcodecs::imread(
                    entry
                        .to_str()
                        .expect_or_log("Failed to convert path to string"),
                    imgcodecs::IMREAD_UNCHANGED,
                )
                .expect_or_log("Failed to read image");

                let mut mean = core::Vector::<f64>::new();
                let mut stddev = core::Vector::<f64>::new();
                if image.empty() {
                    tracing::error!("Image {} is empty", entry.display());
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

                Span::current()
                    .pb_set_message(&format!("{}", entry.file_name().unwrap().to_str().unwrap()));
                Span::current().pb_inc(1);
            }
            .in_current_span(),
        );
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

    drop(header_span_enter);
    drop(header_span);

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
