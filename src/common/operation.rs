use opencv::{
    core::{self, Mat, MatTraitConst, Size, Vector},
    imgcodecs::{self, imread, imwrite},
    imgproc,
};
use std::{fs, path::PathBuf, sync::Arc};
use tokio::{sync::Semaphore, task::JoinSet};
use tracing_unwrap::{OptionExt, ResultExt};

use crate::THREAD_POOL;

#[deprecated]
#[allow(dead_code)]
pub async fn resize_images(
    dataset_path: &String,
    target_height: &i32,
    target_width: &i32,
    filter: &str,
) {
    let mut entries: Vec<PathBuf> = Vec::new();
    let dataset_path = PathBuf::from(dataset_path);
    if dataset_path.is_file() {
        entries.push(dataset_path.clone());
        fs::create_dir_all(format!(
            "{}/output/",
            dataset_path.parent().unwrap().to_str().unwrap()
        ))
        .expect_or_log("Failed to create directory");
    } else {
        entries = fs::read_dir(dataset_path.clone())
            .unwrap()
            .map(|x| x.unwrap().path())
            .collect();

        fs::create_dir_all(format!("{}/output/", dataset_path.to_str().unwrap()))
            .expect_or_log("Failed to create directory");
    }

    let filter = match filter.to_lowercase().as_str() {
        "nearest" => imgproc::INTER_NEAREST,
        "linear" => imgproc::INTER_LINEAR,
        "cubic" => imgproc::INTER_CUBIC,
        "lanczos" => imgproc::INTER_LANCZOS4,
        _ => {
            tracing::error!("Invalid filter type. Please use one of the following: nearest, linear, cubic, gaussian, lanczos");
            return;
        }
    };

    let mut threads = JoinSet::new();
    let semaphore = Arc::new(Semaphore::new(
        (*THREAD_POOL.read().expect_or_log("Get pool error")).into(),
    ));

    for entry in entries {
        let permit = Arc::clone(&semaphore);
        let target_width = *target_width;
        let target_height = *target_height;
        threads.spawn(async move {
            let _permit = permit.acquire().await.unwrap();
            let img = imread(entry.to_str().unwrap(), imgcodecs::IMREAD_COLOR)
                .expect_or_log("Failed to read image");
            if img.channels() != 3 {
                tracing::error!(
                    "Image {} is not RGB image",
                    entry.file_name().unwrap().to_str().unwrap()
                );
                return Err(());
            }

            let mut result = Mat::default();
            opencv::imgproc::resize(
                &img,
                &mut result,
                Size::new(target_width, target_height),
                0.,
                0.,
                filter,
            )
            .expect_or_log("Failed to resize image");

            imwrite(
                format!(
                    "{}/output/{}",
                    entry.parent().unwrap().to_str().unwrap(),
                    entry.file_name().unwrap().to_str().unwrap()
                )
                .as_str(),
                &result,
                &Vector::new(),
            )
            .unwrap();
            tracing::info!(
                "Image {} done",
                entry.file_name().unwrap().to_str().unwrap()
            );
            Ok(())
        });
    }
    while let Some(res) = threads.join_next().await {
        if let Err(e) = res {
            tracing::error!("{}", e);
            continue;
        }
    }
    tracing::info!("All done!");
}

#[derive(Clone, Copy)]
pub enum EdgePosition {
    Top,
    Bottom,
    Left,
    Right,
}

pub async fn strip_image_edge(
    source_path: &String,
    save_path: &String,
    position: &EdgePosition,
    length: &i32,
) {
    let img = imgcodecs::imread(source_path, imgcodecs::IMREAD_UNCHANGED).unwrap();
    tracing::info!("Loaded image: {}", source_path);
    let size = img.size().unwrap();
    let (width, height) = (size.width, size.height);

    let cropped_img = match position {
        EdgePosition::Top => {
            core::Mat::roi(&img, core::Rect::new(0, *length, width, height - *length)).unwrap()
        }
        EdgePosition::Bottom => {
            core::Mat::roi(&img, core::Rect::new(0, 0, width, height - *length)).unwrap()
        }
        EdgePosition::Left => {
            core::Mat::roi(&img, core::Rect::new(*length, 0, width - *length, height)).unwrap()
        }
        EdgePosition::Right => {
            core::Mat::roi(&img, core::Rect::new(0, 0, width - *length, height)).unwrap()
        }
    };

    imgcodecs::imwrite(save_path, &cropped_img, &core::Vector::new()).unwrap();

    tracing::info!("Image {} done", save_path);
}

pub fn crop_rectangle_region(source_path: &String, target_path: &str, corners: &str) {
    let cords: Vec<(i32, i32)> = corners
        .split(';')
        .map(|s| {
            s.split(',')
                .map(|s| s.parse::<i32>().unwrap())
                .collect::<Vec<i32>>()
        })
        .map(|c| {
            if c.len() != 2 {
                tracing::error!("Invalid rectangle coordinates");
            }
            (c[0], c[1])
        })
        .collect();
    if cords.len() != 2 {
        tracing::error!("Invalid rectangle coordinates");
        return;
    }
    tracing::info!("Rectangle coordinates: {:?}", cords);
    let img = imgcodecs::imread(source_path, imgcodecs::IMREAD_UNCHANGED).unwrap();
    tracing::info!("Loaded image: {}", source_path);

    let cropped_img = core::Mat::roi(
        &img,
        core::Rect::new(
            cords[0].0,
            cords[0].1,
            cords[1].0 - cords[0].0,
            cords[1].1 - cords[0].1,
        ),
    )
    .unwrap();
    imgcodecs::imwrite(target_path, &cropped_img, &core::Vector::new()).unwrap();
    tracing::info!("Image {} done", source_path);
}

pub fn normalize(dataset_path: &String, target_max: &f64, target_min: &f64) {
    let entries = fs::read_dir(dataset_path).expect_or_log("Failed to read directory");

    fs::create_dir_all(format!("{}/output/", dataset_path))
        .expect_or_log("Failed to create directory");

    for entry in entries {
        let entry = entry.unwrap();
        if entry.path().is_dir() {
            continue;
        }
        let img = imgcodecs::imread(
            entry
                .path()
                .to_str()
                .expect_or_log("Failed to get image path"),
            imgcodecs::IMREAD_UNCHANGED,
        )
        .expect_or_log("Failed to read image");

        let size = img.size().expect_or_log("Failed to get image size");
        let mut dst = core::Mat::new_rows_cols_with_default(
            size.height,
            size.width,
            img.typ(),
            opencv::core::Scalar::all(0.),
        )
        .expect_or_log("Failed to create dst mat");
        core::normalize(
            &img,
            &mut dst,
            *target_min,
            *target_max,
            core::NORM_MINMAX,
            -1,
            &core::no_array(),
        )
        .expect_or_log("Failed to normalize");

        let dst_path = format!(
            "{}/output/{}",
            dataset_path,
            entry
                .path()
                .file_name()
                .expect_or_log("Failed to get file name")
                .to_str()
                .expect_or_log("Failed to convert file name to string")
                .to_owned()
        );
        imgcodecs::imwrite(&dst_path, &dst, &core::Vector::new())
            .expect_or_log("Failed to write image");
        tracing::info!("Image {} done", dst_path);
    }
}

pub mod resize;
