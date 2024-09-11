use image::imageops::FilterType;
use opencv::{core, core::MatTraitConst, imgcodecs};
use std::{fs, io::Cursor, sync::Arc};
use tokio::{fs::File, io::AsyncWriteExt, sync::Semaphore, task::JoinSet};
use tracing_unwrap::OptionExt;

pub async fn resize_images(
    dataset_path: &String,
    target_height: &u32,
    target_width: &u32,
    filter: &str,
) {
    let filter = match filter.to_lowercase().as_str() {
        "nearest" => FilterType::Nearest,
        "linear" => FilterType::Triangle,
        "cubic" => FilterType::CatmullRom,
        "gaussian" => FilterType::Gaussian,
        "lanczos" => FilterType::Lanczos3,
        _ => {
            tracing::error!("Invalid filter type. Please use one of the following: nearest, linear, cubic, gaussian, lanczos");
            return;
        }
    };

    let entries = fs::read_dir(dataset_path).unwrap();
    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(10));
    fs::create_dir_all(format!("{}\\..\\resized\\", dataset_path)).unwrap();
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
            let dataset_path = dataset_path.clone();
            let target_height = *target_height;
            let target_width = *target_width;
            threads.spawn(async move {
                let _permit = permit.acquire().await.unwrap();

                let img = image::open(entry.path()).unwrap();
                let img_resized = img.resize_exact(target_width, target_height, filter);
                let mut bytes: Vec<u8> = Vec::new();
                img_resized
                    .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
                    .unwrap();
                let mut file = File::create(format!(
                    "{}\\..\\resized\\{}",
                    dataset_path,
                    entry.path().file_name().unwrap().to_str().unwrap()
                ))
                .await
                .unwrap();
                file.write_all(&bytes).await.unwrap();
                tracing::info!(
                    "Image {} done\n",
                    entry.path().file_name().unwrap().to_str().unwrap()
                );
            });
        }
    }
    while threads.join_next().await.is_some() {}
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
        EdgePosition::Left => core::Mat::roi(
            &img,
            core::Rect::new(*length, 0, width - *length, height),
        )
        .unwrap(),
        EdgePosition::Right => {
            core::Mat::roi(&img, core::Rect::new(0, 0, width - *length, height)).unwrap()
        }
    };

    imgcodecs::imwrite(save_path, &cropped_img, &core::Vector::new()).unwrap();

    tracing::info!("Image {} done", save_path);
}
