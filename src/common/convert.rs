use std::{
    collections::HashMap,
    fs,
    sync::{Arc, Mutex},
};

use cocotools::{
    coco::object_detection::{Annotation, Bbox, Category, Dataset, Image, Rle, Segmentation},
    mask::utils::Area,
};
use image::{Rgb, RgbImage};
use tokio::{sync::Semaphore, task::JoinSet};
use tracing_unwrap::ResultExt;

use crate::THREAD_POOL;

fn mask_image_array(image: &RgbImage, rgb: Rgb<u8>) -> ndarray::Array2<u8> {
    let mut mask = ndarray::Array2::<u8>::zeros((image.height() as usize, image.width() as usize));
    for (x, y, pixel) in image.enumerate_pixels() {
        let Rgb([r, g, b]) = pixel;
        let [tr, tg, tb] = &rgb.0;
        if r == tr && g == tg && b == tb {
            mask[[y as usize, x as usize]] = 1;
        }
    }
    mask
}

pub async fn rgb2rle(dataset_path: &String, rgb_list: &str) {
    let mut color_class_map = HashMap::<Rgb<u8>, u32>::new();
    let dataset = Arc::new(Mutex::new(Dataset {
        info: Default::default(),
        images: Vec::<Image>::new(),
        annotations: Vec::<Annotation>::new(),
        categories: Vec::<Category>::new(),
        licenses: vec![],
    }));
    // Add categories
    {
        let mut dataset_guard = dataset.lock().unwrap();

        for (class_id, rgb) in rgb_list.split(";").enumerate() {
            let mut rgb_vec: Vec<u8> = vec![];
            for splited in rgb.split(',') {
                let splited = splited.parse::<u8>().unwrap();
                rgb_vec.push(splited);
            }

            // TODO: add ability for super category
            let rgb = Rgb([rgb_vec[0], rgb_vec[1], rgb_vec[2]]);
            color_class_map.insert(rgb, class_id as u32);
            dataset_guard.categories.push(Category {
                id: class_id as u32,
                name: rgb_vec[3].to_string(),
                supercategory: rgb_vec[3].to_string(),
            });
        }
    }

    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(
        (*THREAD_POOL.read().expect_or_log("Get pool error")).into(),
    ));

    let image_count = Arc::new(Mutex::new(0));
    let annotation_count = Arc::new(Mutex::new(0));
    // Walk through all images in BASE_PATH
    let entries = fs::read_dir(dataset_path).unwrap();
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
            let dataset = Arc::clone(&dataset);
            let color_class_map = color_class_map.clone();
            let image_count = Arc::clone(&image_count);
            let annotation_count = Arc::clone(&annotation_count);
            threads.spawn(async move {
                // Limit tasks to 10
                let _permit = permit.acquire().await.unwrap();

                let img = image::open(entry.path()).unwrap().into_rgb8();
                // First process the image without acquiring the lock, boost performance
                let mut annotations = Vec::<Annotation>::new();

                for (color, class_id) in color_class_map.clone().iter() {
                    let mask: cocotools::mask::Mask = mask_image_array(&img, *color);

                    let rle = Rle::from(&mask);
                    let area = rle.area();
                    let bbox = Bbox::from(&rle);
                    let new_annotation = Annotation {
                        id: 0,
                        image_id: 0,
                        category_id: *class_id,
                        segmentation: Segmentation::Rle(rle),
                        area: area as f64,
                        bbox,
                        iscrowd: 1,
                    };
                    annotations.push(new_annotation);
                }

                // Acquire image lock
                let mut image = Image {
                    id: 0,
                    width: img.width(),
                    height: img.height(),
                    file_name: entry
                        .file_name()
                        .into_string()
                        .unwrap()
                        .replace(".png", ".jpg"),
                    license: Default::default(),
                    flickr_url: Default::default(),
                    coco_url: Default::default(),
                    date_captured: Default::default(),
                };
                {
                    let mut image_guard = image_count.lock().unwrap();
                    image.id = *image_guard;
                    *image_guard += 1;
                }

                // Acquire annotation lock
                {
                    let mut annotation_guard = annotation_count.lock().unwrap();
                    for annotion in annotations.iter_mut() {
                        annotion.id = *annotation_guard;
                        *annotation_guard += 1;
                        annotion.image_id = image.id;
                    }
                }

                // Lock dataset and start to add
                let mut dataset_guard = dataset.lock().unwrap();
                dataset_guard.images.push(image);
                dataset_guard.annotations.extend(annotations);
                println!(
                    "{} finished process",
                    entry.file_name().into_string().unwrap()
                );
            });
        }
    }

    while threads.join_next().await.is_some() {}

    let dataset_guard = dataset.lock().unwrap();
    let dataset_string = serde_json::to_string(&*dataset_guard).unwrap();
    fs::write(
        format!("{}/resized_labels.json", dataset_path),
        dataset_string,
    )
    .unwrap();
}
