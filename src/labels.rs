use std::{
    collections::HashMap,
    fs,
    sync::{Arc, Mutex, RwLock},
};

use cocotools::{
    coco::object_detection::{Annotation, Bbox, Category, Dataset, Image, Rle, Segmentation},
    mask::utils::Area,
};
use image::{Rgb, RgbImage};
use itertools::Itertools;
use opencv::{core::MatTrait, imgproc};
use rand::seq::SliceRandom;
use rand::thread_rng;
use tokio::{fs::File, io::AsyncWriteExt, sync::Semaphore, task::JoinSet};
#[allow(dead_code)]
pub async fn gray_filter() {
    let path = "E:\\YRCC_ori\\trainannot";
    let entries = fs::read_dir(path).unwrap();

    let transform_map = Arc::new(RwLock::new(HashMap::<Rgb<u8>, Rgb<u8>>::new()));
    {
        let mut transform_map = transform_map.write().unwrap();
        transform_map.insert(Rgb([0, 0, 0]), Rgb([0, 0, 0]));
        transform_map.insert(Rgb([1, 1, 1]), Rgb([255, 0, 0]));
        transform_map.insert(Rgb([2, 2, 2]), Rgb([0, 255, 0]));
        transform_map.insert(Rgb([3, 3, 3]), Rgb([0, 0, 255]));
    }
    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(10));
    for entry in entries {
        let entry = entry.unwrap();
        let sem = Arc::clone(&sem);
        let transform_map = Arc::clone(&transform_map);
        threads.spawn(async move {
            let _ = sem.acquire().await.unwrap();
            let img = image::open(entry.path()).unwrap();
            let mut img = img.to_rgb8();
            for (_, _, pixel) in img.enumerate_pixels_mut() {
                let Rgb([r, g, b]) = pixel;
                let transform_map = transform_map.read().unwrap();
                let new_color = transform_map.get(&Rgb([*r, *g, *b])).unwrap();
                *r = new_color.0[0];
                *g = new_color.0[1];
                *b = new_color.0[2];
            }
            img.save(format!(
                "{}\\..\\output\\{}",
                path,
                entry.file_name().into_string().unwrap()
            ))
            .unwrap();
            println!("{} finished", entry.file_name().into_string().unwrap());
        });
    }
    while threads.join_next().await.is_some() {}
}

#[allow(dead_code)]
pub async fn count_types() {
    let path = "E:\\卫星数据\\labels\\label_outputs";
    let entries = fs::read_dir(path).unwrap();
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
        println!("{}: {}", class_id, count);
    }
}

#[allow(dead_code)]
pub fn split_dataset() {
    let path = "E:\\YRCC_ori\\yolo\\labels";
    let entries = fs::read_dir(path).unwrap();
    let mut names = Vec::<String>::new();
    for entry in entries {
        let entry = entry.unwrap();
        let name = entry.file_name().to_str().unwrap().to_string();
        names.push(name);
    }
    names.shuffle(&mut thread_rng());
    let train_count = (names.len() as f64 * 0.8).floor() as usize;
    let train_names = names.iter().take(train_count);
    let val_names = names.iter().skip(train_count);
    let train_path = "E:\\YRCC_ori\\yolo\\train.txt";
    let val_path = "E:\\YRCC_ori\\yolo\\val.txt";
    let mut train_content = String::new();
    let mut val_content = String::new();
    train_names.for_each(|name| {
        train_content.push_str("./images/");
        train_content.push_str(name.replace(".txt", ".png").as_str());
        train_content.push('\n');
    });
    val_names.for_each(|name| {
        val_content.push_str("./images/");
        val_content.push_str(name.replace(".txt", ".png").as_str());
        val_content.push('\n');
    });
    fs::write(train_path, train_content).unwrap();
    fs::write(val_path, val_content).unwrap();
}

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

#[allow(dead_code)]
async fn rgb2rle() {
    // let base_path =
    //     "D:\\Documents\\Competitions\\ChallengeCup\\NWPU_YRCC2\\datasets\\resized_labels";
    let base_path = "E:\\卫星数据\\yolo\\label_imgs";
    let mut color_class_map = HashMap::<Rgb<u8>, u32>::new();
    color_class_map.insert(Rgb([0, 0, 0]), 0);
    color_class_map.insert(Rgb([0, 255, 0]), 1);
    color_class_map.insert(Rgb([255, 255, 0]), 2);
    color_class_map.insert(Rgb([255, 0, 255]), 3);
    color_class_map.insert(Rgb([0, 255, 255]), 4);
    color_class_map.insert(Rgb([0, 0, 255]), 5);
    color_class_map.insert(Rgb([255, 255, 255]), 6);

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
        dataset_guard.categories.push(Category {
            id: 0,
            name: "land".to_string(),
            supercategory: "land".to_string(),
        });
        dataset_guard.categories.push(Category {
            id: 1,
            name: "water".to_string(),
            supercategory: "water".to_string(),
        });
        dataset_guard.categories.push(Category {
            id: 2,
            name: "shore_ice".to_string(),
            supercategory: "shore_ice".to_string(),
        });
        dataset_guard.categories.push(Category {
            id: 3,
            name: "stream_ice".to_string(),
            supercategory: "stream_ice".to_string(),
        });
        dataset_guard.categories.push(Category {
            id: 4,
            name: "vertical_ice".to_string(),
            supercategory: "vertical_ice".to_string(),
        });
        dataset_guard.categories.push(Category {
            id: 5,
            name: "horizontal_ice".to_string(),
            supercategory: "horizontal_ice".to_string(),
        });
        dataset_guard.categories.push(Category {
            id: 6,
            name: "snow".to_string(),
            supercategory: "snow".to_string(),
        });
    }

    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(10));

    let image_count = Arc::new(Mutex::new(0));
    let annotation_count = Arc::new(Mutex::new(0));
    // Walk through all images in BASE_PATH
    let entries = fs::read_dir(base_path).unwrap();
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
        "D:\\Documents\\Competitions\\ChallengeCup\\NWPU_YRCC2\\datasets\\resized_labels.json",
        dataset_string,
    )
    .unwrap();
}

#[allow(dead_code)]
pub async fn rgb2yolo() {
    // let base_path =
    // "D:\\Documents\\Competitions\\ChallengeCup\\NWPU_YRCC2\\datasets\\resized_labels_768";
    let base_path = "E:\\卫星数据\\labels\\final";
    let mut color_class_map = HashMap::<Rgb<u8>, u32>::new();
    color_class_map.insert(Rgb([0, 0, 0]), 0);
    color_class_map.insert(Rgb([0, 255, 0]), 1);
    color_class_map.insert(Rgb([255, 255, 0]), 2);
    color_class_map.insert(Rgb([255, 0, 255]), 3);
    color_class_map.insert(Rgb([0, 255, 255]), 4);
    color_class_map.insert(Rgb([0, 0, 255]), 5);
    color_class_map.insert(Rgb([255, 255, 255]), 6);

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
        dataset_guard.categories.push(Category {
            id: 0,
            name: "land".to_string(),
            supercategory: "land".to_string(),
        });
        dataset_guard.categories.push(Category {
            id: 1,
            name: "water".to_string(),
            supercategory: "water".to_string(),
        });
        dataset_guard.categories.push(Category {
            id: 2,
            name: "ice".to_string(),
            supercategory: "ice".to_string(),
        });
    }
    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(10));

    // Walk through all images in BASE_PATH
    let entries = fs::read_dir(base_path).unwrap();
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
            let color_class_map = color_class_map.clone();
            threads.spawn(async move {
                // Limit tasks to 10
                let _permit = permit.acquire().await.unwrap();

                let img = image::open(entry.path()).unwrap().into_rgb8();

                if !crate::images::check_valid_pixel_count(&img) {
                    return ();
                }

                let mut labels = Vec::<String>::new();
                for (color, class_id) in color_class_map.clone().iter() {
                    // Output image for this class
                    // let mut output_img = img.clone();
                    // TODO: remove this test only code
                    // if color != &Rgb([0, 255, 0]) {
                    //     continue;
                    // }

                    let mut mat = opencv::core::Mat::new_rows_cols_with_default(
                        768,
                        768,
                        opencv::core::CV_8U,
                        opencv::core::Scalar::all(0.),
                    )
                    .unwrap();
                    // println!("{:?}", mat);

                    // Turn rgb label to gray image mask
                    for (x, y, pixel) in img.enumerate_pixels() {
                        let Rgb([r, g, b]) = pixel;
                        let Rgb([tr, tg, tb]) = color;
                        if r == tr && g == tg && b == tb {
                            // Set mat at x,y to 255
                            *mat.at_2d_mut::<u8>(x as i32, y as i32).unwrap() = 255;
                        } else {
                            *mat.at_2d_mut::<u8>(x as i32, y as i32).unwrap() = 0;
                        }
                    }

                    let mut contours =
                        opencv::core::Vector::<opencv::core::Vector<opencv::core::Point>>::new();

                    // Same level next
                    // Same level previous
                    // Child
                    // Parent
                    let mut hierarchy = opencv::core::Vector::<opencv::core::Vec4i>::new();
                    imgproc::find_contours_with_hierarchy_def(
                        &mat,
                        &mut contours,
                        &mut hierarchy,
                        imgproc::RETR_CCOMP,
                        imgproc::CHAIN_APPROX_TC89_KCOS,
                    )
                    .unwrap();

                    // println!("{:?}", contours);
                    // println!("{:?}", hierarchy);

                    let mut combined_contours: Vec<Vec<(i32, i32)>> = Vec::new();

                    // Now go through all the hierarchy and combine contours
                    let mut current_index: i32 = 0;
                    while current_index != -1 && !contours.is_empty() {
                        let current_contour = contours.get(current_index as usize).unwrap();
                        let current_hierarchy = hierarchy.get(current_index as usize).unwrap();

                        let mut parent_points = Vec::<(i32, i32)>::new();
                        current_contour.iter().for_each(|point| {
                            parent_points.push((point.x, point.y));
                        });
                        if current_hierarchy.get(2).unwrap() != &-1 {
                            // Contain child, go through holes
                            let mut child_contour_index = *current_hierarchy.get(2).unwrap();
                            loop {
                                let child_contour =
                                    contours.get(child_contour_index as usize).unwrap();
                                let child_hierarchy =
                                    hierarchy.get(child_contour_index as usize).unwrap();

                                let mut child_points = Vec::<(i32, i32)>::new();
                                child_contour.iter().for_each(|point| {
                                    child_points.push((point.x, point.y));
                                });
                                // Find the nearest point between child_points and contour_points
                                let mut min_distance = f64::MAX;
                                let mut child_index = 0;
                                let mut parent_index = 0;
                                for (i, parent_point) in parent_points.iter().enumerate() {
                                    for (j, child_point) in child_points.iter().enumerate() {
                                        let distance = f64::from(
                                            (parent_point.0 - child_point.0).pow(2)
                                                + (parent_point.1 - child_point.1).pow(2),
                                        )
                                        .sqrt();
                                        if distance < min_distance {
                                            min_distance = distance;
                                            child_index = j;
                                            parent_index = i;
                                        }
                                    }
                                }

                                // Combine two contours
                                let mut new_points = Vec::<(i32, i32)>::new();
                                new_points.extend(parent_points.iter().take(parent_index + 1));
                                new_points.extend(child_points.iter().skip(child_index));
                                new_points.extend(child_points.iter().take(child_index + 1));
                                new_points.extend(parent_points.iter().skip(parent_index));
                                parent_points = new_points;
                                // println!("Combined child");
                                child_contour_index = *child_hierarchy.first().unwrap();
                                if child_contour_index == -1 {
                                    break;
                                }
                            }
                        }
                        // No more child
                        if parent_points.len() > 6 {
                            // Can't form valid polygon
                            combined_contours.push(parent_points);
                        }

                        current_index = *current_hierarchy.first().unwrap();
                    }

                    for contour in combined_contours.iter() {
                        let mut result = String::new();
                        result.push_str(class_id.to_string().as_str());
                        result.push(' ');
                        contour.iter().for_each(|point| {
                            result.push_str(&format!(
                                "{} ",
                                (f64::from(point.1) / f64::from(img.width()))
                            ));
                            result.push_str(&format!(
                                "{} ",
                                f64::from(point.0) / f64::from(img.height())
                            ));
                        });
                        result.push('\n');
                        labels.push(result);

                        /*
                        imageproc::drawing::draw_antialiased_polygon_mut(
                            &mut output_img,
                            contour
                                .iter()
                                .map(|point| imageproc::point::Point {
                                    x: point.1 as i32,
                                    y: point.0 as i32,
                                })
                                .collect::<Vec<imageproc::point::Point<i32>>>()
                                .as_slice(),
                            Rgb([255, 128, 0]),
                            interpolate,
                        );
                        */
                    }
                    /*
                    output_img
                        .save(format!(
                            "./outputs/images/{}/{}",
                            class_id,
                            entry.file_name().into_string().unwrap()
                        ))
                        .unwrap();
                    */
                }
                File::create(format!(
                    "{}/../label_outputs/{}",
                    base_path,
                    entry
                        .file_name()
                        .into_string()
                        .unwrap()
                        .to_string()
                        .replace(".png", ".txt")
                ))
                .await
                .unwrap()
                .write_all(labels.concat().as_bytes())
                .await
                .unwrap();
                println!(
                    "{} finished process",
                    entry.file_name().into_string().unwrap()
                );
            });
        }
    }

    while threads.join_next().await.is_some() {}
}

#[allow(dead_code)]
async fn transform_list() {
    let path = "D:\\Documents\\Competitions\\ChallengeCup\\NWPU_YRCC2\\datasets\\val.txt";
    let content = fs::read_to_string(path).unwrap();
    let mut result = String::new();
    content.lines().for_each(|line| {
        result.push_str(line.replace(".png", "_1.png\n").as_str());
        result.push_str(line.replace(".png", "_2.png\n").as_str());
        result.push_str(line.replace(".png", "_3.png\n").as_str());
    });
    fs::write(path.replace(".txt", "_transformed.txt"), result).unwrap();
}
