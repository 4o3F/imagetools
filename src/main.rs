use std::{
    collections::HashMap,
    fs,
    sync::{Arc, Mutex},
};

use cocotools::{
    coco::object_detection::{Annotation, Bbox, Category, Dataset, Image, Rle, Segmentation},
    mask::utils::Area,
};
use image::{imageops::FilterType, Rgb, RgbImage};
use imageproc::pixelops::interpolate;
use opencv::{core::MatTrait, imgproc};
use tokio::{sync::Semaphore, task::JoinSet};

#[allow(dead_code)]
async fn resize_mask() {
    let path = "D:\\Documents\\Competitions\\ChallengeCup\\NWPU_YRCC2\\datasets\\images";
    let entries = fs::read_dir(path).unwrap();
    let mut threads = JoinSet::new();
    for entry in entries {
        let entry = entry.unwrap();
        if entry
            .path()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .ends_with(".jpg")
        {
            // print!("Image {} started\n", entry.path().file_name().unwrap().to_str().unwrap());
            threads.spawn(async move {
                let img = image::open(entry.path()).unwrap();
                let img_resized = img.resize_exact(512, 512, FilterType::Lanczos3);
                // let img_resized = resize(&img, 512, 512, FilterType::Nearest);
                // for (x, y, pixel) in img_resized.enumerate_pixels() {
                //     let Rgba([r, g, b, _]) = pixel;
                //     match (r, g, b) {
                //         (0, 0, 0) => {}
                //         (0, 255, 0) => {}
                //         (255, 0, 255) => {}
                //         (0, 255, 255) => {}
                //         (r, g, b) => {
                //             println!(
                //                 "Invalid pixel color: ({},{},{}) at {},{} for image {}",
                //                 r,
                //                 g,
                //                 b,
                //                 x,
                //                 y,
                //                 entry.path().file_name().unwrap().to_str().unwrap()
                //             );
                //             return;
                //         }
                //     }
                // }
                img_resized
                    .save(format!(
                        "{}\\resized-lanczos3\\{}",
                        path,
                        entry.path().file_name().unwrap().to_str().unwrap()
                    ))
                    .unwrap();

                // print!("Image {} done\n", entry.path().file_name().unwrap().to_str().unwrap());
            });
        }
    }
    while let Some(_) = threads.join_next().await {}
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
    let base_path =
        "D:\\Documents\\Competitions\\ChallengeCup\\NWPU_YRCC2\\datasets\\resized_labels";
    let mut color_class_map = HashMap::<Rgb<u8>, u32>::new();
    color_class_map.insert(Rgb([0, 0, 0]), 0);
    color_class_map.insert(Rgb([0, 255, 0]), 1);
    color_class_map.insert(Rgb([0, 255, 255]), 2);
    color_class_map.insert(Rgb([255, 0, 255]), 3);

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
                    let mask: cocotools::mask::Mask = mask_image_array(&img, *color).into();

                    let rle = Rle::from(&mask);
                    let area = rle.area();
                    let bbox = Bbox::from(&rle);
                    let new_annotation = Annotation {
                        id: 0,
                        image_id: 0,
                        category_id: *class_id,
                        segmentation: Segmentation::Rle(rle),
                        area: area as f64,
                        bbox: bbox,
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
                    *image_guard = *image_guard + 1;
                }

                // Acquire annotation lock
                {
                    let mut annotation_guard = annotation_count.lock().unwrap();
                    for annotion in annotations.iter_mut() {
                        annotion.id = *annotation_guard;
                        *annotation_guard = *annotation_guard + 1;
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

    while let Some(_) = threads.join_next().await {}

    let dataset_guard = dataset.lock().unwrap();
    let dataset_string = serde_json::to_string(&*dataset_guard).unwrap();
    fs::write(
        "D:\\Documents\\Competitions\\ChallengeCup\\NWPU_YRCC2\\datasets\\resized_labels.json",
        dataset_string,
    )
    .unwrap();
}

#[tokio::main]
async fn main() {
    let base_path =
        "D:\\Documents\\Competitions\\ChallengeCup\\NWPU_YRCC2\\datasets\\resized_labels";
    // let BASE_PATH = "./inputs";
    let mut color_class_map = HashMap::<Rgb<u8>, u32>::new();
    color_class_map.insert(Rgb([0, 0, 0]), 0);
    color_class_map.insert(Rgb([0, 255, 0]), 1);
    color_class_map.insert(Rgb([0, 255, 255]), 2);
    color_class_map.insert(Rgb([255, 0, 255]), 3);

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

                let mut labels = Vec::<String>::new();
                for (color, class_id) in color_class_map.clone().iter() {
                    // Output image for this class
                    let mut output_img = img.clone();
                    // TODO: remove this test only code
                    // if color != &Rgb([0, 255, 0]) {
                    //     continue;
                    // }

                    let mut mat = opencv::core::Mat::new_rows_cols_with_default(
                        512,
                        512,
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
                        imgproc::CHAIN_APPROX_SIMPLE,
                    )
                    .unwrap();

                    // println!("{:?}", contours);
                    // println!("{:?}", hierarchy);

                    let mut combined_contours: Vec<Vec<(i32, i32)>> = Vec::new();

                    // Now go through all the hierarchy and combine contours
                    let mut current_index: i32 = 0;
                    while current_index != -1 && contours.len() > 0 {
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
                                child_contour_index = *child_hierarchy.get(0).unwrap();
                                if child_contour_index == -1 {
                                    break;
                                }
                            }
                        }
                        // No more child
                        if parent_points.len() > 3 {
                            // Can't form valid polygon
                            combined_contours.push(parent_points);
                        }

                        current_index = *current_hierarchy.get(0).unwrap();
                    }

                    // println!("{:?}", combined_contours);

                    // TODO: Remove this test only code
                    // let mut count = 0;
                    // for contour in combined_contours.iter() {
                    //     let result_img = imageproc::drawing::draw_antialiased_polygon(
                    //         &output_img,
                    //         contour
                    //             .iter()
                    //             .map(|point| Point {
                    //                 x: point.1 as i32,
                    //                 y: point.0 as i32,
                    //             })
                    //             .collect::<Vec<Point<i32>>>()
                    //             .as_slice(),
                    //         Rgb([255, 0, 0]),
                    //         interpolate
                    //     );
                    //     result_img
                    //         .save(format!(
                    //             "./outputs/images/{}/{}_{}",
                    //             class_id,
                    //             count,
                    //             entry.file_name().into_string().unwrap().to_string()
                    //         ))
                    //         .unwrap();
                    //     count = count + 1;
                    // }

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
                    }
                    output_img
                        .save(format!(
                            "./outputs/images/{}/{}",
                            class_id,
                            entry.file_name().into_string().unwrap()
                        ))
                        .unwrap();
                }
                fs::write(
                    format!(
                        "./outputs/labels/{}",
                        entry
                            .file_name()
                            .into_string()
                            .unwrap()
                            .to_string()
                            .replace(".png", ".txt")
                    ),
                    labels.concat(),
                )
                .unwrap();
                println!(
                    "{} finished process",
                    entry.file_name().into_string().unwrap()
                );
            });
        }
    }

    while let Some(_) = threads.join_next().await {}
}
