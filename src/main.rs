use std::{
    collections::HashMap,
    fs,
    sync::{Arc, Mutex},
};

use cocotools::{
    coco::{
        self,
        object_detection::{
            Annotation, Bbox, Category, Dataset, HashmapDataset, Image, Rle, Segmentation,
        },
    },
    mask::{conversions::convert_coco_segmentation, utils::Area},
    COCO,
};
use image::{imageops::FilterType, GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use imageproc::{
    contours::{BorderType, Contour},
    pixelops::interpolate,
    point::Point,
};
use rand::Rng;
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
    let BASE_PATH =
        "D:\\Documents\\Competitions\\ChallengeCup\\NWPU_YRCC2\\datasets\\resized_labels";
    let mut COLOR_CLASS_MAP = HashMap::<Rgb<u8>, u32>::new();
    COLOR_CLASS_MAP.insert(Rgb([0, 0, 0]), 0);
    COLOR_CLASS_MAP.insert(Rgb([0, 255, 0]), 1);
    COLOR_CLASS_MAP.insert(Rgb([0, 255, 255]), 2);
    COLOR_CLASS_MAP.insert(Rgb([255, 0, 255]), 3);

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
    let entries = fs::read_dir(BASE_PATH).unwrap();
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
            let color_class_map = COLOR_CLASS_MAP.clone();
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
    let BASE_PATH =
        "D:\\Documents\\Competitions\\ChallengeCup\\NWPU_YRCC2\\datasets\\resized_labels_test";
    let mut COLOR_CLASS_MAP = HashMap::<Rgb<u8>, u32>::new();
    COLOR_CLASS_MAP.insert(Rgb([0, 0, 0]), 0);
    COLOR_CLASS_MAP.insert(Rgb([0, 255, 0]), 1);
    COLOR_CLASS_MAP.insert(Rgb([0, 255, 255]), 2);
    COLOR_CLASS_MAP.insert(Rgb([255, 0, 255]), 3);

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
    let entries = fs::read_dir(BASE_PATH).unwrap();
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
            let color_class_map = COLOR_CLASS_MAP.clone();
            threads.spawn(async move {
                // Limit tasks to 10
                let _permit = permit.acquire().await.unwrap();

                let img = image::open(entry.path()).unwrap().into_rgb8();
                // First process the image without acquiring the lock, boost performance

                let mut labels = Vec::<String>::new();
                for (color, class_id) in color_class_map.clone().iter() {
                    // Output image for this class
                    let mut output_img = img.clone();
                    // Test only
                    if color != &Rgb([255, 0, 255]) {
                        continue;
                    }

                    // Turn rgb label to gray image mask
                    let mut mask = GrayImage::new(img.width(), img.height());
                    for (x, y, pixel) in img.enumerate_pixels() {
                        let Rgb([r, g, b]) = pixel;
                        let Rgb([tr, tg, tb]) = color;
                        if r == tr && g == tg && b == tb {
                            mask.put_pixel(x, y, Luma([255]))
                        } else {
                            mask.put_pixel(x, y, Luma([0]))
                        }
                    }

                    mask.save("./mask.png").unwrap();

                    // Find and filter out contours on the mask
                    let mut contours: Vec<Contour<i32>> =
                        imageproc::contours::find_contours::<i32>(&mask);

                    for contour in contours.iter() {
                        if contour.points[0].x == 14 && contour.points[0].y == 174 {
                            println!("{:?}", contour);
                        }
                    }

                    let mut combined_contours: Vec<Contour<i32>> = contours.clone();

                    let mut count = 0;

                    {
                        let mut index: usize = combined_contours.len();
                        for contour in contours.iter_mut().rev() {
                            if contour.points.len() < 3 {
                                combined_contours.remove(index - 1);
                            } else if contour.parent.is_some()
                                && contour.border_type == BorderType::Hole
                            {
                                let parent: usize = contour.parent.unwrap();

                                // Combine two contours
                                // loop through parent and child points, find the min distance point
                                let mut min_distance = f64::MAX;
                                let mut child_index = 0;
                                let mut parent_index = 0;
                                for (i, parent_point) in
                                    combined_contours[parent].points.iter().enumerate()
                                {
                                    for (j, child_point) in contour.points.iter().enumerate() {
                                        //println!("x distance: {:?}", (parent_point.x - child_point.x));
                                        let distance = f64::from(
                                            (parent_point.x - child_point.x).pow(2)
                                                + (parent_point.y - child_point.y).pow(2),
                                        )
                                        .sqrt();
                                        if distance < min_distance {
                                            min_distance = distance;
                                            parent_index = i;
                                            child_index = j;
                                        }
                                    }
                                }

                                // combine two contours
                                let mut new_points = Vec::<Point<i32>>::new();
                                new_points.extend(
                                    combined_contours[parent]
                                        .points
                                        .iter()
                                        .take(parent_index + 1),
                                );
                                new_points.extend(contour.points.iter().skip(child_index));
                                new_points.extend(contour.points.iter().take(child_index + 1));
                                new_points.extend(
                                    combined_contours[parent]
                                        .points
                                        .iter()
                                        .skip(parent_index + 1),
                                );
                                combined_contours[parent].points = new_points;
                                combined_contours.remove(index - 1);
                            }
                            index = index - 1;
                        }
                    }

                    // Combine child contours to parent contours
                    for contour in combined_contours {
                        let mut result = String::new();
                        result.push_str(class_id.to_string().as_str());
                        result.push(' ');
                        if contour.points.len() < 3 {
                            continue;
                        }
                        contour.points.iter().for_each(|point| {
                            result.push_str(&format!(
                                "{} ",
                                (f64::from(point.x) / f64::from(img.width()))
                            ));
                            result.push_str(&format!(
                                "{} ",
                                f64::from(point.y) / f64::from(img.height())
                            ));
                        });

                        if count == 177 {
                            println!("{:?}", contour)
                        }

                        // let result_img = imageproc::drawing::draw_polygon(
                        //     &output_img,
                        //     //contour.points.iter().map(|&point| Point{x:f32::from(point.x),y:f32::from(point.y)}).collect::<Vec<Point<f32>>>().as_slice(),
                        //     &contour.points,
                        //     Rgb([255, 0, 0]),
                        //     // interpolate,
                        // );
                        // result_img
                        //     .save(format!(
                        //         "./outputs/images/{}/{}_{}",
                        //         class_id,
                        //         count,
                        //         entry.file_name().into_string().unwrap().to_string()
                        //     ))
                        //     .unwrap();
                        result.push('\n');
                        labels.push(result);
                        count = count + 1;
                    }
                    // output_img
                    //     .save(format!(
                    //         "./outputs/images/{}/{}",
                    //         class_id,
                    //         entry.file_name().into_string().unwrap().to_string()
                    //     ))
                    //     .unwrap();
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
