use std::{collections::HashMap, fs, sync::Arc};

use image::Rgb;
use opencv::{core::MatTrait, imgproc};
use tokio::{fs::File, io::AsyncWriteExt, sync::Semaphore, task::JoinSet};
use tracing_unwrap::ResultExt;

use crate::THREAD_POOL;

pub async fn rgb2yolo(dataset_path: &String, rgb_list: &str) {
    let mut color_class_map = HashMap::<Rgb<u8>, u32>::new();
    // 卫星数据
    // color_class_map.insert(Rgb([0, 0, 0]), 0);
    // color_class_map.insert(Rgb([0, 255, 0]), 1);

    // color_class_map.insert(Rgb([255, 255, 0]), 2);
    // color_class_map.insert(Rgb([255, 0, 255]), 3);
    // color_class_map.insert(Rgb([0, 255, 255]), 4);
    // color_class_map.insert(Rgb([0, 0, 255]), 5);
    // color_class_map.insert(Rgb([255, 255, 255]), 6);

    // YRCC_MS without land
    color_class_map.insert(Rgb([0, 255, 0]), 0);
    color_class_map.insert(Rgb([255, 255, 0]), 1);
    color_class_map.insert(Rgb([255, 0, 255]), 2);
    color_class_map.insert(Rgb([0, 255, 255]), 3);
    color_class_map.insert(Rgb([0, 0, 255]), 4);
    color_class_map.insert(Rgb([255, 255, 255]), 5);

    // YRCC_ORI
    // color_class_map.insert(Rgb([0, 255, 255]), 0); // ice
    // color_class_map.insert(Rgb([0, 255, 0]), 1); // water

    // 航拍数据
    // color_class_map.insert(Rgb([0, 0, 0]), 0); // land
    // color_class_map.insert(Rgb([0, 255, 0]), 1);
    // color_class_map.insert(Rgb([0, 255, 255]), 1); // shore_ice
    // color_class_map.insert(Rgb([255, 0, 255]), 2); // stream_ice

    // Albert
    // color_class_map.insert(Rgb([0, 0, 0]), 1);
    // color_class_map.insert(Rgb([128, 128, 128]), 0);
    // color_class_map.insert(Rgb([255, 255, 255]), 2);

    for (class_id, rgb) in rgb_list.split(";").enumerate() {
        let mut rgb_vec: Vec<u8> = vec![];
        for splited in rgb.split(',') {
            let splited = splited.parse::<u8>().unwrap();
            rgb_vec.push(splited);
        }
        color_class_map.insert(Rgb([rgb_vec[0], rgb_vec[1], rgb_vec[2]]), class_id as u32);
    }

    fs::create_dir_all(format!("{}/../outputs/", dataset_path))
        .expect_or_log("Create output dir error");

    let mut threads = JoinSet::new();
    let sem = Arc::new(Semaphore::new(
        (*THREAD_POOL.read().expect_or_log("Get pool error")).into(),
    ));

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
            let color_class_map = color_class_map.clone();
            let dataset_path = dataset_path.clone();
            threads.spawn(async move {
                // Limit tasks to 10
                let _permit = permit.acquire().await.unwrap();

                let img: image::ImageBuffer<Rgb<u8>, Vec<u8>> =
                    image::open(entry.path()).unwrap().into_rgb8();

                let mut labels = Vec::<String>::new();
                for (color, class_id) in color_class_map.clone().iter() {

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
                                if child_points.len() > 3 {
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
                                }
                                child_contour_index = *child_hierarchy.first().unwrap();
                                if child_contour_index == -1 {
                                    break;
                                }
                            }
                        }
                        // No more child
                        if parent_points.len() > 10 {
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
                    "{}/../outputs/{}",
                    dataset_path,
                    entry
                        .file_name()
                        .into_string()
                        .unwrap()
                        .to_string()
                        .replace(".png", ".txt")
                        .replace("v2_", "")
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
