use opencv::{
    core::{self, MatTraitConst, CV_8U},
    imgcodecs,
};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use tracing_unwrap::ResultExt;

#[tracing::instrument]
pub fn calc_iou(target_img: &String, gt_img: &String) {
    tracing::info!("Start loading images");
    let target_img = imgcodecs::imread(&target_img, imgcodecs::IMREAD_COLOR)
        .expect_or_log("Open output image error");
    let gt_img = imgcodecs::imread(&gt_img, imgcodecs::IMREAD_COLOR)
        .expect_or_log("Open ground truth image error");

    tracing::info!("Image loaded");
    if target_img.depth() != CV_8U || gt_img.depth() != CV_8U {
        tracing::error!("Output image and ground truth image must be 8-bit 3-channel images");
        return;
    }

    let intersection: Arc<Mutex<HashMap<(u8, u8, u8), i64>>> = Arc::new(Mutex::new(HashMap::new()));
    let union: Arc<Mutex<HashMap<(u8, u8, u8), i64>>> = Arc::new(Mutex::new(HashMap::new()));
    let confusion_matrix: Arc<Mutex<HashMap<(u8, u8, u8), HashMap<(u8, u8, u8), i64>>>> =
        Arc::new(Mutex::new(HashMap::new()));

    let rows = gt_img.rows();
    let cols = gt_img.cols();

    let span = tracing::span!(tracing::Level::INFO, "calc_iou");
    let _enter = span.enter();

    (0..rows).into_par_iter().for_each(|i| {
        let mut row_intersection: HashMap<(u8, u8, u8), i64> = HashMap::new();
        let mut row_union: HashMap<(u8, u8, u8), i64> = HashMap::new();
        let mut row_confusion_matrix: HashMap<(u8, u8, u8), HashMap<(u8, u8, u8), i64>> =
            HashMap::new();
        for j in 0..cols {
            let pixel1 = target_img
                .at_2d::<core::Vec3b>(i, j)
                .expect_or_log("Get output pixel error");
            let pixel2 = gt_img
                .at_2d::<core::Vec3b>(i, j)
                .expect_or_log("Get ground truth pixel error");

            let color1 = (pixel1[0], pixel1[1], pixel1[2]);
            let color2 = (pixel2[0], pixel2[1], pixel2[2]);

            {
                *row_union.entry(color1).or_insert(0) += 1;
                *row_union.entry(color2).or_insert(0) += 1;

                if pixel1 == pixel2 {
                    *row_intersection.entry(color1).or_insert(0) += 1;
                }

                let entry = row_confusion_matrix
                    .entry(color2)
                    .or_insert_with(HashMap::new);
                *entry.entry(color1).or_insert(0) += 1;
            }
        }

        let mut intersection = intersection.lock().unwrap();
        let mut union = union.lock().unwrap();
        let mut confusion_matrix = confusion_matrix.lock().unwrap();

        for (color, value) in row_intersection.into_iter() {
            *intersection.entry(color).or_insert(0) += value;
        }
        for (color, value) in row_union.into_iter() {
            *union.entry(color).or_insert(0) += value;
        }
        for (color, value) in row_confusion_matrix.into_iter() {
            *confusion_matrix.entry(color).or_insert_with(HashMap::new) = value;
        }
        tracing::trace!("Y {} done", i);
    });

    let mut iou = HashMap::new();
    let mut total_iou = 0.0;
    let mut num_categories = 0;

    let intersection = intersection.lock().unwrap();
    let union = union.lock().unwrap();
    let confusion_matrix = confusion_matrix.lock().unwrap();
    for (&color, &inter) in &*intersection {
        let uni = union.get(&color).unwrap_or(&0);
        if *uni > 0 {
            let iou_value = inter as f64 / *uni as f64;
            iou.insert(color, iou_value);
            total_iou += iou_value;
            num_categories += 1;
        }
    }

    let mean_iou = if num_categories > 0 {
        total_iou / num_categories as f64
    } else {
        0.0
    };

    for (color, &iou_value) in &iou {
        tracing::info!(
            "IoU for color RGB({},{},{}): {}",
            color.0,
            color.1,
            color.2,
            iou_value
        );
    }
    tracing::info!("Mean IoU: {}", mean_iou);
    tracing::info!("Confusion Matrix:");
    for (true_color, predictions) in &*confusion_matrix {
        for (predicted_color, count) in predictions {
            tracing::info!(
                "True: RGB({},{},{}) Predicted: RGB({},{},{}) Count: {}",
                true_color.0,
                true_color.1,
                true_color.2,
                predicted_color.0,
                predicted_color.1,
                predicted_color.2,
                count
            );
        }
    }
}
