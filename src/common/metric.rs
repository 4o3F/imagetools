use opencv::{
    core::{self, MatTraitConst, CV_8U},
    imgcodecs,
};
use rayon::prelude::*;
use rayon_progress::ProgressAdaptor;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use tracing_unwrap::ResultExt;

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

    let confusion_matrix: Arc<Mutex<HashMap<(u8, u8, u8), HashMap<(u8, u8, u8), i64>>>> =
        Arc::new(Mutex::new(HashMap::new()));

    let rows = gt_img.rows();
    let cols = gt_img.cols();

    let row_iter = ProgressAdaptor::new(0..rows);
    let row_progress = row_iter.items_processed();
    let row_total = row_iter.len();
    row_iter.for_each(|i| {
        let mut row_confusion_matrix: HashMap<(u8, u8, u8), HashMap<(u8, u8, u8), i64>> =
            HashMap::new();
        for j in 0..cols {
            let predicted_pixel = target_img
                .at_2d::<core::Vec3b>(i, j)
                .expect_or_log("Get output pixel error");
            let true_pixel = gt_img
                .at_2d::<core::Vec3b>(i, j)
                .expect_or_log("Get ground truth pixel error");

            let predicted_color = (predicted_pixel[0], predicted_pixel[1], predicted_pixel[2]);
            let true_color = (true_pixel[0], true_pixel[1], true_pixel[2]);

            {
                let entry = row_confusion_matrix
                    .entry(true_color)
                    .or_insert_with(HashMap::new);

                *entry.entry(predicted_color).or_insert(0) += 1;
            }
        }
        let mut confusion_matrix = confusion_matrix.lock().unwrap();

        for (true_color, value) in row_confusion_matrix.into_iter() {
            let entry = confusion_matrix
                .entry(true_color)
                .or_insert_with(HashMap::new);
            for (predicted_color, count) in value.into_iter() {
                *entry.entry(predicted_color).or_insert(0) += count;
            }
        }
        if row_progress.get() != 0 && row_progress.get() % 1000 == 0 {
            tracing::info!("Row {} / {} done", row_progress.get(), row_total);
        }
    });

    let confusion_matrix = confusion_matrix.lock().unwrap();
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

    let mut iou_results: HashMap<(u8, u8, u8), f64> = HashMap::new();
    let mut total_intersection: HashMap<(u8, u8, u8), i64> = HashMap::new();
    let mut total_union: HashMap<(u8, u8, u8), i64> = HashMap::new();

    for (true_color, predictions) in &*confusion_matrix {
        let mut intersection = 0;
        let mut union = 0;

        for (predicted_color, count) in predictions {
            intersection += *count;
            union += count + *total_union.entry(predicted_color.clone()).or_insert(0);
        }

        total_intersection.insert(*true_color, intersection);
        total_union.insert(*true_color, union);

        if union > 0 {
            let iou = intersection as f64 / union as f64;
            iou_results.insert(*true_color, iou);
        }
    }

    tracing::info!("IoU Results:");
    for (color, iou) in &iou_results {
        tracing::info!(
            "Color RGB({},{},{}) IoU: {:.4}",
            color.0,
            color.1,
            color.2,
            iou
        );
    }
}
