use std::collections::HashMap;

use opencv::{
    core::{self, MatTraitConst, CV_8U},
    imgcodecs,
};
use tracing_unwrap::ResultExt;

pub fn calc_iou(target_img: &String, gt_img: &String) {
    let target_img = imgcodecs::imread(&target_img, imgcodecs::IMREAD_COLOR)
        .expect_or_log("Open output image error");
    let gt_img = imgcodecs::imread(&gt_img, imgcodecs::IMREAD_COLOR)
        .expect_or_log("Open ground truth image error");

    tracing::info!("Image loaded");
    if target_img.depth() != CV_8U || gt_img.depth() != CV_8U {
        tracing::error!("Output image and ground truth image must be 8-bit 3-channel images");
        return;
    }

    let mut intersection: HashMap<(u8, u8, u8), usize> = HashMap::new();
    let mut union: HashMap<(u8, u8, u8), usize> = HashMap::new();

    let rows = gt_img.rows();
    let cols = gt_img.cols();

    for i in 0..rows {
        for j in 0..cols {
            let pixel1 = target_img
                .at_2d::<core::Vec3b>(i, j)
                .expect_or_log("Get output pixel error");
            let pixel2 = gt_img
                .at_2d::<core::Vec3b>(i, j)
                .expect_or_log("Get ground truth pixel error");

            let color1 = (pixel1[0], pixel1[1], pixel1[2]);
            let color2 = (pixel2[0], pixel2[1], pixel2[2]);

            // 更新交集和并集
            if pixel1[0] > 0 || pixel1[1] > 0 || pixel1[2] > 0 {
                *union.entry(color1).or_insert(0) += 1;
            }
            if pixel2[0] > 0 || pixel2[1] > 0 || pixel2[2] > 0 {
                *union.entry(color2).or_insert(0) += 1;
            }

            if pixel1 == pixel2 {
                *intersection.entry(color1).or_insert(0) += 1;
            }
        }
    }

    let mut iou = HashMap::new();
    let mut total_iou = 0.0;
    let mut num_categories = 0;

    for (&color, &inter) in &intersection {
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
}
