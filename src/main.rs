
mod labels;
mod images;

#[tokio::main]
async fn main() {
    // images::normalize_color().await;
    // images::check_valid_pixel_count();
    // images::split_images().await;
    // labels::rgb2yolo().await;
    // labels::split_dataset();
    labels::gray_filter().await;
}
