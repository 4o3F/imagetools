use image::GenericImageView;

mod images;
mod labels;

#[tokio::main]
async fn main() {
    // images::normalize_color().await;
    // images::check_valid_pixel_count();
    // labels::rgb2yolo().await;
    // labels::count_types().await;
    // images::flip_image().await;
    // images::resize_images().await;
    // labels::split_dataset();
    // labels::gray_filter().await;
    // images::split_images().await;
    // images::valid_image_list().await;
    let mut img = image::io::Reader::open("inputs/input.tiff").unwrap();
    img.no_limits();
    img.set_format(image::ImageFormat::Tiff);
    let img = img.decode().unwrap();
    let mut img = img.into_rgba16();
    // tiff::decoder
    // 'outer: for x in 0..img.width() {
    //     for y in 0..img.height() {
    //         let pixel = img.get_pixel(x, y);
    //         if x == 1 && y == 5 {
    //             break 'outer;
    //         }
    //         println!("Pixel at ({}, {}): {:?}", x, y, pixel);
    //     }
    // }
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let output_pixel = pixel.0;
        pixel.0 = [output_pixel[1], output_pixel[2], output_pixel[3], output_pixel[0]];
    }
    img.save("outputs/output.png").unwrap();
    // println!("Image dimensions: {:?}", dims);
}
