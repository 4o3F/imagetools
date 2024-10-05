use clap::{Parser, Subcommand};
use common::operation::EdgePosition;
use tracing::Level;
use tracing_unwrap::ResultExt;

mod common;
mod yolo;
// mod geo;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Commands
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Commonly used commands for all kind of dataset
    Common {
        #[command(subcommand)]
        command: CommonCommands,
    },
    /// Yolo specific commands
    Yolo {
        #[command(subcommand)]
        command: YoloCommands,
    },
}
#[derive(Subcommand)]
enum CommonCommands {
    /// Crop a rectangle region of the image
    CropRectangle {
        #[arg(short, long, help = "The path for the original image")]
        image_path: String,
        #[arg(short, long, help = "The path for the cropped new image")]
        save_path: String,
        #[arg(
            short,
            long,
            help = "The corner cords of the rectangle in the format x1,y1;x2,y2 left right corner is 0,0"
        )]
        rectangle: String,
    },

    /// Normalize image to given range
    Normalize {
        #[arg(short, long, help = "The path for the original image")]
        dataset_path: String,
        #[arg(long, help = "The max value for normalization")]
        max: f64,
        #[arg(long, help = "The min value for normalization")]
        min: f64,
    },

    // TODO: Add arg for image extension selection
    /// Map one RGB color to another in a given PNG image file
    MapColor {
        #[arg(short, long, help = "In R,G,B format")]
        original_color: String,

        #[arg(short, long, help = "In R,G,B format")]
        new_color: String,

        #[arg(short, long, help = "The path for the original image")]
        image_path: String,

        #[arg(short, long, help = "The path for the mapped new image")]
        save_path: String,
    },

    /// Map one RGB color to another for all PNG images in a given folder
    MapColorDir {
        #[arg(short, long, help = "In R,G,B format")]
        original_color: String,

        #[arg(short, long, help = "In R,G,B format")]
        new_color: String,

        #[arg(
            short,
            long,
            help = "The path for the folder containing original images"
        )]
        image_path: String,

        #[arg(
            short,
            long,
            help = "The path for the folder to save the mapped new images"
        )]
        save_path: String,
    },

    /// Split large images to small pieces for augmentation purposes
    SplitImages {
        #[arg(short, long, help = "The path for the folder containing images")]
        dataset_path: String,

        #[arg(long = "height", help = "Height for each split")]
        target_height: u32,

        #[arg(long = "width", help = "Width for each split")]
        target_width: u32,
    },

    /// Split large images to small pieces for augmentation purposes with bias
    /// Bias is added between each split
    SplitImagesWithBias {
        #[arg(short, long, help = "The path for the folder containing images")]
        dataset_path: String,

        #[arg(short, long, help = "The bias between each split")]
        bias_step: u32,

        #[arg(long = "height", help = "Height for each split")]
        target_height: u32,

        #[arg(long = "width", help = "Width for each split")]
        target_width: u32,
    },

    /// Split large images to small pieces with a filter for enough valid pixels
    SplitImagesWithFilter {
        #[arg(short, long, help = "The path for the folder containing images")]
        image_path: String,

        #[arg(short, long, help = "The path for the folder containing labels")]
        label_path: String,

        #[arg(short, long, help = "RGB list, in R0,G0,B0;R1,G1,B1 format")]
        rgb_list: String,

        #[arg(long = "height", help = "Height for each split")]
        target_height: u32,

        #[arg(long = "width", help = "Width for each split")]
        target_width: u32,

        #[arg(short, help = "Use valid RGB filter mode", default_value = "true")]
        valid_rgb_mode: bool,

        #[arg(short, help = "Skip label processing", default_value = "false")]
        skip_label_process: bool,
    },

    /// Map 8 bit grayscale PNG class image to RGB image
    #[command(name = "class2rgb")]
    Class2RGB {
        #[arg(short, long, help = "The path for the folder containing images")]
        dataset_path: String,

        #[arg(short, long, help = "List of RGB colors, in R0,G0,B0;R1,G1,B1 format")]
        rgb_list: String,
    },

    /// Map RGB image to 8 bit grayscale PNG class image
    #[command(name = "rgb2class")]
    RGB2Class {
        #[arg(short, long, help = "The path for the folder containing images")]
        dataset_path: String,

        #[arg(short, long, help = "List of RGB colors, in R0,G0,B0;R1,G1,B1 format")]
        rgb_list: String,
    },

    /// Resize all images in a given folder to a given size with a given filter
    ResizeImages {
        #[arg(short, long, help = "The path for the folder containing images")]
        dataset_path: String,

        #[arg(long, help = "Target height")]
        height: u32,

        #[arg(long, help = "Target width")]
        width: u32,

        #[arg(
            short,
            long,
            help = "Filter type, could be `nearest`, `linear`, `cubic`, `gaussian` or `lanczos`"
        )]
        filter: String,
    },

    /// Convert RGB semantic segmentation PNG labels to RLE format
    #[command(name = "rgb2rle")]
    Rgb2Rle {
        #[arg(
            short,
            long,
            help = "The path for the folder containing RGB semantic segmentation PNG labels"
        )]
        dataset_path: String,

        #[arg(
            short,
            long,
            help = "Required RGB list, in R0,G0,B0,class_name;R1,G1,B1,class_name format"
        )]
        rgb_list: String,
    },

    /// Split dataset into train and test sets
    SplitDataset {
        #[arg(short, long, help = "The path for the folder containing dataset files")]
        dataset_path: String,

        #[arg(
            short,
            long,
            help = "The ratio of train set, should be between 0 and 1"
        )]
        train_ratio: f32,
    },

    /// Count class for 8 bit PNG image & Calc class balance weight
    CountClasses {
        #[arg(short, long, help = "The path for the folder containing images")]
        dataset_path: String,
    },

    /// Strip image edges
    StripImageEdge {
        #[arg(short = 'o', long, help = "The path for the images")]
        source_path: String,

        #[arg(short, long, help = "The path to save the images")]
        save_path: String,

        #[arg(
            short,
            long,
            help = "The strip direction, can be top/bottom/left/right"
        )]
        direction: String,

        #[arg(short, long, help = "The strip length")]
        length: i32,
    },

    /// Calc the mean and std of a dataset for normalization
    CalcMeanStd {
        #[arg(short, long, help = "The path for the folder containing images")]
        dataset_path: String,
    },
}

#[derive(Subcommand)]
enum YoloCommands {
    /// Split dataset into train and test sets
    /// Will store result in TXT file
    SplitDataset {
        #[arg(short, long, help = "The path for the folder containing TXT labels")]
        dataset_path: String,

        #[arg(
            short,
            long,
            help = "The ratio of train set, should be between 0 and 1"
        )]
        train_ratio: f32,
    },

    /// Count the object number of each type in the dataset
    CountTypes {
        #[arg(short, long, help = "The path for the folder containing TXT labels")]
        dataset_path: String,
    },

    /// Convert RGB labels to YOLO TXT format
    #[command(name = "rgb2yolo")]
    Rgb2Yolo {
        #[arg(
            short,
            long,
            help = "The path for the folder containing RGB semantic segmentation PNG labels"
        )]
        dataset_path: String,

        #[arg(
            short,
            long,
            help = "Required RGB list, in R0,G0,B0,class_name;R1,G1,B1,class_name format"
        )]
        rgb_list: String,
    },
}

#[tokio::main]
async fn main() {
    // env_logger::init();

    // Do tracing init
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::TRACE)
        .with_level(true)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect_or_log("Init tracing failed");

    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Common { command }) => match command {
            CommonCommands::CropRectangle {
                image_path,
                save_path,
                rectangle,
            } => {
                common::operation::crop_rectangle_region(image_path, save_path, rectangle);
            }
            CommonCommands::Normalize {
                dataset_path,
                max,
                min,
            } => {
                common::operation::normalize(dataset_path, max, min);
            }
            CommonCommands::MapColor {
                original_color,
                new_color,
                image_path,
                save_path,
            } => {
                common::remap::remap_color(original_color, new_color, image_path, save_path);
            }
            CommonCommands::MapColorDir {
                original_color,
                new_color,
                image_path,
                save_path,
            } => {
                common::remap::remap_color_dir(original_color, new_color, image_path, save_path)
                    .await;
            }
            CommonCommands::SplitImages {
                dataset_path,
                target_height,
                target_width,
            } => {
                common::augment::split_images(dataset_path, target_height, target_width).await;
            }
            CommonCommands::SplitImagesWithBias {
                dataset_path,
                bias_step,
                target_height,
                target_width,
            } => {
                common::augment::split_images_with_bias(
                    dataset_path,
                    bias_step,
                    target_height,
                    target_width,
                )
                .await;
            }
            CommonCommands::SplitImagesWithFilter {
                image_path,
                label_path,
                target_height,
                target_width,
                rgb_list,
                valid_rgb_mode,
                skip_label_process
            } => {
                common::augment::split_images_with_filter(
                    image_path,
                    label_path,
                    target_height,
                    target_width,
                    rgb_list,
                    *valid_rgb_mode,
                    *skip_label_process
                )
                .await;
            }
            CommonCommands::Class2RGB {
                dataset_path,
                rgb_list,
            } => {
                common::remap::class2rgb(dataset_path, rgb_list).await;
            }
            CommonCommands::ResizeImages {
                dataset_path,
                height,
                width,
                filter,
            } => {
                common::operation::resize_images(dataset_path, height, width, filter).await;
            }
            CommonCommands::Rgb2Rle {
                dataset_path,
                rgb_list,
            } => {
                common::convert::rgb2rle(dataset_path, rgb_list).await;
            }
            CommonCommands::RGB2Class {
                dataset_path,
                rgb_list,
            } => {
                common::remap::rgb2class(dataset_path, rgb_list).await;
            }
            CommonCommands::SplitDataset {
                dataset_path,
                train_ratio,
            } => {
                common::dataset::split_dataset(dataset_path, train_ratio).await;
            }
            CommonCommands::CountClasses { dataset_path } => {
                common::dataset::count_classes(dataset_path).await;
            }
            CommonCommands::StripImageEdge {
                source_path,
                save_path,
                direction,
                length,
            } => {
                let direction = match direction.as_str() {
                    "top" => EdgePosition::Top,
                    "bottom" => EdgePosition::Bottom,
                    "left" => EdgePosition::Left,
                    "right" => EdgePosition::Right,
                    _ => {
                        tracing::error!("Invalid strip direction");
                        return;
                    }
                };
                common::operation::strip_image_edge(source_path, save_path, &direction, length)
                    .await;
            }
            CommonCommands::CalcMeanStd { dataset_path } => {
                common::dataset::calc_mean_std(dataset_path).await;
            }
        },
        Some(Commands::Yolo { command }) => match command {
            YoloCommands::SplitDataset {
                dataset_path,
                train_ratio,
            } => {
                yolo::dataset::split_dataset(dataset_path, train_ratio).await;
            }
            YoloCommands::CountTypes { dataset_path } => {
                yolo::dataset::count_types(dataset_path).await;
            }
            YoloCommands::Rgb2Yolo {
                dataset_path,
                rgb_list,
            } => {
                yolo::convert::rgb2yolo(dataset_path, rgb_list).await;
            }
        },
        None => {
            tracing::error!("No command specified, use --help for more information");
        }
    }

    return;
}
