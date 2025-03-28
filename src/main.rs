use std::sync::{LazyLock, RwLock};

use clap::{ArgAction, Parser, Subcommand};
use common::operation::EdgePosition;
use tracing::level_filters::LevelFilter;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use tracing_unwrap::ResultExt;

mod common;
mod yolo;

static THREAD_POOL: LazyLock<RwLock<u16>> = LazyLock::new(|| RwLock::new(100));

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Commands
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(long, default_value = "10", help = "Thread pool size")]
    thread: u16,
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

        #[arg(
            short,
            long,
            help = "The path for the original image / Directory containing images"
        )]
        dataset_path: String,
    },

    /// Map all the non-valid color in the image to a given color
    MapBackgroundColor {
        #[arg(short, long, help = "In R1,G1,B1;R2,G2,B2 format")]
        valid_colors: String,

        #[arg(short, long, help = "In R,G,B format")]
        new_color: String,

        #[arg(short, long, help = "The path for the original image")]
        dataset_path: String,
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
    /// (Bias is added between each split)
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

    /// Split label images to small pieces with a filter for enough valid pixels
    #[command(name = "split-images-with-rgb-filter")]
    SplitImagesWithRGBFilter {
        #[arg(short, long, help = "The path for the folder containing images")]
        images_path: String,

        #[arg(short, long, help = "RGB list, in R0,G0,B0;R1,G1,B1 format")]
        rgb_list: String,

        #[arg(long = "height", help = "Height for each split")]
        target_height: u32,

        #[arg(long = "width", help = "Width for each split")]
        target_width: u32,

        #[arg(short, help = "Use valid RGB filter mode", default_value = "false", action = ArgAction::SetTrue)]
        valid_rgb_mode: bool,
    },

    /// Split images to small pieces with a filter for label name match
    #[command(name = "split-images-with-label-filter")]
    SplitImagesWithLabelFilter {
        #[arg(short, long, help = "The path for the folder containing images")]
        images_path: String,

        #[arg(
            short,
            long,
            help = "The path for the folder containing labels, should be same as images folder"
        )]
        labels_path: String,

        #[arg(long = "height", help = "Height for each split")]
        target_height: u32,

        #[arg(long = "width", help = "Width for each split")]
        target_width: u32,
    },

    /// Filter dataset with RGB list
    #[command(name = "filter-dataset-with-rgblist")]
    FilterDatasetWithRGBList {
        #[arg(
            short,
            long,
            help = "The path for the folder containing dataset, should contain images and labels folders"
        )]
        dataset_path: String,

        #[arg(short, long, help = "RGB list, in R0,G0,B0;R1,G1,B1 format")]
        rgb_list: String,

        #[arg(short, help = "Use valid RGB filter mode", default_value = "false", action = ArgAction::SetTrue)]
        valid_rgb_mode: bool,
    },

    /// Map 8 bit grayscale PNG class image to RGB image
    #[command(name = "class2rgb")]
    Class2RGB {
        #[arg(
            short,
            long,
            help = "The path for the folder containing images / The path of the image"
        )]
        dataset_path: String,

        #[arg(short, long, help = "List of RGB colors, in R0,G0,B0;R1,G1,B1 format")]
        rgb_list: String,
    },

    /// Map RGB image to 8 bit grayscale PNG class image
    #[command(name = "rgb2class")]
    RGB2Class {
        #[arg(
            short,
            long,
            help = "The path for the folder containing images / The path of the image"
        )]
        dataset_path: String,

        #[arg(short, long, help = "List of RGB colors, in R0,G0,B0;R1,G1,B1 format")]
        rgb_list: String,
    },

    /// Resize all images in a given folder to a given size with a given filter
    ResizeImages {
        #[arg(
            short,
            long,
            help = "The path of the image / directory containing images"
        )]
        dataset_path: String,

        #[arg(long, help = "Target height")]
        height: i32,

        #[arg(long, help = "Target width")]
        width: i32,

        #[arg(
            short,
            long,
            help = "Filter type, could be `nearest`, `linear`, `cubic` or `lanczos`"
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

    /// Generate CSV format dataset list compatible with huggingface dataset library
    GenerateDatasetCSV {
        #[arg(
            short,
            long,
            help = "The path for the dataset root folder, should contain images and labels folders"
        )]
        dataset_path: String,

        #[arg(
            short,
            long,
            help = "The ratio of train set, should be between 0 and 1"
        )]
        train_ratio: f32,
    },

    /// Generate JSON format dataset list compatible with huggingface dataset library
    #[command(name = "generate-dataset-json")]
    GenerateDatasetJSON {
        #[arg(
            short,
            long,
            help = "The path for the dataset root folder, should contain images and labels folders"
        )]
        dataset_path: String,

        #[arg(
            short,
            long,
            help = "The ratio of train set, should be between 0 and 1"
        )]
        train_ratio: f32,
    },

    /// Split dataset into train and test sets and save file names to txt file, for yolo dataset
    #[command(name = "generate-dataset-txt")]
    GenerateDatasetTXT {
        #[arg(
            short,
            long,
            help = "The path for the dataset root folder, should contain images and labels folders"
        )]
        dataset_path: String,

        #[arg(
            short,
            long,
            help = "The ratio of train set, should be between 0 and 1"
        )]
        train_ratio: f32,
    },

    #[command(name = "txt2json")]
    TXT2JSON {
        #[arg(short, long, help = "TXT file path")]
        txt_path: String,
    },

    /// Combine multiple JSON format dataset list compatible with huggingface dataset library
    #[command(name = "combine-dataset-json")]
    CombineDatasetJSON {
        #[arg(
            short,
            long,
            help = "Multiple paths for the dataset root folders, should contain pregenerated json files"
        )]
        dataset_path: Vec<String>,

        #[arg(short, long, help = "Save path for combined JSON file")]
        save_path: String,
    },

    /// Split dataset into train and test sets inplace
    SplitDataset {
        #[arg(
            short,
            long,
            help = "The path for the dataset root folder, should contain images and labels folders"
        )]
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

    /// Count class for 8 bit PNG image & Calc class balance weight
    CountRGB {
        #[arg(short, long, help = "The path for the folder containing images")]
        dataset_path: String,
        #[arg(short, long, help = "RGB list, in R0,G0,B0;R1,G1,B1 format")]
        rgb_list: String,
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

    /// Stich the splited images back together
    StichImages {
        #[arg(
            short,
            long,
            help = "The path for the folder containing splited images"
        )]
        image_output_path: String,

        #[arg(long, help = "The stiched image height")]
        target_height: i32,

        #[arg(long, help = "The stiched image width")]
        target_width: i32,
    },

    /// Calc the mean and std of a dataset for normalization
    CalcMeanStd {
        #[arg(short, long, help = "The path for the folder containing images")]
        dataset_path: String,
    },

    /// Calc the IoU of two images
    #[command(name = "calc-iou")]
    CalcIoU {
        #[arg(short, long, help = "The path for the target image")]
        target_image: String,

        #[arg(short, long, help = "The path for the ground truth image")]
        gt_image: String,
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
    // Do tracing init
    let indicatif_layer = IndicatifLayer::new();
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_writer(indicatif_layer.get_stderr_writer())
        .with_level(true);
    let filter_layer = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .with(indicatif_layer)
        .with(tracing_tracy::TracyLayer::default())
        .init();

    let cli = Cli::parse();

    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.thread.into())
        .build_global()
        .unwrap();

    *THREAD_POOL
        .write()
        .expect_or_log("Get thread pool lock failed") = cli.thread;

    tracing::info!("Using {} threads", cli.thread);

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
                dataset_path,
            } => {
                common::remap::remap_color(original_color, new_color, dataset_path).await;
            }
            CommonCommands::MapBackgroundColor {
                valid_colors,
                new_color,
                dataset_path,
            } => {
                common::remap::remap_background_color(valid_colors, new_color, dataset_path).await;
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
            CommonCommands::SplitImagesWithRGBFilter {
                images_path,
                target_height,
                target_width,
                rgb_list,
                valid_rgb_mode,
            } => {
                common::augment::split_images_with_rgb_filter(
                    images_path,
                    target_height,
                    target_width,
                    rgb_list,
                    *valid_rgb_mode,
                )
                .await;
            }
            CommonCommands::SplitImagesWithLabelFilter {
                images_path,
                labels_path,
                target_height,
                target_width,
            } => {
                common::augment::split_images_with_label_filter(
                    images_path,
                    labels_path,
                    target_height,
                    target_width,
                )
                .await;
            }
            CommonCommands::FilterDatasetWithRGBList {
                dataset_path,
                rgb_list,
                valid_rgb_mode,
            } => {
                common::augment::filter_dataset_with_rgblist(
                    dataset_path,
                    rgb_list,
                    *valid_rgb_mode,
                )
                .await;
            }
            CommonCommands::StichImages {
                image_output_path,
                target_height,
                target_width,
            } => {
                common::augment::stich_images(image_output_path, target_height, target_width).await;
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
            CommonCommands::GenerateDatasetCSV {
                dataset_path,
                train_ratio,
            } => {
                common::dataset::generate_dataset_csv(dataset_path, train_ratio);
            }
            CommonCommands::GenerateDatasetJSON {
                dataset_path,
                train_ratio,
            } => {
                common::dataset::generate_dataset_json(dataset_path, train_ratio);
            }
            CommonCommands::TXT2JSON { txt_path } => {
                common::dataset::txt2json(txt_path);
            }
            CommonCommands::CombineDatasetJSON {
                dataset_path,
                save_path,
            } => {
                common::dataset::combine_dataset_json(dataset_path, save_path);
            }
            CommonCommands::GenerateDatasetTXT {
                dataset_path,
                train_ratio,
            } => {
                common::dataset::generate_dataset_txt(dataset_path, train_ratio).await;
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
            CommonCommands::CountRGB {
                dataset_path,
                rgb_list,
            } => {
                common::dataset::count_rgb(dataset_path, rgb_list).await;
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
            CommonCommands::CalcIoU {
                target_image,
                gt_image,
            } => {
                common::metric::calc_iou(target_image, gt_image);
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
