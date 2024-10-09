# Image Tools
Batch image process tools mainly designed for dataset preprocessing

## Current tools

### Global args

- `thread`  The thread pool size for parallel operations

### Common

- `crop-rectangle`            Crop a rectangle region of the image
- `normalize`                 Normalize image to given range
- `map-color`                 Map one RGB color to another in a given PNG image file
- `map-background-color`      Map all the non-valid color in the image to a given color
- `split-images`              Split large images to small pieces for augmentation purposes
- `split-images-with-bias`    Split large images to small pieces for augmentation purposes with bias (Bias is added between each split)
- `split-images-with-filter`  Split large images to small pieces with a filter for enough valid pixels
- `class2rgb`                 Map 8 bit grayscale PNG class image to RGB image
- `rgb2class`                 Map RGB image to 8 bit grayscale PNG class image
- `resize-images`             Resize all images in a given folder to a given size with a given filter
- `rgb2rle`                   Convert RGB semantic segmentation PNG labels to RLE format
- `split-dataset`             Split dataset into train and test sets
- `count-classes`             Count class for 8 bit PNG image & Calc class balance weight
- `strip-image-edge`          Strip image edges
- `stich-images`              Stich the splited images back together
- `calc-mean-std`             Calc the mean and std of a dataset for normalization
- `calc-iou`                  Calc the IoU of two images


### Yolo

- `split-dataset`  Split dataset into train and test sets Will store result in TXT file
- `count-types`    Count the object number of each type in the dataset
- `rgb2yolo`       Convert RGB labels to YOLO TXT format