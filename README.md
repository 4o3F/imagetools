# Image Tools
Some self used batch image process tools

## Current tools

### Common

- `map-color`                 Map one RGB color to another in a given PNG image file
- `map-color-dir`             Map one RGB color to another for all PNG images in a given folder
- `split-images-with-bias`    Split large images to small pieces for augmentation purposes
- `split-images-with-filter`  Split large images to small pieces with a filter for enough valid pixels
- `class2rgb`                 Map 8 bit grayscale PNG class image to RGB image
- `resize-images`             Resize all images in a given folder to a given size with a given filter
- `rgb2rle`                   Convert RGB semantic segmentation PNG labels to RLE format


### Yolo

- `split-dataset`  Split dataset into train and test sets Will store result in TXT file
- `count-types`    Count the object number of each type in the dataset
- `rgb2yolo`       Convert RGB labels to YOLO TXT format

## TODO
- [x] Fix hole issue (maybe change from imageproc to opencv contour detection)