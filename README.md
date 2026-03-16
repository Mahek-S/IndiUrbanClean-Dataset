# IndiUrbanClean Dataset

## About the Dataset

The **IndiUrbanClean Dataset** is a curated collection of images that represent different cleanliness conditions commonly observed in urban environments. The dataset focuses on situations such as clean streets, open waste dumping, construction debris, dumpyards, and overflowing dustbins. These categories reflect typical urban waste scenarios that can be seen in many cities.

The dataset was created to support basic computer vision tasks, particularly image classification related to urban cleanliness and waste detection. By learning from these images, machine learning models can potentially be used to identify waste conditions in public spaces and support applications related to urban monitoring and environmental awareness.

Images were gathered from publicly available web sources and then manually reviewed and organized into the defined categories to maintain consistency within each class.

## Classes

The dataset includes the following categories:

* clean_street
* open_waste
* construction_waste
* dumpyard
* overfilled_bins

Each image in the dataset belongs to one of these classes.

## Folder Structure

IndiUrbanClean Dataset
│
├── images/
│   ├── clean_street/
│   ├── open_waste/
│   ├── construction_waste/
│   ├── dumpyard/
│   └── overfilled_bins/
│
├── annotations/
│
├── metadata/
│   └── image_metadata.csv
│
├── splits/
│   ├── train/
│   ├── val/
│   └── test/
│
├── docs/
|    ├── dataset_collection.md
|    └── annotation_guidelines.md
│
├── README.md
└── LICENSE

## Metadata

Metadata for the images is available in **metadata/image_metadata.csv**.
This file includes information such as image id, filename, class label, annotator, collection date, and image source.

## Image Sources

Images were collected from publicly available sources such as Google Images, Bing, Flickr, Wikimedia Commons, Unsplash, and Pexels.

## Usage

The dataset can be used for computer vision experiments related to urban cleanliness classification, waste detection, and similar environmental monitoring tasks.

## License

Refer to the LICENSE file included with the dataset.
