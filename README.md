# Object-Detection

# African Wildlife Object Detection

This project focuses on object detection of African wildlife using the YOLOv8 model. The goal is to detect and classify animals such as elephants, buffalos, rhinos, and zebras in various images, with high accuracy and reliability.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Conclusion](#conclusion)
- [Bonus](#bonus)
- [References](#references)

## Introduction

Object detection is a critical task in computer vision with numerous applications. This project leverages the YOLOv8 (You Only Look Once version 8) model, known for its speed and accuracy, to identify and classify African wildlife species from images.

## Dataset

The dataset used for this project is the [African Wildlife Dataset](https://www.kaggle.com/datasets/rohanrao/african-wildlife), which includes images and corresponding labels for various animals.

### Data Preprocessing

1. **Resizing Images**: Images are resized to 640x640 pixels for uniformity.
2. **Splitting Data**:
   - **Training Set**: 60% of the data.
   - **Validation Set**: 20% of the data.
   - **Test Set**: 20% of the data.
3. **Handling Multiple Objects**: Images with multiple objects are specifically moved to the test set to evaluate the model's performance on complex scenarios.

## Setup

### Prerequisites

- Python 3.8+
- Libraries: `torch`, `opencv-python`, `numpy`, `matplotlib`, `shutil`
- YOLOv8 (Installation instructions available in the official [YOLOv8 repository](https://github.com/ultralytics/yolov8))
