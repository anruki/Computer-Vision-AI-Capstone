# Computer-Vision-AI-Capstone
Development of a smart alarm system based in YOLO (You Only Look Once) Algorithm for animal welfare.

## Introduction

This project explores video processing use cases combined with artificial intelligence for livestock management.

The following techniques are applied:

- Preprocessing: Video manipulation and filter application.

- Artificial Intelligence: YOLO algorithm for animal detection.

- Feature Analysis: Extraction of attributes from detected animals, such as speed and position relative to fences.

The preprocessing phase aims to facilitate AI recognition and reduce the computational resources required for running the YOLO algorithm.

The role of multimedia processing in this project is to create a pipeline that transforms video input and uses YOLO detections to extract useful features for animal care.

In total, six programs were developed, each addressing different approaches and scenarios. As the project progresses, new improvements are introduced and tested on various video samples.

YOLO Algorithm (You Only Look Once)

The You Only Look Once (YOLO) algorithm is an open-source system for real-time object detection. It uses a single convolutional neural network (CNN) to detect objects in images.

The network divides the image into regions, predicting bounding boxes and class probabilities for each region. These boxes are then weighted according to the predicted probabilities.

The base algorithm runs at 45 frames per second (FPS) without batch processing on a Titan X GPU.

Architecture:

24 convolutional layers

2 fully connected layers

- 1×1 reduction layers

- 3×3 convolutional layers

The early convolutional layers handle feature extraction, while the fully connected layers predict output probabilities and object coordinates.

The Fast YOLO model uses a 9-layer network, producing a final output prediction tensor of 7×7×30.

**Limitations:**
There are spatial constraints in the prediction boxes, as each cell predicts only two boxes and one class. This limits the number of detectable objects and makes grouped-object detection more challenging.

**Use Case**

For this project, we focused on video processing with YOLO for horse monitoring, in collaboration with Hípica Natur
.

Hípica Natur provided valuable insights into the daily challenges of livestock management that can be addressed through multimedia processing, as well as access to real-world video data from their facilities.

**Why Artificial Intelligence?**

Although integrating Artificial Intelligence adds computational cost to multimedia processing, it provides greater flexibility in recognizing animals across diverse environments.

While horse detection could be achieved through non-AI clustering techniques, this project focuses on monitoring different breeds and sizes of horses, as well as other farm animals (sheep and cows) that share similar visual traits.

The pretrained YOLO model enables effective classification of various animal types regardless of their area or camera angle.
