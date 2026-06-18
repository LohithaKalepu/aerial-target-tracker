# Aerial Target Tracker

# Drone Object Detection

## Overview

The goal of this task is to detect the drone object from the given videos using a deep learning model. On the high level, the pipeline initially processes all `.mp4` videos in the directory, extracts frames at 5 fps, runs the detector model on each frame, and saves the frames where the drone object is detected.

---

## About the contents

1.  `detections.py` – main pipeline code
2.  `drone_video_1.parquet`, `drone_video_2.parquet` - output parquet datasets
3.  `data.yaml`, `dataset_extraction` - helps extract the dataset and yaml helps during and only during training to read the structure of the dataset
4.  `to_parquet.py` - converts the ouput images into parquet (could not upload the detected images because of github storage issues)

---

## Dataset

This task allows us to choose our own dataset from various sources. For this, I used the **Seraphim Drone Detection Dataset** from Hugging Face:
https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset.

This meets the requirements as it --
1. open source on hugging face and contains various drone images from commercial to synthetic.
2. has a wide range of images ~80K (75K traning and remaining test), that are aggregated from various open source dataset like Kaggle and Roboflow Universe and Hugging Face. 
3. labelled and contains the bounding box information for each drone image

This dataset is extrcted using the script given by owner of the dataset in Hugging Face and can be found in the file `dataset_extraction.py`. 
Then, a yaml file is created called `data.yaml` for training purposes. 

---

## Detector

I used one of the mentioned models in the assignment -- the  **RT-DETR (Real-Time Detection Transformer)** from Ultralytics. Even though it is pre-trained, I wanted to fine-tune it. However, I ran into issues because of computational limitations. I work on an Apple Silicon device, so fine-tuning a deep learning model on 75K Images and doing that for at least the minimum of 5 epoches was heavily time consuming ~ 15 hrs per epoch and 75 hours for 5 epoches. Thus, I decided to use 10% of the dataset for 5 epoches; unexpectedly, running this for one epoch also took approximately 1.5 hrs. So, I decided to stop at one epoch, since the assignment does not require fine-tuning, and used those weights for the detector. 

Finally, Fine-tuning is done using -- 
1. **10% of the dataset**
2. **ran for 1 epoch**
3. used the **best weights** (`best.pt` -- not uploaded in this repo because of storage issues) for inference 

---

## Process

This has been done in the `detections.py` file, and the basic pipeline is as follows -- 
1. Look for all `.mp4` files in the directory -- not hardcoded and works for any number of `.mp4` files in the directory
2. Extract frames at **5 fps** per video file
3. Run the detector on each frame
4. Save frames that have at least one drone object detection and a confidence threshold of 0.5
   * Initially, I set the confidence threshold to be 0.25, however I started to notice many false positives -- the detector would falsely identify a drone object as it got confused with clouds and darkness.
   * The most likely cause of this is the limited fine-tuning of the model.
   * Thus, I increased the confidence threshold to 0.5, prioritizing precision over recall, and ultimately reducing the false positives to improve the quality of the detections. 
6. Store the output annotated images in the following manner:

```
detections/
├── <video_name>/
│   ├── frame_0.jpg
│   ├── frame_5.jpg
│   └── ...
```
---

## Output

Then use a simple python code in the file `to_parquet.py` to convert these images into **parquet format** as required by the assignment, and uploaded to Hugging Face. This can be found at: https://huggingface.co/datasets/LohithaKalepu/CS370_assignment3_detections/

Each Parquet file corresponds to one video and contains:

1. Detected image frames
2. Source video name

---

# Kalman Filter Tracking

## Overview

The goal of this task is to track the drone across the frames that have at least one detection. The Kalman Filter should also be able to deal with missing detections over a small number of consecutive frames. 

---

## Contents

The file that deals with this task is called `tracking.py`. This file contains the video processsing, frames sampling, and tracking code.
The output tracked videos are uploaded to youtube, and can be found via the links -- 
1. https://www.youtube.com/watch?v=hrHfwwPrB_Y -- Drone Tracking 1 video
2. https://www.youtube.com/watch?v=sHNCOep29b4 -- Done Tracking 2 video

---

## Kalman Filter State Design and Noise parameters

The Kalman filter is used to estimate the center position of the drone's bounding box. For this project, I have only used a 2D state model with center as I did not consider the velocity. The state vector is given by state = [cx, cy]. The state transition model is given by the matrix -- [1 0; 0 1]. An important thing is that the predicted state remains the same as the previous state unless overwritted/ corrected by a new measurement. 
The measurement vector is given by z = [c sub x, c sub y] and the measurement matrix is given by -- [1 0; 0 1], meaning the detector directly observes the center position. 
There are two main matrices for measurement noise and process noise. Measurement noise gives the detectors bounding box uncertainty. This set to [5 0; 0 5], a relatively low measure so that the tracker stays aligned with the detector measurements and prevents drifts. Process Noise represents how the state changes from frame to frame, and is set to [10, 0; 0, 10], a moderate value to prevent the tracker from becoming fixed. 

---

## Failure Cases and How the tracker handles missing detections

The main failure case is missing detections as it creates a random drift in trajectory. In some cases, there are missing detections due to cloudy regions (low visiblity), drone getting farther away, and low confidence scores. This is handled by creating a small consequetive gap prediction and predictor reset strategies. To explain clearly, when there are missing detections, the kalman filter performs a prediction step every frames, but when there are missing detections, only the prediction step is used. In this step, the trajectory remains continous as the tracker uses predicted center from the previous state. To visualiza the difference, we also draw an orange bounding box where the drone disappears, and the tracker continues the position. However, if the number of consecutive frames with missing detections are greater than 5, the tracker is reset. When there are detections seen again, a new kalman filter is initialized. 
There is another problem that I encountered as I based my tracking only off position instead of considereing velocity for a moving drone. Initially, I set the measurement noise for the initial detectors bounding box matrix to [25 0; 0 25]. Becuase of that, the tracked box diverged from the detector box, meaning it wasnt tracking the detections properly. Therefore I reduced the threshold for it to follow closely, and the trajectory was also starting from drone object. 
Another problem that I initially noticed was that I did not add the bounding box for the detections in part 1, which was essential for tracking. So, I had to rerun the model and save the bounding boxes before proceeding to the tracking part. 
Finally the trajectory line is also updated using the filters center positions. For the missing detections period, the predicted positions are still adeed to keep the trajectory line continuous instead of drifting apart. 

---
