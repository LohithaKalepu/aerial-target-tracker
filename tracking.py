import os
import cv2
import math
import numpy as np
from ultralytics import RTDETR
from filterpy.kalman import KalmanFilter

#congif for generalization
model_path = "runs/detect/train10/weights/best.pt"
input_folder = "."
output_folder = "tracking_outputs"
#same threshold used for detections
confidence_threshold = 0.50
#5fps
target_fps = 5.0
#before resetting tracker, allow at least 5 missing detections
max_missed = 5
os.makedirs(output_folder, exist_ok=True)

# load model
model = RTDETR(model_path)

#get all .mp4 files
video_files = []
for file in os.listdir(input_folder):
    if file.lower().endswith(".mp4"):
        video_files.append(file)
video_files.sort()
#if no .mp4 extension
if len(video_files) == 0:
    print("No mp4 files found.")

#process each video
for video_file in video_files:
    print("\nProcessing:", video_file)
    video_path = os.path.join(input_folder, video_file)
    video_name = video_file.rsplit(".", 1)[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video:", video_file)
        continue
    #get the original fps
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    #set default to 30
    if original_fps <= 0 or math.isnan(original_fps):
        original_fps = 30.0
    #get frame interval for 5 fps
    frame_interval = int(round(original_fps / target_fps))
    if frame_interval < 1:
        frame_interval = 1
    #output video fps
    output_fps = original_fps / frame_interval

    #initialization of kalman
    tracker_started = False
    #the object
    kf_obj = None
    #how many det. missed
    det_missed = 0
    #previously known bounding box width and height
    #so that we can build a rectangle around the estimated center
    w_prev = 0
    h_prev = 0
    #center point of trajectory
    trajectory_points = []
    #all frames
    frame_index = 0
    #how many frames were actually writen to the output
    written_frames = 0
    video_writer = None
    #for each video processed, create an output 
    output_video_path = os.path.join(output_folder, f"{video_name}_tracked.mp4")

    #start reading video frame by frame
    #loop until the video ends
    while True:
        ret, frame = cap.read()
        #if no frame is returns, we have reached the end of video --> break
        if not ret:
            break
        #only process frames at 5 fps
        #skip frames that dont match interval
        if frame_index % frame_interval != 0:
            frame_index += 1
            continue
        #current width and height
        frame_height, frame_width = frame.shape[:2]
        #initialize video output writer
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            #create writer object
            video_writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                output_fps,
                (frame_width, frame_height)
            )

        #for last detections, I forgot to the the bounding box. 
        #keeping the assignment deadline in mind, I did not modify any detections code 
        #so, I rerun the model directly on the frames sampled here, and added bounding boxes to help tracking
        
        #rerun
        results = model(frame, verbose=False)
        #keep highest conf box 
        best_box = None
        best_conf = -1.0
        #if mulitple detections, keep the highest conf one
        #make sure detection output exists
        if results is not None and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    conf = float(b.conf[0].item())
                    #add bounding boxes
                    #for only detections above threshold
                    if conf >= confidence_threshold:
                        #in xyxy format
                        xyxy = b.xyxy[0].cpu().numpy()
                        #to integers
                        x1 = int(xyxy[0])
                        y1 = int(xyxy[1])
                        x2 = int(xyxy[2])
                        y2 = int(xyxy[3])
                        #keep the highest conf detections
                        if conf > best_conf:
                            best_conf = conf
                            best_box = (x1, y1, x2, y2)

        #kalman filter
        #if tracker exists, predict next state before update
        if tracker_started:
            kf_obj.predict()
        #var to see if the frame should be written to output
        keep_frame = False

        #if the detection exists
        if best_box is not None:
            #see best detection
            x1, y1, x2, y2 = best_box
            #compute detection center of bb
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            #compute detection width and height of bb
            bw = x2 - x1
            bh = y2 - y1
            
            #initialize kalman filter after first detection
            if tracker_started is False:
                kf_obj = KalmanFilter(dim_x=2, dim_z=2)
                #state = [cx, cy] transition matrix
                kf_obj.F = np.array([
                    [1, 0],
                    [0, 1],
                ], dtype=float)
                # measurement = [cx, cy] --> maps state directly to observation
                kf_obj.H = np.array([
                    [1, 0],
                    [0, 1]
                ], dtype=float)
                #initialize state from first detected center
                kf_obj.x = np.array([
                    [cx],
                    [cy],
                ], dtype=float)
                #initial high uncertainty --> large initial cov
                kf_obj.P *= 500.0
                #measure noise --> controls how noisy detector mesauremnts are assumed to be
                kf_obj.R = np.array([
                    [1.0, 0.0],
                    [0.0, 1.0]
                ], dtype=float)
                #process noise --> how much state can vary b/w steps
                kf_obj.Q = np.array([
                    [10.0, 0.0],
                    [0.0, 10.0]
                ], dtype=float)

                tracker_started = True
                
            #create measurement vector
            measurement = np.array([
                [cx],
                [cy]
            ], dtype=float)

            #update estimate using current detecting
            kf_obj.update(measurement)
            #remember the most recent width and height for rebuilding
            w_prev = bw
            h_prev = bh
            #reset counter
            det_missed = 0
            #get filtered center from kalman state
            filtered_cx = float(kf_obj.x[0, 0])
            filtered_cy = float(kf_obj.x[1, 0])
            #append to trajectory
            trajectory_points.append((int(round(filtered_cx)), int(round(filtered_cy))))
            #rebuild box using the most recent width and height
            tx1 = int(round(filtered_cx - w_prev / 2))
            ty1 = int(round(filtered_cy - h_prev / 2))
            tx2 = int(round(filtered_cx + w_prev / 2))
            ty2 = int(round(filtered_cy + h_prev / 2))
            #clip rectangle cordinates so they are inside image boundaries
            if tx1 < 0:
                tx1 = 0
            if ty1 < 0:
                ty1 = 0
            if tx2 >= frame_width:
                tx2 = frame_width - 1
            if ty2 >= frame_height:
                ty2 = frame_height - 1

            #draw detector box 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #detector label with confidence
            cv2.putText(
                frame,
                f"drone {best_conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )

            #tracker box centered at kalman-filtered pos
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)
            #tracker label
            cv2.putText(
                frame,
                "track",
                (tx1, max(20, ty1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            #draw full path only after at least 2 points 
            if len(trajectory_points) >= 2:
                #convert to polyline
                pts = np.array(trajectory_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255, 255), 2)
            #keep this frame
            keep_frame = True

        #if there is no detection
        else:
            #if tracker has started, predict
            if tracker_started:
                det_missed += 1
                #predicted center from current kalman state
                predicted_cx = float(kf_obj.x[0, 0])
                predicted_cy = float(kf_obj.x[1, 0])
                
                if det_missed <= max_missed:
                    #append predicted center to trajectory
                    trajectory_points.append((int(round(predicted_cx)), int(round(predicted_cy))))
                    #rebuild predicted box
                    tx1 = int(round(predicted_cx - w_prev / 2))
                    ty1 = int(round(predicted_cy - h_prev / 2))
                    tx2 = int(round(predicted_cx + w_prev / 2))
                    ty2 = int(round(predicted_cy + h_prev / 2))
                    #clip
                    if tx1 < 0:
                        tx1 = 0
                    if ty1 < 0:
                        ty1 = 0
                    if tx2 >= frame_width:
                        tx2 = frame_width - 1
                    if ty2 >= frame_height:
                        ty2 = frame_height - 1

                    #draw predicted box 
                    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 165, 255), 2)
                    cv2.putText(
                        frame,
                        "predicted",
                        (tx1, max(20, ty1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA
                    )

                    #keep drawing trajectory during the gaps as well
                    if len(trajectory_points) >= 2:
                        pts = np.array(trajectory_points, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], False, (0, 255, 255), 2)

                    keep_frame = True

                #if missed detections exceed the limit, reset tracker
                else:
                    tracker_started = False
                    kf_obj = None
                    det_missed = 0
                    w_prev = 0
                    h_prev = 0

        #output frames 
        if keep_frame:
            video_writer.write(frame)
            written_frames += 1
        frame_index += 1
    cap.release()
    if video_writer is not None:
        video_writer.release()
    #summary
    if written_frames == 0:
        print("No output frames written for", video_file)
    else:
        print("Saved output video:", output_video_path)
        print("Done with", video_file)