from ultralytics import RTDETR
import os
#open cv for processing mp4 videos
import cv2

#load the fine-tuned model
model = RTDETR("runs/detect/train10/weights/best.pt")
#current working directory has the videos
video_folder = "."
frames_main = "frames"
detections_main = "detections"
#create a output folder named "detections"
os.makedirs(detections_main, exist_ok=True)
os.makedirs(frames_main, exist_ok=True)

#for all files in the current working directory
for file in os.listdir(video_folder):
    #process all .mp4 files
    if file.endswith(".mp4"):                          
        video_path = os.path.join(video_folder, file)  
        video_name = file.split(".")[0]               
        #open
        cap = cv2.VideoCapture(video_path)
        
        #get original fps and compute interval for 5 fps
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps == 0:
            original_fps = 30 
        frame_interval = int(original_fps / 5)
        
        #make a folder for this video's frames according to the name for organization
        frame_folder = os.path.join(frames_main, video_name) 
        #create one 
        os.makedirs(frame_folder, exist_ok=True)
        #counter
        frame_id = 0 
        saved_id = 0 
        #keep reading frames until the the end
        while True:                                
            ret, frame = cap.read()                    
            if not ret:                                
                break
            if frame_id % frame_interval == 0:
                #save each frame as a jpg image 
                frame_path = os.path.join(frame_folder, f"frame_{saved_id}.jpg")
                cv2.imwrite(frame_path, frame)        
                saved_id += 1        
            #move on
            frame_id += 1                 
        cap.release()         

#for all files in the current working directory
for folder in os.listdir(frames_main):
    folder_path = os.path.join(frames_main, folder)     
    #for each video frames folder inside the main frames folder
    if os.path.isdir(folder_path): 
        #output detection corresponding to this video frames folder                            
        save_folder = os.path.join(detections_main, folder)   
        os.makedirs(save_folder, exist_ok=True)   

        #for each frames
        for file in os.listdir(folder_path):    
            #and files that end with .jpg --> frames
            if file.endswith(".jpg"):                          
                img_path = os.path.join(folder_path, file)   
                #read each frame  
                frame = cv2.imread(img_path)  
                #and run the detector on each one                 
                results = model(frame, conf=0.5, verbose=False)
                #if there is a detection
                if len(results[0].boxes) > 0:
                    #then, draw a bounding box 
                    annotated = results[0].plot()
                    #create file path and save the image
                    save_path = os.path.join(save_folder, file)
                    #same bounding box frame as an image 
                    cv2.imwrite(save_path, annotated)
        print(f"Finished detection for video: {folder}")
print("Done")