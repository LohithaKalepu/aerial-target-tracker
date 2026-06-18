from datasets import Dataset
from PIL import Image
import os

detections_folder = "detections"
#go through each video folder
for video in os.listdir(detections_folder):
    video_path = os.path.join(detections_folder, video)

    if os.path.isdir(video_path):
        data = []
        #go through each image in that video folder
        for file in os.listdir(video_path):
            if file.endswith(".jpg"):
                img_path = os.path.join(video_path, file)
                img = Image.open(img_path)
                data.append({
                    "image": img,
                    "video": video
                })
        #create dataset for this video
        dataset = Dataset.from_list(data)
        #save as parquet
        save_path = os.path.join(detections_folder, f"{video}.parquet")
        dataset.to_parquet(save_path)
        print(f"Saved {save_path}")

print("Done")