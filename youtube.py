import yt_dlp as youtube_dl
import os
import pandas as pd
from tqdm import tqdm
import h5py
import numpy as np
import torch

def download_youtube_video(yt_id, video_id):
    try:
        youtube_url = f"https://www.youtube.com/watch?v={yt_id}"
        print(youtube_url)
        
        save_path = f"dataset/youtube/{video_id}"
        os.makedirs(save_path, exist_ok=True)  # 创建保存路径

        ydl_opts = {
            'outtmpl': os.path.join(save_path, f'{video_id}.mp4'),
            'format': 'bestvideo[height<=480]+bestaudio/best[height<=480]',  # 选择480p视频
            'cookiefile': 'cookies.txt',
            'merge_output_format': 'mp4',
        }

        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
        except Exception as download_e:
            print(f"Error downloading video {video_id}: {download_e}")
            return 0

        video_file_path = os.path.join(save_path, f"{video_id}.mp4")
        
        print(f"successfully download video to {video_file_path}")
        
        title = info.get('title', 'Unknown Title')
        tags = info.get('tags', [])
        categories = info.get('categories', [])

        info_path = os.path.join(save_path, f"{video_id}_info.txt")
        
        try:
            with open(info_path, "w") as f:
                f.write(f"Video Title: {title}\n")
                f.write(f"Categories: {categories}\n")
                if tags:
                    f.write(f"Tags: {tags}\n")
        except Exception as open_e:
            print(f"Error writing info to {info_path}: {open_e}")
            return 0

        print(f"successfully save video info to {info_path}")
        try:
            file_size = os.path.getsize(video_file_path) / (1024 * 1024)  # 获取文件大小
        except Exception as size_e:
            print(f"Error getting size of {video_file_path}: {size_e}")
            return 0
        return file_size

    except Exception as e:
        print(f"General error downloading video {video_id}: {e}")
        return 0

size_sum = 0
size_max = 3

meta_data = "dataset/metadata.csv"
dataset = 'dataset/mr_hisum.h5'
video_data = h5py.File(dataset, 'r')
df = pd.read_csv(meta_data)
df_ls=list(df.itertuples())
df_ls=df_ls[111:]
for row in tqdm(df_ls):
    yt_id = row.youtube_id
    video_id = row.video_id
    size = download_youtube_video(yt_id, video_id)
    if size != 0:
        gtscore = np.array(video_data[video_id + '/gtscore'])
        n_frames = gtscore.shape[0]
        np.save(os.path.join("dataset/youtube", video_id, "gtscore.npy"), gtscore)
        np.save(os.path.join("dataset/youtube", video_id, "n_frames.npy"), n_frames)
        print(f"successfully save gtscore and n_frames to {video_id}")
    size_sum = size_sum + size
    print(f"-{size_sum}-")
    if size_sum >= size_max * 1024:
        print(f'Total downloaded size exceeds {size_max}GB limit. Stopping download.')
        break

