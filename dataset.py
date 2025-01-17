import pandas as pd
import os

def aggregate_ot_and_generate_csv(csv_file, base_path, output_file):
    # 1. 读取CSV文件
    df = pd.read_csv(csv_file)
    # 获取header
    header = df.columns.tolist()  # 将header转换为列表
    df = df.to_numpy()
    
    # 2. 根据 video_id 聚集 OT
    video_ls = os.listdir(base_path)
    video_ls = sorted(video_ls)
    # video_ls=["video_1","video_2"]

    # 3. 准备新数据列表
    new_rows = []

    # 4. 遍历各个 video_id，找到 MP4 文件并构造新行
    for video_id in video_ls:
        print(f"Processing video_id: {video_id}")

        # 获取与当前 video_id 对应的所有行
        corresponding_dates = df[df[:,-2] == video_id]
        # 读取 metainfo 信息
        metainfo_path = os.path.join(base_path, f"{video_id}/{video_id}_info.txt")
        
        with open(metainfo_path, 'r', encoding='utf-8') as f:
            metainfo = f.readlines()
        
        metainfo = ' '.join(metainfo).replace('\n', '')
        print(metainfo)
        
        # 用 metainfo 更新对应日期的最后一列
        corresponding_dates[:, -1] = metainfo
        
        # 为每个 video_id 的 OT 值添加索引号
        # 这里我们将索引添加到数据中
        for index, row in enumerate(corresponding_dates):
            new_row = list(row)  # 将行转换为列表
            new_row.insert(0, index)
            new_rows.append(new_row)  # 添加新行到新的数据列表

    # 5. 创建新的 DataFrame，并保存为新的 CSV 文件
    # 由于我们在添加新行时已经包括了最后的索引列，所以我们需要更新 header
    new_header = ['index']+header  # 在原有header中添加索引列
    new_df = pd.DataFrame(new_rows)
    new_df.columns = new_header  # 设置新的header
    new_df.to_csv(output_file, index=False, header=True)

# 示例调用
aggregate_ot_and_generate_csv('dataset/gtscore.csv', 'dataset/youtube', 'dataset/gtscore_video.csv')