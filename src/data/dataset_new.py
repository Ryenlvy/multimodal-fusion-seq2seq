import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class FusionDatasetNew(Dataset):
    """处理新数据集结构的数据集类"""
    def __init__(self, csv_file, base_path, num_points=1024, radar_points=256, transform=None, cache_data=True, verbose=False):
        """
        初始化数据集
        
        Args:
            csv_file: 包含序列信息的CSV文件路径
            base_path: 数据根目录
            num_points: 点云采样点数
            radar_points: 雷达点数
            transform: 图像变换
            cache_data: 是否缓存数据以加速训练
            verbose: 是否打印详细信息
        """
        self.data_df = pd.read_csv(csv_file)
        self.base_path = base_path
        self.num_points = num_points
        self.radar_points = radar_points
        self.cache_data = cache_data
        self.verbose = verbose
        self.data_cache = {}
        
        # 打印数据集列名，帮助调试
        if self.verbose:
            print(f"数据集CSV列名: {self.data_df.columns.tolist()}")
        
        # 检查列名并映射到标准名称
        self.column_mapping = {}
        
        # 查找图像列
        img_columns = [col for col in self.data_df.columns if 'img_' in col or 'rgb' in col]
        if img_columns:
            self.column_mapping['img'] = img_columns
            if self.verbose:
                print(f"找到图像列: {img_columns}")
        else:
            if self.verbose:
                print("警告: 未找到图像列")
            
        # 查找点云列
        pc_columns = [col for col in self.data_df.columns if 'pc_' in col or 'lidar' in col]
        if pc_columns:
            self.column_mapping['pc'] = pc_columns
            if self.verbose:
                print(f"找到点云列: {pc_columns}")
        else:
            if self.verbose:
                print("警告: 未找到点云列")
            
        # 查找雷达列
        radar_columns = [col for col in self.data_df.columns if 'radar_' in col or 'radar' in col]
        if radar_columns:
            self.column_mapping['radar'] = radar_columns
            if self.verbose:
                print(f"找到雷达列: {radar_columns}")
        else:
            if self.verbose:
                print("警告: 未找到雷达列")
            
        # 查找波束列
        beam_columns = [col for col in self.data_df.columns if 'beam_' in col or 'beam' in col]
        if beam_columns:
            self.column_mapping['beam'] = beam_columns
            if self.verbose:
                print(f"找到波束列: {beam_columns}")
        else:
            if self.verbose:
                print("警告: 未找到波束列")
        
        # 设置图像变换 - 减小图像尺寸以加快处理速度
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),  # 进一步减小图像尺寸
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        # 检查缓存
        if self.cache_data and idx in self.data_cache:
            return self.data_cache[idx]
            
        # 获取序列信息
        seq_row = self.data_df.iloc[idx]
        
        # 打印一次行数据，帮助调试
        if idx == 0 and self.verbose:
            print(f"样本数据示例: {seq_row.to_dict()}")
        
        # 图像数据
        images = []
        points = []
        radar = []
        
        # 获取输入序列
        for i in range(8):
            # 检查是否有对应的列
            img_col = f'img_{i}'
            pc_col = f'pc_{i}'
            radar_col = f'radar_{i}'
            
            # 加载图像
            try:
                if img_col in self.data_df.columns:
                    img_path = seq_row[img_col]
                    # 如果路径是相对路径，则添加基础路径
                    if not os.path.isabs(img_path):
                        img_path = os.path.join(self.base_path, img_path.lstrip('./'))
                    img = Image.open(img_path).convert('RGB')
                    img = self.transform(img)
                else:
                    if self.verbose:
                        print(f"警告: 找不到列 {img_col}，使用零张量代替")
                    img = torch.zeros(3, 64, 64)  # 使用更小的零张量
            except Exception as e:
                if self.verbose:
                    print(f"加载图像出错: {img_path if 'img_path' in locals() else 'unknown'}, 错误: {e}")
                img = torch.zeros(3, 64, 64)  # 使用更小的零张量
            images.append(img)
            
            # 加载点云 - 使用更高效的采样方法
            try:
                if pc_col in self.data_df.columns:
                    pc_path = seq_row[pc_col]
                    # 如果路径是相对路径，则添加基础路径
                    if not os.path.isabs(pc_path):
                        pc_path = os.path.join(self.base_path, pc_path.lstrip('./'))
                    
                    # 检查文件扩展名，处理不同格式的点云文件
                    if pc_path.endswith('.bin'):
                        # 使用内存映射加载大文件
                        pc_data = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
                    elif pc_path.endswith('.ply'):
                        # 需要安装open3d库来处理ply文件
                        import open3d as o3d
                        pcd = o3d.io.read_point_cloud(pc_path)
                        pc_data = np.asarray(pcd.points)
                        # 如果点云只有xyz坐标，添加一个强度列（全为1）
                        if pc_data.shape[1] == 3:
                            intensity = np.ones((pc_data.shape[0], 1))
                            pc_data = np.hstack([pc_data, intensity])
                    else:
                        raise ValueError(f"不支持的点云文件格式: {pc_path}")
                else:
                    if self.verbose:
                        print(f"警告: 找不到列 {pc_col}，使用零点云代替")
                    pc_data = np.zeros((self.num_points, 4), dtype=np.float32)
                
                # 高效采样点云
                if len(pc_data) > self.num_points:
                    # 使用随机索引而不是choice，更高效
                    indices = np.random.permutation(len(pc_data))[:self.num_points]
                    pc_data = pc_data[indices]
                elif len(pc_data) > 0 and len(pc_data) < self.num_points:
                    # 重复点以达到所需数量
                    repeat_count = self.num_points // len(pc_data) + 1
                    pc_data = np.tile(pc_data, (repeat_count, 1))[:self.num_points]
                else:
                    pc_data = np.zeros((self.num_points, 4), dtype=np.float32)
                
                # 只使用xyz坐标
                pc_data = pc_data[:, :3]
            except Exception as e:
                if self.verbose:
                    print(f"加载点云出错: {pc_path if 'pc_path' in locals() else 'unknown'}, 错误: {e}")
                pc_data = np.zeros((self.num_points, 3), dtype=np.float32)
            points.append(torch.from_numpy(pc_data).float())
            
            # 加载雷达 - 使用更高效的采样方法
            try:
                if radar_col in self.data_df.columns:
                    radar_path = seq_row[radar_col]
                    # 如果路径是相对路径，则添加基础路径
                    if not os.path.isabs(radar_path):
                        radar_path = os.path.join(self.base_path, radar_path.lstrip('./'))
                    
                    # 使用mmap_mode加载大型npy文件以减少内存使用
                    radar_data = np.load(radar_path, mmap_mode='r')
                    
                    # 如果雷达数据太大，进行更激进的降采样
                    if len(radar_data.shape) >= 2 and radar_data.shape[0] * radar_data.shape[1] > 10000:
                        # 对每个维度进行降采样
                        sample_factor = max(1, int(np.sqrt(radar_data.shape[0] * radar_data.shape[1] / 10000)))
                        if len(radar_data.shape) == 2:
                            radar_data = radar_data[::sample_factor, ::sample_factor]
                        elif len(radar_data.shape) == 3:
                            radar_data = radar_data[::sample_factor, ::sample_factor, :]
                        elif len(radar_data.shape) == 4:
                            radar_data = radar_data[::sample_factor, ::sample_factor, ::sample_factor, :]
                else:
                    if self.verbose:
                        print(f"警告: 找不到列 {radar_col}，使用零雷达数据代替")
                    radar_data = np.zeros((self.radar_points, 5), dtype=np.float32)
                
                # 高效采样雷达点
                if len(radar_data) > self.radar_points:
                    indices = np.random.permutation(len(radar_data))[:self.radar_points]
                    radar_data = radar_data[indices]
                elif len(radar_data) > 0 and len(radar_data) < self.radar_points:
                    repeat_count = self.radar_points // len(radar_data) + 1
                    radar_data = np.tile(radar_data, (repeat_count, 1))[:self.radar_points]
                else:
                    radar_data = np.zeros((self.radar_points, 5), dtype=np.float32)
            except Exception as e:
                if self.verbose:
                    print(f"加载雷达数据出错: {radar_path if 'radar_path' in locals() else 'unknown'}, 错误: {e}")
                radar_data = np.zeros((self.radar_points, 5), dtype=np.float32)
            
            # 转换为张量，保留复数数据的结构
            radar_tensor = torch.from_numpy(radar_data.astype(np.float32))
            radar.append(radar_tensor)
        
        # 堆叠序列
        images = torch.stack(images, dim=0)  # [8, 3, 64, 64]
        points = torch.stack(points, dim=0)  # [8, num_points, 3]
        radar = torch.stack(radar, dim=0)  # [8, radar_points, 5]
        
        # 获取输出序列（波束）
        target = []
        for i in range(13):
            beam_col = f'beam_{i}'
            try:
                if beam_col in self.data_df.columns:
                    beam_idx = seq_row[beam_col]
                    if isinstance(beam_idx, str):
                        beam_idx = int(beam_idx)
                else:
                    if self.verbose:
                        print(f"警告: 找不到列 {beam_col}，使用0代替")
                    beam_idx = 0
                target.append(beam_idx)
            except Exception as e:
                if self.verbose:
                    print(f"加载波束数据出错: {beam_col}, 错误: {e}")
                target.append(0)
        
        # 转换为张量
        target = torch.tensor(target, dtype=torch.long)
        
        # 缓存数据
        if self.cache_data:
            self.data_cache[idx] = (images, points, radar, target)
        
        return images, points, radar, target 