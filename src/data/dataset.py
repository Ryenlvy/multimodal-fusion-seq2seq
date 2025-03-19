import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import open3d as o3d

def load_point_cloud(file_path, num_points=1024):
    """加载点云数据并进行采样"""
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    if len(points) >= num_points:
        choice = np.random.choice(len(points), num_points, replace=False)
        points = points[choice]
    else:
        choice = np.random.choice(len(points), num_points, replace=True)
        points = points[choice]
    
    return points.astype(np.float32)

def load_radar_data(file_path, num_points=512):
    """加载雷达数据并进行采样"""
    try:
        radar_data = np.load(file_path)
        
        # 处理复数数据 - 取绝对值
        if np.iscomplexobj(radar_data):
            radar_data = np.abs(radar_data)
        
        # 确保数据是二维的
        if radar_data.ndim == 1:
            # 如果是一维数据，转换为二维
            radar_data = radar_data.reshape(-1, 1)
            # 添加一个全零列，使其成为2列
            zeros = np.zeros((radar_data.shape[0], 1), dtype=np.float32)
            radar_data = np.hstack((radar_data, zeros))
        elif radar_data.ndim > 2:
            # 如果维度过高，只取前两个维度
            radar_data = radar_data.reshape(-1, 2)
        
        # 确保数据至少有2列
        if radar_data.shape[1] < 2:
            zeros = np.zeros((radar_data.shape[0], 2 - radar_data.shape[1]), dtype=np.float32)
            radar_data = np.hstack((radar_data, zeros))
        elif radar_data.shape[1] > 2:
            # 如果列数过多，只取前两列
            radar_data = radar_data[:, :2]
        
        # 确保雷达数据有足够的点
        if len(radar_data) >= num_points:
            choice = np.random.choice(len(radar_data), num_points, replace=False)
            radar_data = radar_data[choice]
        else:
            # 如果点数不足，则通过重复采样来补足
            choice = np.random.choice(len(radar_data), num_points, replace=True)
            radar_data = radar_data[choice]
        
        return radar_data.astype(np.float32)
    except Exception as e:
        print(f"加载雷达数据出错: {file_path}, 错误: {e}")
        # 返回一个全零的雷达数据作为替代
        return np.zeros((num_points, 2), dtype=np.float32)

class FusionDataset(Dataset):
    """融合数据集类，用于加载图像、点云和雷达序列数据"""
    def __init__(self, csv_file, base_path="./scenario32", num_points=1024, radar_points=256, transform=None, include_radar=False):
        self.data = pd.read_csv(csv_file)
        self.num_points = num_points
        # 减少雷达点数以节省内存
        self.radar_points = radar_points
        self.base_path = os.path.abspath(base_path)  # 转换为绝对路径
        self.include_radar = include_radar
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 验证数据集路径
        self._validate_paths()
        
        # 缓存雷达数据路径
        self.radar_base_path = "/home/ryne/workplace/fpointnet/combine/scenario32/unit1/radar_data"
        if not os.path.exists(self.radar_base_path):
            print(f"警告: 雷达数据基础路径不存在: {self.radar_base_path}")
            # 尝试其他可能的路径
            alt_paths = [
                os.path.join(self.base_path, "unit1", "radar_data"),
                os.path.join(self.base_path, "radar_data"),
                "/home/ryne/workplace/fpointnet/combine/radar_data"
            ]
            for path in alt_paths:
                if os.path.exists(path):
                    self.radar_base_path = path
                    print(f"使用替代雷达数据路径: {self.radar_base_path}")
                    break
    
    def _validate_paths(self):
        """验证数据集中的文件路径是否存在"""
        # 检查前5个样本的路径
        sample_size = min(5, len(self.data))
        for i in range(sample_size):
            row = self.data.iloc[i]
            for t in range(1, 9):
                img_path = os.path.join(self.base_path, row[f'x_frame_{t}'])
                lidar_path = os.path.join(self.base_path, row[f'x_lidar_{t}'])
                
                # 尝试修复路径
                if not os.path.exists(img_path):
                    # 尝试移除开头的点号
                    alt_path = os.path.join(self.base_path, row[f'x_frame_{t}'].lstrip('./'))
                    if os.path.exists(alt_path):
                        print(f"警告: 路径需要修复，原路径: {img_path}, 修复后: {alt_path}")
                        # 更新CSV中的所有路径
                        self.data[f'x_frame_{t}'] = self.data[f'x_frame_{t}'].str.lstrip('./')
                        self.data[f'x_lidar_{t}'] = self.data[f'x_lidar_{t}'].str.lstrip('./')
                        for j in range(1, 14):
                            if f'y_frame_{j}' in self.data.columns:
                                self.data[f'y_frame_{j}'] = self.data[f'y_frame_{j}'].str.lstrip('./')
                        break
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        images = []
        point_clouds = []
        radar_data = []
        
        # 加载8帧的图像和点云
        for t in range(1, 9):
            # 获取路径并移除可能的前导 './'
            frame_path = row[f'x_frame_{t}'].lstrip('./')
            lidar_path = row[f'x_lidar_{t}'].lstrip('./')
            
            img_path = os.path.join(self.base_path, frame_path)
            lidar_path = os.path.join(self.base_path, lidar_path)
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"图像文件不存在: {img_path}")
            if not os.path.exists(lidar_path):
                raise FileNotFoundError(f"点云文件不存在: {lidar_path}")
            
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            images.append(image.unsqueeze(0))
            
            pts = load_point_cloud(lidar_path, num_points=self.num_points)
            pts = torch.tensor(pts)
            point_clouds.append(pts.unsqueeze(0))
            
            # 如果包含雷达数据，则加载雷达数据
            if self.include_radar:
                radar_file = f"radar_data_{t}.npy"
                radar_path = os.path.join(self.radar_base_path, radar_file)
                
                if os.path.exists(radar_path):
                    rdr = load_radar_data(radar_path, num_points=self.radar_points)
                else:
                    print(f"警告: 雷达数据不存在: {radar_path}，使用零填充")
                    rdr = np.zeros((self.radar_points, 2), dtype=np.float32)
                
                rdr = torch.tensor(rdr)
                radar_data.append(rdr.unsqueeze(0))
            
        # 拼接所有帧的数据
        images = torch.cat(images, dim=0)
        point_clouds = torch.cat(point_clouds, dim=0)
        
        # 读取 13 个时间步的目标波束索引
        target = []
        for t in range(1, 14):
            target.append(int(row[f'y_beam_{t}']))
        target = torch.tensor(target, dtype=torch.long)
        
        if self.include_radar:
            radar_data = torch.cat(radar_data, dim=0)
            return images, point_clouds, radar_data, target
        else:
            return images, point_clouds, target 