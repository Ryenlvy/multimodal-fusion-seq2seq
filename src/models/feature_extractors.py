import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ImageFeatureExtractor(nn.Module):
    """图像特征提取器，基于ResNet50"""
    def __init__(self, output_dim=128):
        super(ImageFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, output_dim)
        
    def forward(self, x):
        x = self.backbone(x)  # (B, 2048, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class FrustumPointNetFeatureExtractor(nn.Module):
    """点云特征提取器，基于PointNet架构"""
    def __init__(self, output_dim=128):
        super(FrustumPointNetFeatureExtractor, self).__init__()
        # 前景分割分支
        self.seg_mlp = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.seg_fc = nn.Conv1d(64, 2, 1)  # 输出两个类别：背景和前景

        # 特征提取分支
        self.feat_mlp = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, output_dim, 1)
        )
        
    def forward(self, x):
        B, N, _ = x.size()
        x_trans = x.transpose(2, 1)
        
        # 前景分割分支
        seg_feat = self.seg_mlp(x_trans)
        seg_logits = self.seg_fc(seg_feat)
        seg_scores = F.softmax(seg_logits, dim=1)
        obj_mask = seg_scores[:, 1:2, :]
        features = self.feat_mlp(x_trans)
        features = features * obj_mask
        features, _ = torch.max(features, dim=2)
        return features 

class RadarFeatureExtractor(nn.Module):
    """雷达数据特征提取器 - 使用2D FFT和ResNet50"""
    def __init__(self, input_channels=5, output_dim=128):
        super(RadarFeatureExtractor, self).__init__()
        # 使用ResNet50作为特征提取器
        resnet = models.resnet50(pretrained=True)
        
        # 修改第一层以接受不同的输入通道数
        if input_channels != 3:
            resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 去掉最后的全连接层（保留全局池化层）
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, output_dim)  # ResNet50的特征维度是2048
        
    def forward(self, x):
        # 处理形状为 [B, 4, 256, 250] 的雷达数据
        if x.dim() == 4 and x.size(1) == 4:
            B = x.size(0)
            x_complex = torch.fft.fft2(x.float())
            x_magnitude = torch.abs(x_complex)
            x_log = torch.log(x_magnitude + 1e-10)
            x_norm = (x_log - x_log.min()) / (x_log.max() - x_log.min() + 1e-10)
            
            # 确保通道数匹配ResNet的输入通道数
            if x_norm.size(1) != self.backbone[0].in_channels:
                if x_norm.size(1) < self.backbone[0].in_channels:
                    # 添加额外通道
                    padding = torch.zeros(B, self.backbone[0].in_channels - x_norm.size(1), 
                                         x_norm.size(2), x_norm.size(3), device=x.device)
                    x_norm = torch.cat([x_norm, padding], dim=1)
                else:
                    x_norm = x_norm[:, :self.backbone[0].in_channels, :, :]
            
            # 应用ResNet提取特征
            x = self.backbone(x_norm)  # [B, 2048, 1, 1]
            x = x.view(x.size(0), -1)  # [B, 2048]
            x = self.fc(x)  # [B, output_dim]
            return x
        
        # 处理形状为 [B, C, H, W, 2] 的复数雷达数据
        elif x.dim() == 5 and x.size(-1) == 2:
            B = x.size(0)
            # 将复数数据转换为幅度
            x_real = x[..., 0]
            x_imag = x[..., 1]
            x_magnitude = torch.sqrt(x_real**2 + x_imag**2)
            
            # 应用2D FFT变换
            x_complex = torch.fft.fft2(x_magnitude)
            
            # 计算幅度谱
            x_fft_magnitude = torch.abs(x_complex)
            
            # 对数变换
            x_log = torch.log(x_fft_magnitude + 1e-10)
            
            # 归一化
            x_norm = (x_log - x_log.min()) / (x_log.max() - x_log.min() + 1e-10)
            
            # 确保通道数匹配
            if x_norm.size(1) != self.backbone[0].in_channels:
                if x_norm.size(1) < self.backbone[0].in_channels:
                    padding = torch.zeros(B, self.backbone[0].in_channels - x_norm.size(1), 
                                         x_norm.size(2), x_norm.size(3), device=x.device)
                    x_norm = torch.cat([x_norm, padding], dim=1)
                else:
                    x_norm = x_norm[:, :self.backbone[0].in_channels, :, :]
            
            # 应用ResNet提取特征
            x = self.backbone(x_norm)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        
        # 处理其他情况
        else:
            # 如果是其他形状，尝试重塑为2D图像格式
            B = x.size(0)
            
            if x.dim() == 3:  # [B, N, C]
                # 转置为 [B, C, N]
                x = x.transpose(2, 1)
                
                # 尝试将其重塑为2D图像
                C = x.size(1)
                N = x.size(2)
                
                # 计算最接近的平方根以获得合理的高度和宽度
                H = int(N**0.5)
                W = N // H
                
                if H * W < N:
                    W += 1
                
                # 填充到H*W
                if H * W > N:
                    padding = torch.zeros(B, C, H*W - N, device=x.device)
                    x = torch.cat([x, padding], dim=2)
                
                # 重塑为 [B, C, H, W]
                x = x.reshape(B, C, H, W)
            
            # 确保通道数匹配
            if x.size(1) != self.backbone[0].in_channels:
                if x.size(1) < self.backbone[0].in_channels:
                    padding = torch.zeros(B, self.backbone[0].in_channels - x.size(1), 
                                         x.size(2), x.size(3), device=x.device)
                    x = torch.cat([x, padding], dim=1)
                else:
                    x = x[:, :self.backbone[0].in_channels, :, :]
            
            # 应用ResNet提取特征
            x = self.backbone(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x 