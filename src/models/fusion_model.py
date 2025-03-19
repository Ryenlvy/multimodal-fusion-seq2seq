import torch
import torch.nn as nn
import torch.nn.functional as F
from .feature_extractors import ImageFeatureExtractor, FrustumPointNetFeatureExtractor, RadarFeatureExtractor

class MultiModalFusion(nn.Module):
    """融合图像、点云和雷达特征的模块 - 轻量级版本"""
    def __init__(self, image_feature_dim=64, point_feature_dim=64, radar_feature_dim=64, fused_dim=128):
        super(MultiModalFusion, self).__init__()
        self.image_extractor = ImageFeatureExtractor(output_dim=image_feature_dim)
        self.point_extractor = FrustumPointNetFeatureExtractor(output_dim=point_feature_dim)
        self.radar_extractor = RadarFeatureExtractor(output_dim=radar_feature_dim)
        
        # 使用MLP进行特征融合
        total_dim = image_feature_dim + point_feature_dim + radar_feature_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.LayerNorm(total_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(total_dim // 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, image, points, radar):
        # 提取各模态特征
        img_feat = self.image_extractor(image)
        pt_feat = self.point_extractor(points)
        radar_feat = self.radar_extractor(radar)
        
        # 简单拼接并通过MLP融合
        combined = torch.cat([img_feat, pt_feat, radar_feat], dim=-1)
        fused = self.fusion_mlp(combined)
        
        return fused

class FusionSeq2Seq(nn.Module):
    """基于轻量级GRU的多模态融合模型（无注意力机制）"""
    def __init__(self,
                 fused_dim=256,
                 hidden_dim=128,
                 num_layers=1,  # 减少到单层GRU
                 dropout=0.1,
                 beam_embedding_dim=8,
                 num_beams=65,
                 seq_out_len=13):
        super(FusionSeq2Seq, self).__init__()
        
        self.modal_fusion = MultiModalFusion(fused_dim=fused_dim)
        
        # 特征降维
        self.feature_reduction = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 轻量级编码器GRU
        self.encoder_rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # 解码器输入嵌入
        self.beam_embedding = nn.Embedding(num_beams, beam_embedding_dim)
        
        # 解码器GRU
        self.decoder_rnn = nn.GRU(
            input_size=beam_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, num_beams)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_out_len = seq_out_len
        self.num_beams = num_beams
    
    def forward(self, images_seq, points_seq, radar_seq, target_seq=None):
        B, seq_in, _, H, W = images_seq.size()
        device = images_seq.device
        
        # 提取并融合每个时间步的多模态特征
        fused_features = []
        for t in range(seq_in):
            img = images_seq[:, t]
            pts = points_seq[:, t]
            
            # 处理雷达数据
            if radar_seq.dim() == 5:  # [B, seq_in, N, C, 2]
                rdr = radar_seq[:, t]
            else:
                rdr = radar_seq
            
            fused = self.modal_fusion(img, pts, rdr)
            fused = self.feature_reduction(fused)
            fused_features.append(fused)
        
        # 将融合特征堆叠为序列
        encoder_inputs = torch.stack(fused_features, dim=1)
        
        # 编码器前向传播
        _, hidden = self.encoder_rnn(encoder_inputs)
        
        # 准备解码器输入
        if target_seq is not None and self.training:
            # 训练模式：使用教师强制
            seq_out_len = min(self.seq_out_len, target_seq.size(1))
            target_input = target_seq[:, :seq_out_len-1]
            
            # 嵌入目标序列
            decoder_input = self.beam_embedding(target_input)  # [B, seq_len-1, beam_embedding_dim]
            
            # 解码器前向传播
            decoder_output, _ = self.decoder_rnn(decoder_input, hidden)
            
            # 预测波束
            outputs = self.output_layer(decoder_output)
            
            return outputs
        else:
            # 推理模式：自回归生成
            outputs = []
            # 初始化第一个token
            current_token = torch.zeros(B, dtype=torch.long, device=device)
            
            # 限制生成的序列长度
            seq_out_len = min(self.seq_out_len, 13)
            
            # 当前隐藏状态
            current_hidden = hidden
            
            for i in range(seq_out_len):
                # 嵌入当前token
                current_input = self.beam_embedding(current_token).unsqueeze(1)  # [B, 1, beam_embedding_dim]
                
                # 解码器RNN前向传播
                decoder_output, current_hidden = self.decoder_rnn(current_input, current_hidden)
                
                # 预测波束
                prediction = self.output_layer(decoder_output.squeeze(1))
                
                # 添加到输出列表
                outputs.append(prediction.unsqueeze(1))
                
                # 获取下一个token
                current_token = prediction.argmax(dim=-1)
            
            # 拼接所有时间步的输出
            outputs = torch.cat(outputs, dim=1)
            return outputs
    
    def predict(self, images_seq, points_seq, radar_seq):
        """用于推理阶段，返回预测的波束序列"""
        with torch.no_grad():
            outputs = self.forward(images_seq, points_seq, radar_seq)
            predictions = outputs.argmax(dim=-1)
            return predictions