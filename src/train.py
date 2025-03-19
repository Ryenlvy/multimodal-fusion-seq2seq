import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import yaml
import logging
from datetime import datetime
import gc

from src.data.dataset_new import FusionDatasetNew
from src.models.fusion_model import FusionSeq2Seq

def setup_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def train(config, logger=None):
    if logger is None:
        logger = setup_logger()
    
    # 提取配置参数并确保类型正确
    num_epochs = int(config['training']['num_epochs'])
    batch_size = int(config['training']['batch_size'])
    learning_rate = float(config['training']['learning_rate'])
    patience = int(config['training']['patience'])
    warmup_epochs = int(config['training']['warmup_epochs'])
    
    csv_train = config['data']['train_csv']
    csv_val = config['data']['val_csv']
    base_path = config['data']['base_path']
    
    num_beams = int(config['model']['num_beams'])
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载数据集
    logger.info("加载数据集...")
    # 进一步减少点数以节省内存
    radar_points = int(config['data'].get('radar_points', 32))  # 大幅减少默认点数
    num_points = int(config['data'].get('num_points', 256))  # 大幅减少点云点数

    # 检查并修正基础路径
    if not os.path.exists(base_path):
        logger.warning(f"配置的基础路径 {base_path} 不存在!")
        # 尝试使用正确的路径
        corrected_path = "data/scenario34_new/scenario34"
        if os.path.exists(corrected_path):
            logger.info(f"尝试使用替代路径: {corrected_path}")
            base_path = corrected_path
        else:
            logger.warning(f"替代路径 {corrected_path} 也不存在，请检查数据位置")

    logger.info(f"使用基础路径: {base_path}")
    
    # 大幅减少批量大小和工作进程数量以避免内存溢出
    original_batch_size = batch_size
    original_workers = int(config['training'].get('num_workers', 2))

    # 如果批量大小太大，减少它
    if batch_size > 4:
        batch_size = 4  # 进一步减少批量大小
        logger.warning(f"减少批量大小从 {original_batch_size} 到 {batch_size} 以避免内存溢出")

    # 减少工作进程数量到最小
    num_workers = 0  # 使用主进程加载数据，避免多进程内存开销
    if num_workers < original_workers:
        logger.warning(f"减少工作进程数量从 {original_workers} 到 {num_workers} 以避免内存溢出")

    # 禁用固定内存预取以减少内存使用
    pin_memory = False

    # 启用数据缓存以加速训练，但限制缓存大小
    train_dataset = FusionDatasetNew(
        csv_train, 
        base_path=base_path, 
        radar_points=radar_points,
        num_points=num_points,
        cache_data=False,  # 禁用缓存以减少内存使用
        verbose=False  # 关闭详细日志以提高速度
    )
    
    val_dataset = FusionDatasetNew(
        csv_val, 
        base_path=base_path, 
        radar_points=radar_points,
        num_points=num_points,
        cache_data=False,
        verbose=False
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False  # 禁用持久工作进程
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    logger.info(f"使用批量大小: {batch_size}, 工作进程数: {num_workers}")
    logger.info("初始化模型...")
    

    fused_dim = int(config['model']['fused_dim'])
    hidden_dim = int(config['model'].get('hidden_dim', 256))
    num_layers = int(config['model'].get('num_layers', 2))
    dropout = float(config['model'].get('dropout', 0.1))
    beam_embedding_dim = int(config['model']['beam_embedding_dim'])
    seq_out_len = int(config['model']['seq_out_len'])
    model = FusionSeq2Seq(
        fused_dim=fused_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        beam_embedding_dim=beam_embedding_dim,
        num_beams=num_beams,
        seq_out_len=seq_out_len
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
   
    def warmup_lr_scheduler(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return max(0.1, 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs))))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_scheduler)

    best_top1_acc = 0
    patience_counter = 0
    checkpoint_dir = "checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    logger.info("开始训练...")
    for epoch in range(num_epochs):
        gc.collect()
        torch.cuda.empty_cache()
        
        model.train()
        running_loss = 0.0

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, points, radar, targets in train_iter:
            images = images.to(device)
            points = points.to(device)
            radar = radar.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images, points, radar, targets)
                    B, seq_len = targets.size()
                    output_seq_len = outputs.size(1)
                    
                    if output_seq_len < seq_len:
                        target_for_loss = targets[:, :output_seq_len]
                    elif output_seq_len > seq_len:
                        outputs = outputs[:, :seq_len, :]
                        target_for_loss = targets
                    else:
                        target_for_loss = targets
                    
                    loss = criterion(outputs.reshape(-1, num_beams), target_for_loss.reshape(-1))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images, points, radar, targets)

                B, seq_len = targets.size()
                output_seq_len = outputs.size(1)
                
                if output_seq_len < seq_len:
                    target_for_loss = targets[:, :output_seq_len]
                elif output_seq_len > seq_len:
                    outputs = outputs[:, :seq_len, :]
                    target_for_loss = targets
                else:
                    target_for_loss = targets
                
                loss = criterion(outputs.reshape(-1, num_beams), target_for_loss.reshape(-1))
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()

            del images, points, radar, outputs
            if 'target_for_loss' in locals():
                del target_for_loss

            if train_iter.n % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 更新进度条
            train_iter.set_postfix(loss=loss.item())
        
        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        model.eval()
        total = 0
        top1_correct = 0
        top2_correct = 0
        top3_correct = 0
        top4_correct = 0
        top5_correct = 0
        
        with torch.no_grad():
            for images, points, radar, targets in tqdm(val_loader, desc="验证中"):
                images = images.to(device)
                points = points.to(device)
                radar = radar.to(device)
                targets = targets.to(device)
                
                outputs = model(images, points, radar)

                last_outputs = outputs[:, -5:, :]  # [B, 5, num_beams]
                last_targets = targets[:, -5:]  # [B, 5]
                
                B_val = last_targets.size(0)
                
                for t in range(5):
                    logits = last_outputs[:, t, :]  # [B, num_beams]
                    _, topk = logits.topk(5, dim=1)  # [B, 5]
                    preds = logits.argmax(dim=1)  # [B]
                    correct = (preds == last_targets[:, t]).sum().item()
                    total += B_val
                    top1_correct += correct
                    
                    for i in range(B_val):
                        gt = last_targets[i, t].item()
                        top2_correct += int(gt in topk[i, :2].cpu().numpy())
                        top3_correct += int(gt in topk[i, :3].cpu().numpy())
                        top4_correct += int(gt in topk[i, :4].cpu().numpy())
                        top5_correct += int(gt in topk[i, :5].cpu().numpy())

        top1_acc = top1_correct / total
        top2_acc = top2_correct / total
        top3_acc = top3_correct / total
        top4_acc = top4_correct / total
        top5_acc = top5_correct / total
        
        logger.info(f"验证 Top1: {top1_acc:.4f}, Top2: {top2_acc:.4f}, Top3: {top3_acc:.4f}, Top4: {top4_acc:.4f}, Top5: {top5_acc:.4f}")
        
        # 早停检查
        if top1_acc > best_top1_acc:
            best_top1_acc = top1_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"保存最佳模型，Top1准确率: {best_top1_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"早停触发！{patience}轮未见改善")
                break
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"当前学习率: {current_lr:.6f}")
        
        # 在每个epoch结束后清理内存
        gc.collect()
        torch.cuda.empty_cache()
    
    logger.info(f"训练结束！最佳Top1准确率: {best_top1_acc:.4f}")
    
    # 加载最佳模型并保存最终模型
    model.load_state_dict(torch.load(best_model_path))
    final_model_path = os.path.join(checkpoint_dir, "fusion_seq2seq_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"最终模型保存到: {final_model_path}")
    
    return model, best_top1_acc 
