import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import yaml
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib as mpl

from src.data.dataset_new import FusionDatasetNew
from src.models.fusion_model import FusionSeq2Seq

# 设置matplotlib支持中文显示
def setup_chinese_font():
    """设置matplotlib支持中文显示"""
    try:
        # 尝试使用系统中的中文字体
        fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'AR PL UMing CN', 'STSong', 'NSimSun']
        for font in fonts:
            try:
                mpl.rcParams['font.family'] = font
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                # 测试字体是否支持中文
                fig = plt.figure(figsize=(1, 1))
                plt.text(0.5, 0.5, '测试', fontsize=12)
                plt.close(fig)
                print(f"使用中文字体: {font}")
                return True
            except:
                continue
        
        # 如果没有找到合适的中文字体，使用英文标签
        print("警告: 未找到支持中文的字体，将使用英文标签")
        return False
    except Exception as e:
        print(f"设置中文字体时出错: {e}")
        return False

def test_model(model, test_loader, device, num_beams=65, output_dir="results"):
    """
    测试模型并生成评估报告
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        num_beams: 波束数量
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    use_chinese = setup_chinese_font()
    
    # 将模型设置为评估模式
    model.eval()
    
    # 初始化评估指标
    all_preds = []
    all_targets = []
    top1_correct = 0
    top2_correct = 0
    top3_correct = 0
    top5_correct = 0
    total = 0
    
    # 测试模型
    with torch.no_grad():
        for images, points, targets in tqdm(test_loader, desc="测试中"):
            images = images.to(device)
            points = points.to(device)
            targets = targets.to(device)
            
            # 获取预测结果
            outputs = model(images, points, target_seq=None)
            last_outputs = outputs[-5:]  # 取最后5个时间步
            last_targets = targets[:, -5:]
            B_val = last_targets.size(0)
            last_targets = last_targets.transpose(0, 1)
            
            for t in range(5):
                logits = last_outputs[t]
                _, topk = logits.topk(5, dim=1)
                preds = logits.argmax(dim=1)
                
                # 收集所有预测和目标
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(last_targets[t].cpu().numpy())
                
                # 计算准确率
                correct = (preds == last_targets[t]).sum().item()
                total += B_val
                top1_correct += correct
                
                for i in range(B_val):
                    gt = last_targets[t][i].item()
                    top2_correct += int(gt in topk[i, :2].cpu().numpy())
                    top3_correct += int(gt in topk[i, :3].cpu().numpy())
                    top5_correct += int(gt in topk[i, :5].cpu().numpy())
    
    # 计算准确率
    top1_acc = top1_correct / total
    top2_acc = top2_correct / total
    top3_acc = top3_correct / total
    top5_acc = top5_correct / total
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    if use_chinese:
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
    else:
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 生成分类报告
    report = classification_report(all_targets, all_preds, output_dict=True)
    
    # 可视化准确率
    plt.figure(figsize=(10, 6))
    accuracies = [top1_acc, top2_acc, top3_acc, top5_acc]
    labels = ['Top-1', 'Top-2', 'Top-3', 'Top-5']
    plt.bar(labels, accuracies, color=['blue', 'green', 'orange', 'red'])
    plt.ylim(0, 1.0)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    if use_chinese:
        plt.title('准确率测试结果')
        plt.ylabel('准确率')
    else:
        plt.title('Accuracy Results')
        plt.ylabel('Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'accuracy.png'))
    plt.close()
    
    # 打印结果
    print(f"Top-1 准确率: {top1_acc:.4f}")
    print(f"Top-2 准确率: {top2_acc:.4f}")
    print(f"Top-3 准确率: {top3_acc:.4f}")
    print(f"Top-5 准确率: {top5_acc:.4f}")
    print(f"测试结果已保存到: {output_dir}")
    
    # 返回结果
    results = {
        'top1_acc': top1_acc,
        'top2_acc': top2_acc,
        'top3_acc': top3_acc,
        'top5_acc': top5_acc,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return results

def visualize_predictions(model, test_loader, device, num_samples=5, output_dir="results/visualizations"):
    """
    可视化模型预测结果
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        num_samples: 可视化样本数量
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    use_chinese = setup_chinese_font()
    
    # 将模型设置为评估模式
    model.eval()
    
    # 可视化预测结果
    samples_processed = 0
    with torch.no_grad():
        for images, points, targets in test_loader:
            if samples_processed >= num_samples:
                break
                
            images = images.to(device)
            points = points.to(device)
            
            # 获取预测结果
            predictions = model.predict(images, points)
            
            # 处理每个批次中的样本
            for i in range(min(len(images), num_samples - samples_processed)):
                # 获取当前样本的预测和真实标签
                pred_seq = predictions[:, i].cpu().numpy()
                target_seq = targets[i].cpu().numpy()
                
                # 创建可视化图表
                plt.figure(figsize=(12, 6))
                plt.plot(range(len(target_seq)), target_seq, 'b-', label='真实波束' if use_chinese else 'True Beam')
                plt.plot(range(len(pred_seq)), pred_seq, 'r--', label='预测波束' if use_chinese else 'Predicted Beam')
                plt.xlabel('时间步' if use_chinese else 'Time Step')
                plt.ylabel('波束索引' if use_chinese else 'Beam Index')
                if use_chinese:
                    plt.title(f'样本 {samples_processed + 1} 的波束预测')
                else:
                    plt.title(f'Beam Prediction for Sample {samples_processed + 1}')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, f'sample_{samples_processed + 1}.png'))
                plt.close()
                
                samples_processed += 1
                if samples_processed >= num_samples:
                    break
    
    print(f"已生成 {samples_processed} 个样本的可视化结果，保存到: {output_dir}") 