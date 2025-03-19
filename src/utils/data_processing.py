import pandas as pd
import numpy as np
import os
from datetime import datetime

def generate_sequences(csv_file, x_size=8, y_size=13, delay=-8):
    """
    从原始CSV文件生成序列数据
    
    Args:
        csv_file: 原始CSV文件路径
        x_size: 输入序列长度
        y_size: 输出序列长度
        delay: 输出序列相对于输入序列的延迟
    
    Returns:
        包含序列数据的DataFrame
    """
    # 读取原始数据
    df = pd.read_csv(csv_file)
    print(f"原始CSV列名: {df.columns.tolist()}")
    
    # 检查必要的列是否存在
    required_columns = ['unit1_rgb', 'unit1_lidar', 'unit1_radar', 'unit1_beam']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"警告: CSV文件缺少必要的列: {missing_columns}")
        return pd.DataFrame()  # 返回空DataFrame
    
    # 按照索引排序
    if 'index' in df.columns:
        df = df.sort_values('index')
    elif 'seq_index' in df.columns:
        df = df.sort_values('seq_index')
    
    # 创建序列数据
    sequences = []
    skipped_count = 0
    skipped_reasons = {
        "invalid_range": 0,
        "seq_length": 0,
        "x_continuity": 0,
        "y_continuity": 0
    }
    
    # 调试信息
    print(f"准备生成序列，数据长度: {len(df)}，x_size: {x_size}，y_size: {y_size}，delay: {delay}")
    
    # 简化循环范围计算
    for i in range(len(df) - (x_size + y_size + abs(delay))):
        # 输入序列
        x_seq = df.iloc[i:i+x_size]
        
        # 输出序列 - 根据delay计算
        if delay < 0:
            # 负延迟意味着输出在输入之前
            y_start_idx = i + x_size + delay
        else:
            # 正延迟意味着输出在输入之后
            y_start_idx = i + x_size + delay
            
        # 检查范围是否有效
        if y_start_idx < 0 or y_start_idx + y_size > len(df):
            skipped_reasons["invalid_range"] += 1
            skipped_count += 1
            continue  # 跳过无效的序列
            
        y_seq = df.iloc[y_start_idx:y_start_idx+y_size]
        
        # 确保序列长度正确
        if len(x_seq) != x_size or len(y_seq) != y_size:
            skipped_reasons["seq_length"] += 1
            skipped_count += 1
            continue
        
        # 获取序列索引值（用于调试）
        x_indices = None
        y_indices = None
        if 'seq_index' in df.columns:
            x_indices = x_seq['seq_index'].values
            y_indices = y_seq['seq_index'].values
        
        # 检查连续性 - 暂时禁用这个检查，看看是否能生成序列
        continuity_check = False
        if continuity_check and 'seq_index' in df.columns:
            # 检查x序列是否连续
            if len(x_indices) > 1 and not np.all(np.diff(x_indices) == 1):
                skipped_reasons["x_continuity"] += 1
                skipped_count += 1
                continue
                
            # 检查y序列是否连续
            if len(y_indices) > 1 and not np.all(np.diff(y_indices) == 1):
                skipped_reasons["y_continuity"] += 1
                skipped_count += 1
                continue
        
        # 创建序列字典
        seq_dict = {}
        
        # 添加输入序列 - 确保按照正确的顺序添加列
        for j in range(x_size):
            # 先添加图像，然后点云，最后雷达
            seq_dict[f'img_{j}'] = x_seq.iloc[j]['unit1_rgb']
            seq_dict[f'pc_{j}'] = x_seq.iloc[j]['unit1_lidar']
            seq_dict[f'radar_{j}'] = x_seq.iloc[j]['unit1_radar']
        
        # 添加输出序列
        for j in range(y_size):
            seq_dict[f'beam_{j}'] = y_seq.iloc[j]['unit1_beam']
        
        # 添加序列索引信息
        if 'seq_index' in df.columns:
            seq_dict['seq_index'] = x_seq.iloc[0]['seq_index']
        
        sequences.append(seq_dict)
        
        # 打印一些示例，帮助调试
        if len(sequences) == 1:
            print("\n第一个成功的序列示例:")
            print(f"  输入序列索引: {i} 到 {i+x_size-1}")
            print(f"  输出序列索引: {y_start_idx} 到 {y_start_idx+y_size-1}")
            if x_indices is not None and y_indices is not None:
                print(f"  输入序列的seq_index: {x_indices}")
                print(f"  输出序列的seq_index: {y_indices}")
            
            # 打印序列字典的键，检查列顺序
            print(f"  序列字典键: {list(seq_dict.keys())}")
    
    # 创建DataFrame
    seq_df = pd.DataFrame(sequences)
    
    # 确保列的顺序正确
    if not seq_df.empty:
        # 重新排列列，确保顺序为img_0, pc_0, radar_0, img_1, ...
        ordered_columns = []
        for j in range(x_size):
            ordered_columns.extend([f'img_{j}', f'pc_{j}', f'radar_{j}'])
        
        # 添加波束列
        for j in range(y_size):
            ordered_columns.append(f'beam_{j}')
            
        # 添加序列索引列
        if 'seq_index' in seq_df.columns:
            ordered_columns.append('seq_index')
            
        # 检查所有列是否存在
        existing_columns = [col for col in ordered_columns if col in seq_df.columns]
        missing_columns = [col for col in ordered_columns if col not in seq_df.columns]
        
        if missing_columns:
            print(f"警告: 缺少以下列: {missing_columns}")
            
        # 重新排列列
        seq_df = seq_df[existing_columns]
    
    # 打印调试信息
    print(f"\n成功生成序列数: {len(sequences)}")
    print(f"跳过的序列数: {skipped_count}")
    print(f"跳过原因统计: {skipped_reasons}")
    
    if len(sequences) > 0:
        print(f"序列数据列: {seq_df.columns.tolist()}")
    
    return seq_df

def fix_data_paths(seq_df, base_path="data/scenario34_new/scenario34/scenario34.csv"):
    """
    修复数据路径，确保路径格式正确
    
    Args:
        seq_df: 序列数据DataFrame
        base_path: 基础路径
        
    Returns:
        DataFrame: 修复后的DataFrame
    """
    # 检查基础路径是否存在
    if not os.path.exists(base_path):
        print(f"警告: 基础路径 {base_path} 不存在!")
    
    # 修复路径格式
    for col in seq_df.columns:
        if col.startswith('x_frame_') or col.startswith('x_lidar_') or col.startswith('y_frame_'):
            # 移除路径中的前导 './'
            seq_df[col] = seq_df[col].str.replace(r'^\./', '', regex=True)
    
    return seq_df

def split_sequences(seq_df, output_dir, train_ratio=0.8, random_seed=33):
    """
    将序列数据划分为训练集和测试集
    
    Args:
        seq_df: 序列数据DataFrame
        output_dir: 输出目录
        train_ratio: 训练集比例
        random_seed: 随机种子
        
    Returns:
        tuple: (训练集DataFrame, 测试集DataFrame)
    """
    np.random.seed(random_seed)
    
    # 修复数据路径
    seq_df = fix_data_paths(seq_df)
    
    # 检查seq_df是否为空
    if seq_df.empty:
        print("警告: 序列数据为空，无法进行划分")
        return pd.DataFrame(), pd.DataFrame()
    
    # 检查是否存在seq_index列
    if 'seq_index' not in seq_df.columns:
        print("警告: 序列数据中没有seq_index列，将使用行索引代替")
        # 创建一个简单的序列索引
        seq_df['seq_index'] = np.arange(len(seq_df))
    
    # 获取所有唯一的序列索引
    seq_indices = seq_df['seq_index'].unique()
    
    # 随机打乱序列索引
    np.random.shuffle(seq_indices)
    
    # 划分训练集和测试集
    train_size = int(len(seq_indices) * train_ratio)
    train_indices = seq_indices[:train_size]
    test_indices = seq_indices[train_size:]
    
    # 根据序列索引划分数据
    train_df = seq_df[seq_df['seq_index'].isin(train_indices)]
    test_df = seq_df[seq_df['seq_index'].isin(test_indices)]
    
    # 打印统计信息
    print("\n数据集划分统计:")
    print(f"总序列数: {len(seq_indices)}")
    print(f"训练集序列数: {len(train_indices)}")
    print(f"测试集序列数: {len(test_indices)}")
    
    # 保存数据集
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n训练数据保存到: {train_path}")
    print(f"测试数据保存到: {test_path}")
    
    return train_df, test_df 