import subprocess
import sys
import os

def setup_environment():
    """设置项目环境"""
    print("开始设置项目环境...")
    
    # 检查是否在conda环境中
    in_conda = os.environ.get('CONDA_DEFAULT_ENV') is not None
    
    if in_conda:
        print("检测到Conda环境，使用conda安装依赖...")
        # 首先安装numpy 1.x版本
        subprocess.check_call([
            "conda", "install", "-y", "numpy<2.0.0"
        ])
        
        # 然后安装其他依赖
        subprocess.check_call([
            "conda", "install", "-y", 
            "pytorch", "torchvision", "torchaudio", 
            "pandas", "matplotlib", "seaborn", "scikit-learn", "pyyaml", "tqdm"
        ])
        
        # 单独安装open3d（可能需要从pip安装）
        try:
            subprocess.check_call(["conda", "install", "-y", "open3d"])
        except:
            print("从conda安装open3d失败，尝试使用pip安装...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "open3d"])
    else:
        print("使用pip安装依赖...")
        # 使用pip安装依赖
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
    
    # 验证安装
    try:
        import numpy
        import torch
        import open3d
        print(f"NumPy版本: {numpy.__version__}")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"Open3D版本: {open3d.__version__}")
        print("环境设置成功!")
    except ImportError as e:
        print(f"安装验证失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    setup_environment() 