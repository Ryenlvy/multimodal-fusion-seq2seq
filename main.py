import os
import argparse
import yaml
import sys
import logging
from datetime import datetime

def check_environment():
    """Check environment compatibility"""
    try:
        import numpy as np
        if int(np.__version__.split('.')[0]) >= 2:
            print("Warning: Detected NumPy 2.x version, which may cause compatibility issues.")
            print("Suggestion: Please downgrade to NumPy 1.x, for example, run: pip install numpy<2.0.0")
            return False
        
        import torch
        import open3d
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install the required dependencies first by running: python setup_env.py")
        return False

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Image-Point Cloud Fusion Sequence Prediction Model')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'prepare_data', 'all'], default='train', help='Running mode')
    parser.add_argument('--data_csv', type=str, default='data/scenario34_new/scenario34/scenario34.csv', help='Path to raw data CSV')
    parser.add_argument('--output_dir', type=str, default='./prepared_data', help='Data output directory')
    parser.add_argument('--model_path', type=str, default='checkpoint/fusion_seq2seq_model.pth', help='Model path for test mode')
    parser.add_argument('--results_dir', type=str, default='results', help='Test results output directory')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize test results')
    parser.add_argument('--num_vis_samples', type=int, default=5, help='Number of samples to visualize')
    return parser.parse_args()

def main():
    """Main function"""
    # Check environment
    if not check_environment():
        sys.exit(1)
        
    # Import dependencies (import after environment check to avoid import errors)
    import torch
    from torch.utils.data import DataLoader
    
    from src.data.dataset import FusionDataset
    from src.models.fusion_model import FusionSeq2Seq
    from src.utils.data_processing import generate_sequences, split_sequences
    from src.train import train, setup_logger
    from test.test_model import test_model, visualize_predictions
    
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger = setup_logger()
    logger.info(f"Loaded configuration file: {args.config}")
    
    # Validate configuration
    validation_error = validate_config(config)
    if validation_error:
        logger.error(f"Configuration file validation failed: {validation_error}")
        sys.exit(1)
    
    # Check data paths
    config = check_data_paths(config)
    
    # Perform actions based on mode
    if args.mode == 'prepare_data' or args.mode == 'all':
        prepare_data(args)
    
    if args.mode == 'train' or args.mode == 'all':
        logger.info("Starting model training...")
        model, best_acc = train(config, logger)
        logger.info(f"Training completed, best accuracy: {best_acc:.4f}")
    
    if args.mode == 'test' or args.mode == 'all':
        logger.info("Starting model testing...")
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FusionSeq2Seq(
            fused_dim=int(config['model']['fused_dim']),
            encoder_hidden_size=int(config['model']['encoder_hidden_size']),
            decoder_hidden_size=int(config['model']['decoder_hidden_size']),
            num_layers=int(config['model']['num_layers']),
            beam_embedding_dim=int(config['model']['beam_embedding_dim']),
            num_beams=int(config['model']['num_beams']),
            seq_out_len=int(config['model']['seq_out_len']),
            teacher_forcing_ratio=float(config['model']['teacher_forcing_ratio'])
        ).to(device)
        
        # If coming from training mode, no need to reload the model
        if args.mode == 'test':
            logger.info(f"Loading model: {args.model_path}")
            model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        # Load test data
        test_dataset = FusionDataset(
            config['data']['val_csv'], 
            base_path=config['data']['base_path'],
            num_points=config['data']['num_points']
        )
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
        
        # Test the model
        results = test_model(model, test_loader, device, num_beams=config['model']['num_beams'], output_dir=args.results_dir)
        
        # Visualize predictions
        if args.visualize:
            logger.info("Generating prediction visualizations...")
            visualize_predictions(model, test_loader, device, num_samples=args.num_vis_samples, 
                                 output_dir=os.path.join(args.results_dir, 'visualizations'))

def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_config(config):
    """Validate whether configuration parameters are complete and correctly typed"""
    required_params = {
        'training': {
            'num_epochs': int,
            'batch_size': int,
            'learning_rate': float,
            'patience': int,
            'warmup_epochs': int
        },
        'data': {
            'train_csv': str,
            'val_csv': str,
            'base_path': str
        },
        'model': {
            'num_beams': int,
            'fused_dim': int,
            'beam_embedding_dim': int,
            'seq_out_len': int
        }
    }
    
    # Optional parameters and their default values
    optional_params = {
        'model': {
            'hidden_dim': (int, 512),
            'num_layers': (int, 2),
            'dropout': (float, 0.1),
            'teacher_forcing_ratio': (float, 0.5)
        },
        'data': {
            'num_points': (int, 1024),
            'radar_points': (int, 256)
        },
        'training': {
            'num_workers': (int, 2)
        }
    }
    
    # Validate required parameters
    for section, params in required_params.items():
        if section not in config:
            return f"Missing configuration section: '{section}'"
        
        for param, param_type in params.items():
            if param not in config[section]:
                return f"Missing parameter: '{section}.{param}'"
            
            try:
                param_type(config[section][param])
            except ValueError:
                return f"Parameter type error: '{section}.{param}' should be {param_type.__name__}"
    
    # Validate optional parameters (if present)
    for section, params in optional_params.items():
        if section in config:
            for param, (param_type, _) in params.items():
                if param in config[section]:
                    try:
                        param_type(config[section][param])
                    except ValueError:
                        return f"Parameter type error: '{section}.{param}' should be {param_type.__name__}"
    
    return None

def check_data_paths(config):
    """Check if data paths exist"""
    base_path = config['data']['base_path']
    train_csv = config['data']['train_csv']
    val_csv = config['data']['val_csv']
    
    # Check base path
    if not os.path.exists(base_path):
        print(f"Warning: Base path {base_path} does not exist!")
        # Attempt to find the correct path
        if os.path.exists('./data/scenario34'):
            print(f"Found possible alternative path: ./data/scenario34")
            config['data']['base_path'] = './data/scenario34'
        elif os.path.exists('/home/ryne/workplace/fpointnet/combine/data/scenario34'):
            print(f"Found possible alternative path: /home/ryne/workplace/fpointnet/combine/data/scenario34")
            config['data']['base_path'] = '/home/ryne/workplace/fpointnet/combine/data/scenario34'
        elif os.path.exists('../data/scenario34'):
            print(f"Found possible alternative path: ../data/scenario34")
            config['data']['base_path'] = '../data/scenario34'
    
    # Check CSV files
    if not os.path.exists(train_csv):
        print(f"Warning: Training data CSV {train_csv} does not exist!")
    
    if not os.path.exists(val_csv):
        print(f"Warning: Validation data CSV {val_csv} does not exist!")
    
    return config
