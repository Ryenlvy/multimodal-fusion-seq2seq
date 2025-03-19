# Multimodal Fusion Sequence Prediction Model

## Project Overview

This project implements a multimodal fusion sequence prediction model designed to process and integrate image, point cloud, and radar data for sequence prediction. The model follows an encoder-decoder architecture and utilizes a GRU network for sequence processing, making it suitable for beam prediction tasks in autonomous driving scenarios.

## Features

- Multimodal data fusion (Image, Point Cloud, Radar)
- Sequence-to-sequence prediction architecture
- Memory-optimized design for large-scale data processing
- Full pipeline support for training, testing, and data preparation

## Installation

```bash

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

Before running the model, prepare the dataset using the following command:

```bash
python main.py --mode prepare_data --data_csv path/to/your/data.csv --output_dir ./prepared_data
```

## Model Training

Train the model using the command:

```bash
python main.py --mode train --config config/default.yaml
```

## Model Testing

After training, test the model using:

```bash
python main.py --mode test --model_path checkpoint/fusion_seq2seq_model.pth --results_dir results
```

To visualize the prediction results, add the `--visualize` parameter:

```bash
python main.py --mode test --model_path checkpoint/fusion_seq2seq_model.pth --results_dir results --visualize --num_vis_samples 5
```

## Project Structure

```
multimodal-fusion-seq2seq/
├── config/                  # Configuration files
│   └── default.yaml         # Default configuration
├── data/                    # Data directory
├── src/                     # Source code
│   ├── data/                # Data processing module
│   │   └── dataset_new.py   # Dataset class
│   ├── models/              # Model definitions
│   │   ├── fusion_model.py  # Fusion model
│   │   └── feature_extractors.py # Feature extractors
│   └── utils/               # Utility functions
├── test/                    # Test scripts
├── logs/                    # Logs directory
├── checkpoint/              # Model checkpoints
├── results/                 # Test results
├── main.py                  # Main script
└── README.md                # Project documentation
```

## Model Architecture

The project implements a multimodal fusion sequence prediction model, consisting of the following components:

1. **Feature Extractors**:
   - Image Feature Extractor (ResNet50-based)
   - Point Cloud Feature Extractor (PointNet-based)
   - Radar Feature Extractor (using 2D FFT and ResNet50)

2. **Multimodal Fusion Module**:
   - Fuses features from different modalities into a unified representation

3. **Sequence Encoder**:
   - Uses a GRU network to encode input sequences

4. **Sequence Decoder**:
   - Uses a GRU network to generate output sequences
   - Supports teacher forcing training


## Memory Optimization

This project employs various memory optimization strategies to handle large-scale multimodal data efficiently:

1. Dynamic batch size adjustment
2. Gradient accumulation
3. Mixed precision training
4. Memory caching management
5. Optimized number of worker processes

## TODO List

- [ ] Use pixel-level fusion instead of simple early fusion in the future  
- [ ] Lightweight model optimization  
- [ ] Data augmentation
## Citation

If you use this project in your research, please cite:

```
@misc{multimodal-fusion-seq2seq,
  author = {Ryenlvy},
  title = {Multimodal Fusion Sequence-to-Sequence Model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Ryenlvy/multimodal-fusion-seq2seq}}
}
```

## License

MIT