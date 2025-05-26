# RainHCNet
RainHCNet is a PyTorch-based spatiotemporal sequence precipitation nowcasting model. For more information, please refer to RainHCNet

File Structure
**train_seq.npy & test_seq.npy**: Files defining the data sequence order for training and testing on KNMI datasets.

**tool.py**: Contains essential preprocessing functions including data transformation, evaluation metrics, and visualization utilities.

**RainHCNet/HCS_Attention.py**: Implementation of the Hybrid Channel-Spatial attention (HCSA) module.

**RainHCNet/RainHCNet.py**: Core network architecture implementing encoder-decoder structure with cross-scale supervision module.

**RainHCNet/train.py**: Training pipeline for the model.

**RainHCNet/test.py**: Testing and evaluation pipeline for the model.

# Dataset Requirements
To use this model, you need to apply for access to the following datasets:

KNMI Dataset: Apply through the official KNMI portal at KNMI.

SEVIR Dataset: Apply through the official SEVIR (Storm Event Imagery) dataset portal at SEVIR.

Shanghai Dataset: Apply through the official Shanghai meteorological data portal at Shanghai.

# Train
Once you have obtained the required datasets, you can either load pre-trained models or train the model from scratch: python RainHCNet-main/RainHCNet/train.py

# Test
To evaluate the model performance: python RainHCNet-main/RainHCNet/test.py

# Environment
Python 3.6+, Pytorch 1.0 and Ubuntu.

# Citation
If you use this code in your research, please cite our paper:
 ```
@ARTICLE{10918910,
author={Wang, Lei and Wang, Zheng and Hu, Wenjun and Bai, Cong},
journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
title={RainHCNet: Hybrid High-Low Frequency and Cross-Scale Network for Precipitation Nowcasting},
year={2025},
volume={18},
number={},
pages={8923-8937},
keywords={Rain;Computational modeling;Radar;Forecasting;Predictive models;Meteorological radar;Feature extraction;Accuracy;Radar tracking;Long short term memory;High-intensity rainfall;low-frequency information;multi-scale features learning;precipitation nowcasting},
doi={10.1109/JSTARS.2025.3549678}}
 ```
