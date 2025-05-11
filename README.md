# Parallel and Distributed Computing: Binary Classification Project

## Overview
This project compares the performance of tree-based (XGBoost) and deep learning (PyTorch) models for binary classification under different compute configurations (serial CPU, parallel CPU, and GPU acceleration). The study focuses on optimizing processing time while maintaining classification accuracy.

## Key Features
- Comprehensive data preprocessing pipeline:
  - Missing value imputation
  - Categorical feature encoding
  - Quantile normalization
  - SMOTE for class imbalance
- Multiple model implementations:
  - XGBoost (serial CPU, parallel CPU, GPU)
  - Custom PyTorch neural network (CPU, GPU)
- Performance benchmarking:
  - Accuracy and F1 score comparisons
  - Training time measurements
  - Resource utilization analysis

## Dataset
The dataset (`pdc_dataset_with_target.csv`) contains:
- 40,100 samples
- 4 numerical features
- 3 categorical features
- Binary target label (moderate class imbalance)

## Technical Specifications
### Data Preprocessing
- Missing value imputation (mean/median based on skewness)
- One-hot encoding for categorical features
- Quantile transformation for numerical features
- SMOTE oversampling for class imbalance

### Models
**XGBoost:**
- Serial CPU (`n_jobs=1`)
- Parallel CPU (`n_jobs=-1`)
- GPU (`tree_method='gpu_hist'`)

**PyTorch Neural Network:**
- Architecture: [Input → 32 → 64 → 16 → 1]
- Activation: ReLU (Sigmoid output)
- Optimizer: Adam (lr=0.001)
- Loss: Binary Cross-Entropy
- GPU acceleration with AMP (Automatic Mixed Precision)

## Results Summary
| Model                | Accuracy | F1 Score | Training Time | Time Reduction |
|----------------------|----------|----------|---------------|----------------|
| XGBoost Serial CPU   | 0.5245   | 0.4327   | 0.61s         | -              |
| XGBoost Parallel CPU | 0.5245   | 0.4327   | 0.38s         | 37.7%          |
| XGBoost GPU          | 0.5197   | 0.4321   | 0.57s         | 6.55%          |
| PyTorch CPU          | 0.5084   | 0.4513   | 119.37s       | -              |
| PyTorch GPU          | 0.4898   | 0.4592   | 37.34s        | 71.8%          |

## Key Findings
- **Best Performance**: XGBoost (parallel CPU) achieved the best balance of speed (0.38s) and accuracy (52.45%)
- **GPU Acceleration**: Most effective for PyTorch (71.8% speedup) but less impactful for XGBoost (6.55% speedup)
- **F1 Scores**: PyTorch models performed better on class imbalance despite lower accuracy
- **Dataset Challenges**: Moderate accuracy across all models suggests inherent data complexity

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdc-classification-project.git
   cd pdc-classification-project
   ```

2. Install dependencies:
   ```bash
   pip install torch pandas numpy matplotlib scikit-learn imblearn xgboost
   ```

3. Download the dataset and place it in the `data` directory.

```

## Contributors
- Muhammad Qasim (221-1994)
- Ayaan Khan (221-2066)
- Abu Bakr Nadeem (221-2003)

## License
This project is licensed under the MIT License 

## References
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
- Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library.
