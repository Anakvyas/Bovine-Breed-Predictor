# Project Structure

This document outlines the organization of the Cattle Breed Identification System codebase.

## Directory Structure

```
Breed-Predictor/
│
├── app.py                          # Main Flask application (run this!)
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
├── PROJECT_STRUCTURE.md           # This file
├── best_model.pth                 # Trained model weights (gitignored)
├── breed_traits.csv              # Breed characteristics data
│
├── templates/                     # Flask HTML templates
│   ├── index.html                # Main prediction interface
│   ├── demo.html                 # Breed database page
│   ├── documentation.html        # API documentation
│   └── support.html              # Support center
│
├── static/                        # Static web assets
│   ├── style.css                 # Additional CSS styles
│   └── results/                  # Generated prediction results
│
├── src/                          # Organized source code modules
│   ├── __init__.py              # Package initialization
│   │
│   ├── data_processing/         # Data preprocessing and preparation
│   │   ├── __init__.py         
│   │   ├── augmentation.py     # Data augmentation utilities
│   │   ├── dedup.py           # Dataset deduplication
│   │   └── split_train_val.py # Train/validation splitting
│   │
│   ├── training/               # Model training and optimization
│   │   ├── __init__.py        
│   │   └── train_cnn_final.py # Main training script
│   │
│   ├── evaluation/             # Model evaluation and testing
│   │   ├── __init__.py        
│   │   ├── evaluate_test.py           # Basic model evaluation
│   │   └── evaluate_test_with_mistakes.py # Error analysis
│   │
│   └── utils/                  # Utilities and helper functions
│       ├── __init__.py        
│       ├── demo-fusion_gradcam.py     # Grad-CAM visualization
│       ├── make_silhouettes.py        # Silhouette generation
│       └── predict_and_feedback.py    # Prediction utilities
│
└── data/ (gitignored)              # Training datasets
    ├── train/                      # Training images
    ├── val/                        # Validation images
    ├── test/                       # Test images
    └── silhouettes/                # Breed silhouette templates
```

## Module Descriptions

### 🌐 Main Application (`app.py`)
- **Purpose**: Flask web application and REST API  
- **Responsibilities**: HTTP request handling, web interface, API responses, Grad-CAM visualization

### 📊 Data Processing Module (`src/data_processing/`)
- **Purpose**: Dataset preparation and preprocessing
- **Files**:
  - `augmentation.py`: Image augmentation for training data
  - `dedup.py`: Remove duplicate images from dataset
  - `split_train_val.py`: Split dataset into train/validation sets
- **Responsibilities**: Data cleaning, augmentation, dataset organization

### 🧠 Training Module (`src/training/`)
- **Purpose**: Neural network training and model optimization
- **Files**:
  - `train_cnn_final.py`: Main training script with EfficientNet
- **Responsibilities**: Model training, hyperparameter tuning, checkpointing

### 📈 Evaluation Module (`src/evaluation/`)
- **Purpose**: Model performance assessment and analysis
- **Files**:
  - `evaluate_test.py`: Basic model evaluation metrics
  - `evaluate_test_with_mistakes.py`: Detailed error analysis
- **Responsibilities**: Accuracy measurement, error analysis, performance reporting

### 🔧 Utils Module (`src/utils/`)
- **Purpose**: Helper functions and visualization tools
- **Files**:
  - `demo-fusion_gradcam.py`: Grad-CAM heatmap generation
  - `make_silhouettes.py`: Create breed silhouette templates
  - `predict_and_feedback.py`: Prediction utilities and feedback handling
- **Responsibilities**: Visualization, utility functions, prediction helpers

## Usage Instructions

### Running the Application
```bash
# Start the web application
python app.py

# Access at http://localhost:5000
```

### Training a New Model
```bash
# Prepare dataset
python src/data_processing/split_train_val.py
python src/data_processing/augmentation.py

# Train model
python src/training/train_cnn_final.py

# Evaluate results
python src/evaluation/evaluate_test.py
```

### Data Processing Workflow
```bash
# 1. Remove duplicates
python src/data_processing/dedup.py

# 2. Split into train/val
python src/data_processing/split_train_val.py

# 3. Apply augmentation
python src/data_processing/augmentation.py
```

### Evaluation and Analysis
```bash
# Basic evaluation
python src/evaluation/evaluate_test.py

# Detailed error analysis
python src/evaluation/evaluate_test_with_mistakes.py

# Generate visualizations
python src/utils/demo-fusion_gradcam.py
```

## Development Guidelines

### Adding New Features
1. Place code in appropriate module based on functionality
2. Update `__init__.py` files with proper imports
3. Add documentation for new functions
4. Update this structure document

### Code Organization Principles
- **Separation of Concerns**: Each module has a specific purpose
- **Modularity**: Functions are reusable across modules
- **Maintainability**: Clear structure for easy updates
- **Professional Standards**: Enterprise-grade organization

### Import Guidelines
```python
# Use relative imports within modules
from ..utils import visualization_helpers
from .data_preprocessing import augment_image

# Import organized modules from src
from src.training.train_cnn_final import train_model
from src.data_processing.augmentation import apply_augmentation
```

