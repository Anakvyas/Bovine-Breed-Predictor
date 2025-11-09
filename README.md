# = Cattle Breed Predictor

An AI-powered cattle breed identification system using computer vision and machine learning. This application predicts cattle breeds from images using a trained EfficientNet model and provides visual explanations through Grad-CAM heatmaps.

## =� Features

- **Multi-Breed Classification**: Identifies 5 cattle breeds (Gir, Holstein-Friesian, Murrah, Jersey, Nili Ravi)
- **Visual Explanations**: Grad-CAM heatmaps show which image regions influenced predictions
- **Multi-Modal Analysis**: Combines image analysis with cattle traits (milk, weight, lactation, disease scores)
- **Crossbreed Detection**: Flags potential crossbreeds when confidence is low
- **Silhouette Overlays**: Visual breed comparison with silhouette templates
- **Feedback System**: Collects user feedback to improve model performance
- **REST API**: Easy integration with web applications

## =� Supported Breeds

| Breed | Code | Characteristics |
|-------|------|----------------|
| Gir | `Gir` | Indigenous Indian breed, heat-resistant |
| Holstein-Friesian | `H_F` | High milk production, black and white |
| Murrah | `Murrah` | Buffalo breed, excellent for dairy |
| Jersey | `Jersey` | Small size, high butterfat content |
| Nili Ravi | `nili_ravi` | Buffalo breed, good milk yield |

## =� Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster inference)
- 4GB+ RAM

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd Breed-Predictor

# Install dependencies
pip install -r requirements.txt

# Download or train the model (see Training section)
# Ensure best_model.pth is in the project root
```

### Dependencies
- **Flask**: Web framework for API
- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **OpenCV**: Image processing
- **Pillow**: Image handling
- **NumPy**: Numerical computations
- **Flask-CORS**: Cross-origin resource sharing

## =� Quick Start

### 1. Start the Server
```bash
python app.py
```
The server will start at `http://localhost:5000`

### 2. Test with API
```bash
curl -X POST \
  -F "image=@path/to/cattle_image.jpg" \
  -F "milk=15.0" \
  -F "weight=450" \
  -F "lact=10" \
  -F "disease=2" \
  http://localhost:5000/predict
```

## =� API Documentation

### POST `/predict`
Predicts cattle breed from uploaded image and traits.

**Request:**
- **Form Data:**
  - `image`: Image file (JPG/PNG)
  - `milk`: Milk production (liters/day, optional)
  - `weight`: Weight in kg (optional)
  - `lact`: Lactation period (optional)
  - `disease`: Disease resistance score (optional)

**Response:**
```json
{
  "top3": [
    ["Jersey", 0.85],
    ["Gir", 0.12],
    ["H_F", 0.03]
  ],
  "final_pred": "Jersey",
  "cnn_conf": 0.85,
  "crossbreed_flag": false,
  "heatmap_url": "/static/results/image_20231109_heat.jpg",
  "silhouette_url": "/static/results/image_20231109_sil_overlay.jpg"
}
```

### POST `/feedback`
Submit user feedback for model improvement.

**Request:**
```json
{
  "image": "filename.jpg",
  "predicted_final": "Jersey",
  "action": "correct|incorrect",
  "scores": {"user_rating": 5}
}
```

## >� Model Architecture

- **Base Model**: EfficientNet-B0
- **Input Size**: 224�224 pixels
- **Classes**: 5 cattle breeds
- **Training**: Transfer learning with data augmentation
- **Inference**: CPU/GPU/MPS support

### Model Performance
- **Validation Accuracy**: ~92%
- **Model Size**: 16MB
- **Inference Time**: ~200ms (CPU), ~50ms (GPU)

## <� Training Your Own Model

### Data Preparation
```bash
# Organize your dataset:
dataset/

