# 🧠 Parkinson's Handwriting Detection AI

A healthcare-oriented deep learning system built using **TensorFlow**, **Keras**, and **Flask** to analyze handwriting patterns and assist in early Parkinson's disease screening.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)
- [License](#license)

## 🎯 Overview

Parkinson's disease is a neurodegenerative disorder that affects movement control. One of the earliest visible signs is **micrographia**—changes in handwriting patterns where handwriting becomes small, cramped, and irregular.

This project leverages **Convolutional Neural Networks (CNNs)** to automatically analyze handwriting images and detect patterns associated with Parkinson's disease. The system is deployed as a web application using Flask, making it accessible and user-friendly.

## ✨ Features

- **Deep Learning Model**: Custom CNN architecture trained on handwriting samples
- **Web Interface**: User-friendly Flask-based web application
- **Real-time Predictions**: Instant analysis of uploaded handwriting images
- **Confidence Scoring**: Shows prediction confidence percentage
- **Image Processing**: Automatic preprocessing and normalization of input images
- **Early Stopping**: Prevents overfitting with automatic model monitoring
- **Detailed Metrics**: Evaluation with confusion matrix and classification metrics

## 📁 Project Structure

```
parkinsons/
├── app.py                      # Flask web application
├── train.py                    # Model training script
├── parkinson_detector.h5       # Trained model weights
├── best_model.h5               # Best model checkpoint during training
├── templates/
│   └── index.html              # Web interface HTML
├── static/                     # Static assets (CSS, JS, images)
├── Dataset/
│   ├── Healthy/                # Handwriting samples from healthy individuals
│   └── Parkinson/              # Handwriting samples from Parkinson's patients
├── env/                        # Python virtual environment
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/parkinsons-detection.git
cd parkinsons-detection
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv env
env\Scripts\activate

# macOS/Linux
python3 -m venv env
source env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies included:**

- TensorFlow 2.15.0 - Deep learning framework
- Keras 2.15.0 - Neural network API
- OpenCV 4.12.0.88 - Image processing
- NumPy 2.2.6 - Numerical computing
- Scikit-learn 1.7.2 - Machine learning utilities
- Flask 3.1.2 - Web framework
- Pillow 12.0.0 - Image manipulation
- SciPy 1.16.3 - Scientific computing
- h5py 3.15.1 - HDF5 file format support
- Joblib 1.5.2 - Serialization

### Step 4: Verify Installation

```bash
python -c "import tensorflow; import flask; import cv2; print('✅ All packages installed successfully!')"
```

## 💻 Usage

### Training the Model

To train the model on your dataset:

```bash
python train.py
```

This will:

1. Load images from the `Dataset/Healthy/` and `Dataset/Parkinson/` directories
2. Preprocess and normalize the images
3. Split data into training (80%) and testing (20%) sets
4. Train the CNN model with data augmentation
5. Save the trained model as `parkinson_detector.h5`
6. Display detailed evaluation metrics

### Running the Web Application

```bash
python app.py
```

Then open your browser and navigate to:

```
http://localhost:5000
```

### Making Predictions

1. Upload a handwriting image (PNG, JPG, etc.)
2. Click the "Predict" button
3. View the result with confidence percentage

## 🧠 Model Architecture

The model uses a **Convolutional Neural Network (CNN)** with the following architecture:

```
Input: (128, 128, 1) - Grayscale images

Conv2D(32 filters, 3x3) → ReLU
MaxPooling2D(2x2)
↓
Conv2D(64 filters, 3x3) → ReLU
MaxPooling2D(2x2)
↓
Flatten()
↓
Dense(128) → ReLU → Dropout(0.5)
↓
Dense(2) → Softmax (Output: [Healthy, Parkinson])
```

**Training Configuration:**

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Batch Size: 16
- Max Epochs: 30 (with early stopping)
- Data Augmentation: Rotation (15°), Zoom (10%), Horizontal Flip
- Early Stopping Patience: 5 epochs

## 📊 Dataset

The project expects the following directory structure for datasets:

```
Dataset/
├── Healthy/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── Parkinson/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

**Image Requirements:**

- Format: PNG, JPG, JPEG
- Recommended Size: 128×128 pixels (automatically resized)
- Color: Grayscale or RGB (converted to grayscale)

## 📈 Results

The model evaluation includes:

- **Test Accuracy**: Overall accuracy on unseen test data
- **Confusion Matrix**: True/False Positives and Negatives
- **Precision & Recall**: Per-class performance metrics
- **F1-Score**: Harmonic mean of precision and recall

Example output:

```
Test accuracy: 92.50%

Confusion Matrix:
            Predicted
             H    P
Actual H  [ 85   5]
       P  [ 3   87]

Healthy Metrics:
Precision: 0.9659
Recall: 0.9444
F1-score: 0.9550

Parkinson Metrics:
Precision: 0.9457
Recall: 0.9667
F1-score: 0.9561
```

## 🛠️ Technologies Used

| Technology       | Version   | Purpose                                |
| ---------------- | --------- | -------------------------------------- |
| **Python**       | 3.11      | Programming language                   |
| **TensorFlow**   | 2.15.0    | Deep learning framework                |
| **Keras**        | 2.15.0    | Neural network API                     |
| **OpenCV**       | 4.12.0.88 | Image processing and computer vision   |
| **NumPy**        | 2.2.6     | Numerical computing and arrays         |
| **Scikit-learn** | 1.7.2     | Machine learning utilities and metrics |
| **Flask**        | 3.1.2     | Web framework                          |
| **Pillow**       | 12.0.0    | Image manipulation                     |
| **SciPy**        | 1.16.3    | Scientific computing                   |
| **h5py**         | 3.15.1    | HDF5 file format support               |
| **Joblib**       | 1.5.2     | Parallel computing and serialization   |

## � Dependencies & Installation

All required packages are listed in `requirements.txt`. To install them:

```bash
# Activate your virtual environment first
# Windows: env\Scripts\activate
# macOS/Linux: source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
pip list
```

**Key Dependencies:**

- **Deep Learning**: TensorFlow/Keras for model training and inference
- **Image Processing**: OpenCV and Pillow for image manipulation
- **Data Processing**: NumPy, SciPy for numerical operations
- **ML Utilities**: Scikit-learn for metrics and preprocessing
- **Web Framework**: Flask for the web interface
- **File I/O**: h5py for saving/loading model weights

## �🔧 Configuration

### Model Hyperparameters

Edit `train.py` to adjust:

```python
img_size = 128              # Image resolution
batch_size = 16             # Training batch size
epochs = 30                 # Maximum training epochs
patience = 5                # Early stopping patience
learning_rate = 0.001       # Model learning rate (Adam optimizer)
```

### Flask Configuration

Edit `app.py` to adjust:

```python
UPLOAD_FOLDER = 'uploads'   # Directory for uploaded images
MAX_FILE_SIZE = 16 * 1024 * 1024  # Maximum upload size (16MB)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
```

## 🔮 Future Improvements

- [ ] Add model explainability with GradCAM visualization
- [ ] Implement user authentication and result history
- [ ] Add support for batch processing
- [ ] Deploy to cloud (AWS, Google Cloud, Heroku)
- [ ] Create mobile app version
- [ ] Add more advanced architectures (ResNet, DenseNet)
- [ ] Implement A/B testing for different models
- [ ] Add real-time performance monitoring
- [ ] Create API endpoint for third-party integration
- [ ] Add multi-language support

## 🤝 Contributors

- **Akhilesh Adepu** - Lead Developer
- **Eshwar Prasad** - ML Engineering
- **Vyshnavi** - Data Processing
- **Valmi** - Frontend Development

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

## ⚠️ Disclaimer

This tool is designed for **educational and research purposes only** and should **NOT** be used as a medical diagnostic tool. Always consult with healthcare professionals for medical diagnosis and treatment. The model predictions are probabilistic and may have limitations.

## � Troubleshooting

### Issue: TensorFlow installation fails

**Solution**: Install CUDA toolkit and cuDNN for GPU support, or use CPU-only version:

```bash
pip install tensorflow-cpu
```

### Issue: Model file not found (parkinson_detector.h5)

**Solution**: Train the model first:

```bash
python train.py
```

### Issue: Port 5000 already in use

**Solution**: Change the port in `app.py`:

```python
app.run(debug=True, port=5001)
```

### Issue: Image not recognized

**Solution**: Ensure the image is in PNG, JPG, or JPEG format and is not corrupted

### Issue: Low prediction accuracy

**Solution**:

- Add more training data to the Dataset folders
- Increase epochs in `train.py`
- Check if images are clear and properly preprocessed

### Issue: Out of Memory error during training

**Solution**: Reduce batch size in `train.py`:

```python
batch_size = 8  # Reduce from 16
```

## �📞 Support

For issues, questions, or suggestions, please open an issue on GitHub or contact the development team.

## 🙏 Acknowledgments

- Dataset sourced from publicly available Parkinson's handwriting datasets
- Built with guidance from modern deep learning best practices
- Inspired by healthcare AI research and accessibility initiatives

---

**Last Updated:** November 2025  
**Version:** 1.0.0
