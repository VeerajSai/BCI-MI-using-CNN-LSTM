# 🧠 BCI Motor Imagery Classification using CNN-LSTM

<div align="center">
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)
  [![Keras](https://img.shields.io/badge/Keras-2.0+-red.svg)](https://keras.io/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
  [![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue.svg)](https://www.kaggle.com/code/veeraj16/cnn-lstm)
  
  *Decoding brain signals for motor imagery classification using hybrid deep learning*
  
  🧠 **Brain-Computer Interface** | 🎯 **Motor Imagery** | 🤖 **Deep Learning** | 📊 **EEG Analysis**
  
</div>

---

## 🌟 What is This Project About?

This project tackles one of the most fascinating challenges in **Brain-Computer Interface (BCI)** technology: **Motor Imagery (MI) classification**. By combining the spatial feature extraction power of **Convolutional Neural Networks (CNNs)** with the temporal sequence modeling capabilities of **Long Short-Term Memory (LSTM)** networks, we can decode brain signals and classify different motor imagery tasks.

**Motor Imagery** refers to the mental rehearsal of motor actions without actual movement - imagine moving your left hand, right hand, feet, or tongue. Our hybrid CNN-LSTM architecture can distinguish between these different imagined movements by analyzing EEG brain signals.

### 🔬 Key Features:

- 🧠 **EEG Signal Processing**: Advanced preprocessing of brain signals
- 🎨 **Spatial Feature Extraction**: CNN layers capture spatial patterns in EEG data
- 🔄 **Temporal Sequence Modeling**: LSTM networks learn temporal dependencies
- 🎯 **Multi-class Classification**: Classify different motor imagery tasks
- 📊 **Comprehensive Analysis**: Detailed visualization and performance metrics
- 🔧 **Production-Ready**: Clean, modular, and scalable implementation

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/VeerajSai/BCI-MI-using-CNN-LSTM.git
cd BCI-MI-using-CNN-LSTM

# Install dependencies
pip install -r requirements.txt

# Run the model training
python train.py
```

## 🏗️ Architecture Overview

```
EEG Data → Preprocessing → CNN Layers → Feature Maps → LSTM Layers → Dense → MI Classification
    ↓           ↓            ↓             ↓            ↓          ↓           ↓
Raw EEG    Filtering/    Spatial      Temporal     Sequential  Final     Left/Right/
Signals    Normalization  Features     Features     Memory     Dense     Feet/Tongue
```

### 🔍 Model Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Preprocessing** | EEG signal cleaning | Bandpass filtering, normalization, epoching |
| **CNN Block** | Spatial pattern extraction | Conv1D/2D, MaxPooling, Dropout |
| **LSTM Block** | Temporal dependency modeling | Bidirectional LSTM, sequence learning |
| **Classification** | Motor imagery prediction | Dense layers, softmax activation |

## 📊 Performance Metrics

<div align="center">

| Metric | Description | Status |
|--------|-------------|---------|
| **Accuracy** | Multi-class classification accuracy | ✅ Optimized |
| **Precision** | Class-wise prediction precision | ✅ High |
| **Recall** | True positive detection rate | ✅ Excellent |
| **F1-Score** | Balanced precision-recall metric | ✅ Robust |
| **Kappa Score** | Agreement beyond chance | ✅ Strong |
| **Training Time** | Model convergence time | ⚡ Efficient |

*Performance metrics are dataset-dependent and will vary based on subject and session*

</div>

## 🛠️ Implementation Details

### EEG Data Preprocessing
```python
def preprocess_eeg(raw_data, fs=250):
    # Bandpass filtering (8-30 Hz for motor imagery)
    filtered_data = bandpass_filter(raw_data, low=8, high=30, fs=fs)
    
    # Common Average Reference (CAR)
    car_data = apply_car(filtered_data)
    
    # Epoching and normalization
    epochs = create_epochs(car_data, events, tmin=0, tmax=4)
    normalized_epochs = normalize_epochs(epochs)
    
    return normalized_epochs
```

### Hybrid CNN-LSTM Architecture
```python
def create_cnn_lstm_model(input_shape, num_classes=4):
    model = Sequential([
        # CNN for spatial feature extraction
        Conv2D(32, (1, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((1, 2)),
        
        Conv2D(64, (1, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((1, 2)),
        Dropout(0.25),
        
        # Reshape for LSTM
        Reshape((timesteps, features)),
        
        # LSTM for temporal modeling
        LSTM(128, return_sequences=True, dropout=0.2),
        LSTM(64, dropout=0.2),
        
        # Classification layers
        Dense(50, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model
```

## 🎯 Motor Imagery Classes

### 🤚 Classification Tasks:
- **Left Hand**: Imagining left hand movement
- **Right Hand**: Imagining right hand movement  
- **Feet**: Imagining foot/feet movement
- **Tongue**: Imagining tongue movement

### 📈 Applications:
- **Assistive Technology**: Control devices for paralyzed patients
- **Rehabilitation**: Motor function recovery training
- **Gaming**: Mind-controlled gaming interfaces
- **Research**: Understanding brain motor control mechanisms


## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for faster training)
- 8GB+ RAM
- EEG datasets (BCI Competition IV, etc.)

### Dependencies Required
```bash
pip install tensorflow>=2.8.0
pip install keras>=2.8.0
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install mne>=0.24.0        # For EEG data handling
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install pandas>=1.3.0
```

## 📚 Usage Guide

### Training the Model
```python
# Train on BCI Competition dataset
python train.py --dataset bci_iv --subject 1 --epochs 100 --batch_size 32

# Train with custom EEG data
python train.py --data_path ./data/custom/ --model_type cnn_lstm --lr 0.001
```

### Evaluating Performance
```python
# Evaluate trained model
python evaluate.py --model_path ./results/models/best_model.h5 --test_data ./data/test/

# Cross-subject evaluation
python evaluate.py --cross_subject --subjects 1,2,3,4 --cv_folds 5
```

### Real-time Classification
```python
# Real-time BCI classification
python real_time_classify.py --model ./results/models/subject1_model.h5 --stream_data
```

## 🎨 Visualization & Analysis

### EEG Signal Visualization
- Raw and filtered EEG signals
- Topographic maps of brain activity
- Time-frequency analysis (spectrograms)
- Event-related potentials (ERPs)

### Model Analysis
- Training/validation curves
- Confusion matrices per class
- Feature importance visualization
- Classification accuracy per subject

### Performance Metrics
- Subject-specific accuracy analysis
- Cross-session validation results
- Statistical significance testing
- Comparison with baseline methods

## 🤝 Contributing

We welcome contributions to improve BCI motor imagery classification! Here's how you can help:

1. **🍴 Fork** the repository
2. **🌿 Create** your feature branch (`git checkout -b feature/NewPreprocessing`)
3. **💻 Commit** your changes (`git commit -m 'Add advanced filtering'`)
4. **🚀 Push** to the branch (`git push origin feature/NewPreprocessing`)
5. **📬 Open** a Pull Request

### Areas for Contribution
- [ ] Support for more EEG datasets
- [ ] Advanced preprocessing techniques
- [ ] Attention mechanisms for interpretability
- [ ] Real-time optimization
- [ ] Better Architecture
- [ ] Cross-subject transfer learning
- [ ] Web deployment

## 📊 Benchmarks & Comparisons

| Method | Approach | Advantages | Best Use Case |
|--------|----------|------------|---------------|
| **CNN-LSTM (This Project)** | **Hybrid Spatial-Temporal** | **Best of both worlds** | **Motor Imagery BCI** |
| CSP + SVM | Classical ML | Fast, interpretable | Simple motor tasks |
| Pure CNN | Deep spatial | Good for spatial patterns | Image-like EEG data |
| Pure LSTM/RNN | Sequential modeling | Temporal dependencies | Time series EEG |
| Transformer | Attention-based | Long-range dependencies | Complex sequences |

## 🏆 Project Highlights

- 🧠 **BCI Focus**: Specifically designed for brain-computer interface applications
- 🎯 **Motor Imagery**: Specialized for classifying imagined movements
- 📊 **EEG Expertise**: Advanced EEG signal processing and analysis
- 🔬 **Research Quality**: Based on established BCI research methodologies
- 🚀 **Production Ready**: Scalable architecture for real-world deployment
- 📈 **Comprehensive**: End-to-end pipeline from raw EEG to classification

## 🔮 Future Roadmap

- [ ] **Real-time Optimization**: Low-latency classification for online BCI
- [ ] **Advanced Architectures**: Attention mechanisms and transformer variants
- [ ] **Multi-modal Integration**: Combine EEG with other biosignals
- [ ] **Edge Deployment**: Optimize for embedded BCI systems
- [ ] **Clinical Validation**: Extensive testing with patient populations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  
  **⭐ Star this repo if you found it helpful for BCI research!**
  
  Made with ❤️ by [VeerajSai](https://github.com/VeerajSai)
  
  *"The mind is not a vessel to be filled, but a fire to be kindled." - Plutarch*
  
</div>
