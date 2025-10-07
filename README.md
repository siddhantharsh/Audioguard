# 🎵 Audioguard: Urban Sound Classification

An AI-powered system for real-time urban sound classification using deep learning. The model can detect and classify 10 different types of urban sounds in real-time using your microphone input.

## 🎧 Supported Sound Classes

1. Air Conditioner
2. Car Horn
3. Children Playing
4. Dog Bark
5. Drilling
6. Engine Idling
7. Gun Shot
8. Jackhammer
9. Siren
10. Street Music

## 📊 Model Performance

The current model achieves:
- Overall Accuracy: 76%
- Best performing classes: 
  - Siren (86% F1-score)
  - Air Conditioner (82% F1-score)
- Detailed performance metrics and visualizations are available in `artifacts/reports/`

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- A working microphone for real-time demo

### Installation

1. Clone the repository:
```bash
git clone https://github.com/siddhantharsh/Audioguard.git
cd Audioguard
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv310
venv310\Scripts\activate

# Linux/Mac
python3 -m venv venv310
source venv310/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Real-time Sound Detection

Run the real-time demo to classify sounds using your microphone:
```bash
python src/realtime_demo.py
```
- The program will continuously listen to your microphone input
- Predictions will be shown in real-time with confidence scores
- Press Ctrl+C to stop

#### 2. Model Performance Analysis

To see detailed performance metrics and visualizations:
```bash
python src/analyze_performance.py
```

This will generate several visualizations in `artifacts/reports/`:
- Confusion Matrix
- ROC Curves
- Prediction Confidence Distribution
- Per-class Performance Metrics

## 📁 Project Structure

```
audioguard/
├── artifacts/                  # Generated artifacts
│   ├── charts/                # Training visualizations
│   ├── models/                # Trained models
│   └── reports/               # Performance reports
├── data/                      # Dataset folder
│   └── UrbanSound8K/         # Dataset documentation
├── src/                       # Source code
│   ├── analyze_performance.py # Performance analysis
│   ├── config.py             # Configuration
│   ├── dataset.py            # Data loading
│   ├── eval.py               # Model evaluation
│   ├── model.py              # Model architecture
│   ├── preprocess.py         # Audio preprocessing
│   ├── realtime_demo.py      # Real-time demo
│   └── train.py              # Training script
└── requirements.txt          # Dependencies
```

## 🔧 Model Architecture

The model uses a Transformer-based architecture:
- Log-mel spectrogram input
- Patch embedding layer
- Transformer encoder blocks
- Classification head

Key parameters:
- Input duration: 4 seconds
- Sample rate: 22050 Hz
- Mel bands: 64
- Patch size: 8
- Model dimension: 256
- Number of heads: 4
- Transformer layers: 4

## 🎯 Training Your Own Model

To train the model from scratch, you'll need the UrbanSound8K dataset:

1. Download the dataset from [UrbanSound8K website](https://urbansounddataset.weebly.com/)
2. Extract and place the audio files in `data/UrbanSound8K/audio/`
3. Run the training:
```bash
python src/train.py
```

## 📈 Performance Monitoring

Track the model's performance:
1. `artifacts/models/training_history.json`: Training metrics
2. `artifacts/reports/`: Latest evaluation results
3. `artifacts/charts/`: Performance visualizations

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 Citation

This project uses the UrbanSound8K dataset. If you use this code for academic research, please cite:

```
J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 
22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. - Urban Sound Classification

An audio classification model using Transformer architecture to classify urban sounds into 10 different categories. The model is trained on the UrbanSound8K dataset and can classify sounds like air conditioners, car horns, children playing, dog barks, drilling, engine idling, gunshots, jackhammers, sirens, and street music.

## Project Structure

```
mlproject/
│
├── artifacts/           # Generated artifacts
│   ├── charts/         # Visualization plots
│   ├── models/         # Saved models
│   └── data/           # Processed data
│
├── data/               # Raw data
│   └── UrbanSound8K/   # Dataset files
│
├── src/                # Source code
│   ├── config.py       # Configuration parameters
│   ├── dataset.py      # Data loading and preprocessing
│   ├── model.py        # Model architecture
│   ├── train.py        # Training script
│   ├── eval.py         # Evaluation script
│   ├── analyze_training.py  # Training analysis
│   └── realtime_demo.py     # Real-time inference
│
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Features

- Audio classification into 10 urban sound categories
- Transformer-based architecture for better temporal pattern recognition
- Real-time inference capabilities
- Training analysis and visualization tools
- Model evaluation metrics

## Setup

1. Create a virtual environment:
```bash
python -m venv venv310
source venv310/bin/activate  # On Windows use: venv310\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare the dataset:
   - Download UrbanSound8K dataset
   - Place it in the `data/` directory

## Usage

1. Training:
```bash
python src/train.py
```

2. Evaluation:
```bash
python src/eval.py
```

3. Analyze Training:
```bash
python src/analyze_training.py
```

4. Real-time Demo:
```bash
python src/realtime_demo.py
```

## Model Architecture

The model uses a Transformer architecture with:
- Log-mel spectrogram input
- Custom patch embedding layer
- Positional encoding
- Multi-head attention layers
- Dense feed-forward networks

## Performance

The model achieves:
- Training accuracy: [Your training accuracy]
- Validation accuracy: [Your validation accuracy]
- Test accuracy: [Your test accuracy]

## Future Improvements

- Add data augmentation techniques
- Experiment with different model architectures
- Implement cross-validation
- Add support for more audio formats
- Create web interface for demo

## License

[Add your license information]

## Acknowledgments

- UrbanSound8K dataset creators
- TensorFlow team
- [Add any other acknowledgments]