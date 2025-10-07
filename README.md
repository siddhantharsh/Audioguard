# Audioguard - Urban Sound Classification

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