# Vision Transformer (ViT) from Scratch in PyTorch

## Overview
This repository contains an implementation of the **Vision Transformer (ViT)** from scratch using **PyTorch**. ViT is a transformer-based model for image classification that replaces traditional convolutional neural networks (CNNs) with self-attention mechanisms. This project is designed for learning, experimentation, and benchmarking transformer-based vision models.

## Features
- Implements Vision Transformer (ViT) architecture from scratch
- Tokenization of input images using patch embeddings
- Multi-head self-attention mechanism
- Position embeddings for spatial information
- End-to-end training pipeline for image classification
- Uses PyTorch for flexibility and performance

## Installation
To get started, clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/imaditya123/Vision-transformer.git
cd vision-transformer-pytorch

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Training the Vision Transformer
Run the following command to train the model:
```bash
python src/train.py 
```

<!-- ### Evaluating the Model
To evaluate a trained model:
```bash
python evaluate.py --model_path checkpoints/vit_model.pth
```

### Inference
To use the trained model for inference on a single image:
```bash
python inference.py --image_path path/to/image.jpg
``` -->

## Model Architecture
![vit_figure.png](/vit_figure.png)

- **Patch Embedding**: Splits input images into fixed-size patches and embeds them into a lower-dimensional space.
- **Position Embeddings**: Adds positional encodings to preserve spatial relationships.
- **Transformer Encoder**: Consists of multiple self-attention layers and feed-forward networks.
- **MLP Head**: Fully connected layers for classification.

## Dataset
This implementation supports **CIFAR-10** and **ImageNet** datasets. You can also provide your own dataset by modifying the data loader.

<!-- ## Results
| Model   | Dataset  | Accuracy |
|---------|---------|----------|
| |  |    |
-->

## References
- [Vision Transformer Paper (Dosovitskiy et al.)](https://arxiv.org/abs/2010.11929)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## Contributing
Feel free to contribute by submitting issues or pull requests. Improvements and suggestions are always welcome!

## License
This project is licensed under the **Apache-2.0 license**. See the [LICENSE](LICENSE) file for details.

