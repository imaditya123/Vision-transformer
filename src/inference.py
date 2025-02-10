import argparse
import config
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from model import ViT
from dataset import getClasses
import utils
from PIL import Image

def inference(image_path:str):
    class_names=getClasses()
    # Data Transformations
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),  # Resize to ViT input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    model = ViT(img_size=config.IMG_SIZE,
                patch_size=config.PATCH_SIZE, 
                hidden_dim=config.HIDDEN_DIM, 
                filter_size=config.FILTER_SIZE, 
                num_heads=config.NUM_HEADS, 
                n_layers=config.N_LAYERS, 
                dropout_rate=config.DROPOUT_RATE, 
                num_classes=config.NUM_CLASSES).to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    utils.load_model(model=model,device=config.DEVICE,optimizer=optimizer,file_path="models/vit_model_v1.pth")
   # Load Image
    image = Image.open(image_path).convert("RGB")  # Ensure 3-channel image
    image = transform(image).unsqueeze(0).to(config.DEVICE)  # Add batch dimension

    # Model Inference
    model.eval()
    with torch.no_grad():
        output = model(image)  # Get model predictions
        predicted_class = torch.argmax(output, dim=1).item()  # Get class index

    # Print the predicted class
    print(f"Predicted Class: {class_names[predicted_class]}")  # Convert index to class name





if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser(description="Predict image class using Vision Transformer (ViT).")
    
    # Add Argument for Image Path
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image.")
    
    # Parse Arguments
    args = parser.parse_args()

    # Run Inference
    inference(args.image_path)
