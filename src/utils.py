import torch
import torch.nn as nn
from tqdm import tqdm


def initialize_weights(x):
  nn.init.xavier_uniform_(x.weight)
  if x.bias is not None:
    nn.init.constant_(x.bias,0)

def save_model(model, optimizer, epoch, file_path):
    """
    Save the model and optimizer state dictionaries along with the epoch.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file_path)
    print(f"Model saved at epoch {epoch} to {file_path}")

def load_model(model, optimizer, file_path, device):
    """
    Load the model and optimizer state dictionaries from a checkpoint.

    Returns:
        epoch (int): The epoch number stored in the checkpoint.
    """
    checkpoint = torch.load(file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    model.to(device)
    print(f"Model loaded from {file_path}, starting at epoch {epoch}")
    return epoch


def test(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for images, labels in tqdm(test_loader,leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train(model, train_loader,test_loader, criterion, optimizer, device,num_epochs=10):
    
    for epoch in range(num_epochs):
      model.train()  # Set model to training mode
      running_loss = 0.0
      correct = 0
      total = 0
      for images, labels in tqdm(train_loader):
          images, labels = images.to(device), labels.to(device)  # Move to GPU/CPU

          optimizer.zero_grad()  # Reset gradients
          outputs = model(images)  # Forward pass
          loss = criterion(outputs, labels)  # Compute loss
          loss.backward()  # Backpropagation
          optimizer.step()  # Update weights

          running_loss += loss.item()
          _, predicted = torch.max(outputs, 1)  # Get predictions
          correct += (predicted == labels).sum().item()
          total += labels.size(0)

      train_avg_loss = running_loss / len(train_loader)
      train_accuracy = 100 * correct / total
      test_loss,test_acc=test(model, test_loader, criterion, device)
      print(f"Training: Loss({train_avg_loss:4f}), Acc({train_accuracy:4f})| Test: Loss({test_loss:4f}), Acc({test_acc:4f})")

