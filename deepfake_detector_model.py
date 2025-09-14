# Deepfake Detection with PyTorch using Transfer Learning
# Complete Pipeline: Data Loading, Preprocessing, Training, Evaluation

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ================== Dataset Class ==================
class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with subdirectories 'real' and 'fake'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load real images (label 0)
        real_dir = os.path.join(data_dir, 'real')
        for img_name in os.listdir(real_dir):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                self.images.append(os.path.join(real_dir, img_name))
                self.labels.append(0)
                
        # Load fake images (label 1)
        fake_dir = os.path.join(data_dir, 'fake')
        for img_name in os.listdir(fake_dir):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                self.images.append(os.path.join(fake_dir, img_name))
                self.labels.append(1)
        
        print(f"Loaded {len(self.images)} images from {data_dir}")
        print(f"Real images: {self.labels.count(0)}, Fake images: {self.labels.count(1)}")
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image using PIL
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ================== Model Definition ==================
class DeepfakeDetector(nn.Module):
    def __init__(self, fine_tune=True):
        super(DeepfakeDetector, self).__init__()
        
        # Load pre-trained EfficientNet model
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        
        if fine_tune:
            # Freeze early layers
            for param in list(self.efficientnet.parameters())[:-20]:
                param.requires_grad = False
        
        # Replace the final classifier
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_ftrs, 2)  # 2 output classes: real and fake
        )
        
    def forward(self, x):
        return self.efficientnet(x)

# ================== Training Function ==================
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    # Track best model weights
    best_acc = 0.0
    best_model_weights = None
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            # Update best model weights
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = model.state_dict().copy()
                
        print()
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    return model, history

# ================== Evaluation Functions ==================
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake class
    
    # Print classification metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Compute ROC curve 
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()
    
    return {
        'classification_report': classification_report(all_labels, all_preds, target_names=['Real', 'Fake'], output_dict=True),
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr
    }

def plot_training_history(history):
    """Plot training and validation loss and accuracy"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# ================== Main Execution ==================
def main():
    # Data directories - adjust these to your paths
    train_dir = r'C:\Users\anoop\Downloads\archive\deepfake_dataset\train'
    val_dir = r'C:\Users\anoop\Downloads\archive\deepfake_dataset\val'
    test_dir = r'C:\Users\anoop\Downloads\archive\deepfake_dataset\test'
    
    # Define transformations
    # For deepfake detection, preserving facial details is important
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation and test
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DeepfakeDataset(train_dir, transform=train_transform)
    val_dataset = DeepfakeDataset(val_dir, transform=val_test_transform)
    test_dataset = DeepfakeDataset(test_dir, transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Initialize model
    model = DeepfakeDetector(fine_tune=True).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Train the model
    model, history = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)
    
    # Plot training history
    plot_training_history(history)
    
    # Save the model
    torch.save(model.state_dict(), 'deepfake_detector_model.pth')
    print("Model saved to deepfake_detector_model.pth")
    
    # Evaluate the model on test set
    print("\nEvaluating model on test set...")
    results = evaluate_model(model, test_loader)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()