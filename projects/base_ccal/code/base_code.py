import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc,
    confusion_matrix
)
import cv2
import seaborn as sns
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import wandb
import random
# wandb.login(key='REDACTED_WANDB_API_KEY')  


config = {
    'data_dir': '/kaggle/input/lung-split',  
    'batch_size': 32,
    'learning_rate': 0.00001,
    'epochs': 50,
    'num_heads': 4,
    'key_dim': 16,
    'dropout': 0.5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

class LungDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = {'Bengin cases': 0, 'Malignant cases': 1, 'Normal cases': 2}
        
        self.samples = []
        split_path = os.path.join(data_dir, split)
        
        for class_name, label in self.classes.items():
            class_path = os.path.join(split_path, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.samples.append((os.path.join(class_path, img_name), label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_counts(self):
        class_counts = {0: 0, 1: 0, 2: 0}
        for _, label in self.samples:
            class_counts[label] += 1
        return class_counts

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 3:  
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:  
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            image = clahe.apply(image)

        return Image.fromarray(image)

def get_data_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0,                       # Shift, zoom, shear
                                translate=(0.1, 0.1), 
                                scale=(0.9, 1.1), 
                                shear=10, 
                                fill=0), 
        transforms.RandomHorizontalFlip(p=0.5),
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, test_transforms

def get_balanced_loader(data_dir, split='train', batch_size=32):
    train_transforms, test_transforms = get_data_transforms()
    
    if split == 'train':
        dataset = LungDataset(data_dir, split, train_transforms)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=4)
    else:
        dataset = LungDataset(data_dir, split, test_transforms)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=4)

def get_loaders(data_dir, batch_size=32):
    train_loader = get_balanced_loader(data_dir, 'train', batch_size)
    test_loader = get_balanced_loader(data_dir, 'test', batch_size)
    return train_loader, test_loader


#=================== Architecture =================
class CNN_Model1(nn.Module):
    def __init__(self,inp_chn = 3, out_chn=32):
        super(CNN_Model1,self).__init__()
        self.conv1 = nn.Conv2d(inp_chn,out_chn,kernel_size=3,padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(out_chn,out_chn*2,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(out_chn*2,out_chn*2,kernel_size=3,padding=1)

        self.conv4 = nn.Conv2d(out_chn*2,out_chn*4,kernel_size=3,padding=1)
        self.conv5= nn.Conv2d(out_chn*4,out_chn*4,kernel_size=3,padding=1)
        
        self.conv6 = nn.Conv2d(out_chn*4,out_chn*8,kernel_size=3,padding=1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self,x):
        # print(x.shape)
        x = self.relu(self.conv1(x))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = self.relu(self.conv2(x))
        # print(x.shape)
        x = self.relu(self.conv3(x))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = self.relu(self.conv4(x))
        # print(x.shape)
        x = self.relu(self.conv5(x))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = self.relu(self.conv6(x))
        # print(x.shape)
        x = self.global_max_pool(x)
        # print(x.shape)

        # print(x.view(x.shape[0],-1).shape)
        return x.view(x.shape[0],-1)  # return (B,C) = (1,256)

class CNN_Model2(nn.Module):
    def __init__(self,inp_chn = 3, out_chn=16):
        super(CNN_Model2,self).__init__()
        self.conv1 = nn.Conv2d(inp_chn,out_chn,kernel_size=3,padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(out_chn,out_chn*2,kernel_size=3,padding=1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self,x):
        # print(x.shape)
        x = self.relu(self.conv1(x))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = self.relu(self.conv2(x))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = self.global_max_pool(x)
        # print(x.shape)
        # print(x.view(x.shape[0],-1).shape)
        return x.view(x.shape[0],-1) # return (B,C) = (1,32)
    
class MainModel(nn.Module):
    def __init__(self, num_classes=3, num_heads=4, key_dim=16, num_tokens=4, dropout_p=config['dropout']):
        super(MainModel, self).__init__()
        self.cnn1 = CNN_Model1()              # (B, C)
        self.cnn2 = CNN_Model2()              # ( B, C)
        concat_dim = 256 + 32                 # 288 per paper

        # Attention width per PyTorch: embed_dim = num_heads * key_dim = 64
        embed_dim = num_heads * key_dim       # mirrors heads=4, key_dim=16 in paper
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Project 288 → (T * E), then reshape to (B, T, E) for MHA
        self.feature_projection = nn.Linear(concat_dim, self.num_heads  * embed_dim)

        # Two stacked MHA layers 
        self.mha1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.mha2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, batch_first=True)

        # MLP head mirrors: GAP1D → BN → Dense(256) → GELU → BN → Dropout → Dense(3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.batch_norm1 = nn.BatchNorm1d(embed_dim)
        self.dense1 = nn.Linear(embed_dim, 256)
        self.gelu = nn.GELU()
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout_p)
        self.ffn = nn.Linear(256, num_classes)

    def forward(self, x):
        f1 = self.cnn1(x)                                    # B × 256
        f2 = self.cnn2(x)                                    # B × 32
        concat_features = torch.cat([f1, f2], dim=1)         # B × 288

        proj = self.feature_projection(concat_features)      # B × (T*E)
        tokens = proj.view(proj.size(0), self.num_heads , self.embed_dim)  # B × T × E

        attn_out1, attn_w1 = self.mha1(tokens, tokens, tokens)             # B × T × E
        attn_out2, attn_w2 = self.mha2(attn_out1, attn_out1, attn_out1)    # B × T × E

        x = attn_out2.transpose(1, 2)                        # B × E × T
        x = self.global_avg_pool(x).squeeze(-1)              # B × E

        x = self.batch_norm1(x)
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.batch_norm2(x)
        x = self.dropout(x)
        logits = self.ffn(x)
        # Optionally return both attention maps
        return logits, (attn_w1, attn_w2)


#================= Training & Testing ==================

def train_model(model, train_loader, val_loader, num_epochs=1, learning_rate=0.001, device='cuda'):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Track best validation accuracy
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for data, targets in tqdm(train_loader, desc=f'Training Epoch {epoch+1}'):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs, _ = model(data)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Check if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            # Save the best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
                'train_acc': train_acc,
                'train_loss': avg_train_loss
            }, 'best_model.pth')
            print(f'✓ New best model saved! Val Acc: {val_acc:.2f}%')

        # Simple epoch output
        print(f'Epoch {epoch+1:2d}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Basic wandb logging
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc
        })

    # Print best model info
    print(f'\nTraining completed!')
    print(f'Best model saved at epoch {best_epoch} with validation accuracy: {best_val_acc:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def load_best_model(model, device='cuda'):
    """Load the saved best model"""
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f'Loaded best model from epoch {checkpoint["epoch"]}')
    print(f'Best validation accuracy: {checkpoint["val_acc"]:.2f}%')
    
    return model

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc='Testing'):
            data, targets = data.to(device), targets.to(device)
            outputs, _ = model(data)
            probs = F.softmax(outputs, dim=1)
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    auc = roc_auc_score(all_targets, np.array(all_probs), multi_class='ovr', average='weighted')

    # Simple results display
    print(f'\nTest Results:')
    print(f'Accuracy:  {accuracy*100:.2f}%')
    print(f'Precision: {precision*100:.2f}%')
    print(f'Recall:    {recall*100:.2f}%')
    print(f'F1-Score:  {f1*100:.2f}%')
    print(f'AUC:       {auc*100:.2f}%')

    # Log to wandb
    wandb.log({
        "test_accuracy": accuracy * 100,
        "test_precision": precision * 100,
        "test_recall": recall * 100,
        "test_f1": f1 * 100,
        "test_auc": auc * 100
    })

    # Simple confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.show()

    return accuracy, precision, recall, f1, auc

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(train_accuracies, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(val_accuracies, 'r-', label='Val Acc', linewidth=2)
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mark best validation accuracy
    best_val_idx = np.argmax(val_accuracies)
    ax2.scatter(best_val_idx, val_accuracies[best_val_idx], 
                color='red', s=100, marker='*', zorder=5, 
                label=f'Best: {val_accuracies[best_val_idx]:.2f}%')
    ax2.legend()

    plt.tight_layout()
    wandb.log({"training_curves": wandb.Image(plt)})
    plt.show()



train_loader, test_loader = get_loaders('/kaggle/input/iiq-othnccd-lung-cancer-split', batch_size=config['batch_size'])

# Initialize Weights & Biases
wandb.init(
    project="base_ccal",
    name="base_model_training",
    config=config
)




# Create model
print("Creating CCAL model...")
model = MainModel(
    num_classes=3,
    num_heads=config['num_heads'],
    key_dim=config['key_dim'],
    dropout_p = config['dropout']
)

print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Log model architecture
wandb.watch(model, log="all", log_freq=100)

# Log model info to W&B
wandb.log({
    "total_parameters": sum(p.numel() for p in model.parameters()),
    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),

})
wandb.login()  


# Train model
print("Starting training...")
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, test_loader, 
    num_epochs=config['epochs'], 
    learning_rate=config['learning_rate'],
    device=config['device']
)

# # Plot training history
plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)

# Evaluate on test set
print("Evaluating model...")
accuracy, precision, recall, f1, auc = evaluate_model(model, test_loader, device=config['device'])

# Save model
torch.save(model.state_dict(), 'base_ccalv2.1.pth')
print("Model saved as 'base_ccal.pth'")

# Save model artifact to W&B
# artifact = wandb.Artifact('base_ccal', type='model')
# artifact.add_file('base_ccal.pth')
# wandb.log_artifact(artifact)

# Finish W&B run
wandb.finish(), 