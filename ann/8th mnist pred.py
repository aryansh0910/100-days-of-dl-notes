# ============================================================
# HANDWRITTEN DIGIT CLASSIFICATION USING ANN
# MNIST Dataset — Complete End to End Project
# ============================================================




# ============================================================
# WHAT IS THIS PROJECT?
# ============================================================

# MNIST = Modified National Institute of Standards and Technology
# It is the "Hello World" of deep learning 🌍
# Contains 70,000 handwritten digit images (0 to 9)
# Each image = 28×28 pixels = 784 pixel values
# Goal = look at image → predict which digit it is (0-9)

# ANALOGY:
# Imagine you're a postal worker 📮
# You receive thousands of handwritten envelopes
# You need to read the PIN code digits written on them
# Your ANN = an automated digit reader
# Input  = pixel values of the handwritten digit image
# Output = which digit is it? (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

# THIS IS A MULTI-CLASS CLASSIFICATION PROBLEM
# 10 possible output classes (one per digit)
# We use Softmax + CrossEntropyLoss (not Sigmoid + BCE)




# ============================================================
# MNIST DATASET VISUAL
# ============================================================

#
#   Each image looks like this (28×28 pixels):
#
#   ............................
#   ............................
#   ............###.............
#   ..........#####.............
#   .........####...............
#   .........###................
#   .........###................
#   ..........####..............
#   ............#####...........
#   ..............####..........
#   ...............###..........
#   ...............###..........
#   ..............####..........
#   .........########...........
#   ............................
#
#   This is the digit "5"
#   Each "." = pixel value near 0 (black background)
#   Each "#" = pixel value near 255 (white digit)
#
#   We FLATTEN this 28×28 image into 784 values
#   and feed it into our ANN as input




# ============================================================
# STEP 0: IMPORTS
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Output: Using device: cpu (or cuda)




# ============================================================
# STEP 1: LOAD AND PREPARE MNIST DATA
# ============================================================

# ANALOGY:
# torchvision.datasets is like an Amazon delivery 📦
# You just tell it what you want (MNIST)
# It downloads, unpacks, and delivers it ready to use

# Define transforms
# Transform = preprocessing applied to every image automatically
transform = transforms.Compose([
    transforms.ToTensor(),              # PIL image → tensor (0-255 → 0.0-1.0)
    transforms.Normalize((0.1307,),     # subtract mean
                         (0.3081,))     # divide by std
    # These exact values (0.1307, 0.3081) are the
    # pre-computed mean and std of the entire MNIST dataset
])

# Download and load datasets
train_dataset = datasets.MNIST(
    root='./data',        # where to save
    train=True,           # training set (60,000 images)
    download=True,        # download if not present
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,          # test set (10,000 images)
    download=True,
    transform=transform
)

print(f"\nTraining samples : {len(train_dataset)}")  # Output: 60000
print(f"Testing samples  : {len(test_dataset)}")     # Output: 10000
print(f"Image shape      : {train_dataset[0][0].shape}")
# Output: torch.Size([1, 28, 28])
# 1 = grayscale channel, 28×28 = image size




# ============================================================
# STEP 2: CREATE DATA LOADERS
# ============================================================

# ANALOGY:
# DataLoader = a conveyor belt in a factory 🏭
# Instead of feeding ALL 60,000 images at once (too heavy!)
# it feeds them in small BATCHES of 64 at a time
# Much more memory efficient and trains faster!

BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True          # shuffle every epoch so model doesn't memorize order
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False         # no need to shuffle test data
)

print(f"\nBatch size       : {BATCH_SIZE}")
print(f"Training batches : {len(train_loader)}")   # Output: 938 (60000/64)
print(f"Testing batches  : {len(test_loader)}")    # Output: 157 (10000/64)

# Look at one batch
images, labels = next(iter(train_loader))
print(f"\nOne batch images shape : {images.shape}")
# Output: torch.Size([64, 1, 28, 28])
# 64 images, 1 channel, 28×28 pixels

print(f"One batch labels shape : {labels.shape}")
# Output: torch.Size([64])
# 64 labels (one per image)




# ============================================================
# STEP 3: VISUALIZE SOME SAMPLES
# ============================================================

# Always look at your data before training!
# ANALOGY: before cooking, check your ingredients 🥘

def show_samples(loader, n=10):
    images, labels = next(iter(loader))
    plt.figure(figsize=(15, 2))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.suptitle('Sample MNIST Images', fontsize=14)
    plt.tight_layout()
    plt.show()

show_samples(train_loader)
# Output: shows 10 handwritten digit images with their labels




# ============================================================
# STEP 4: BUILD THE ANN MODEL
# ============================================================

# INPUT LAYER:
# Image = 28×28 = 784 pixels
# We FLATTEN it → 784 input neurons
#
# HIDDEN LAYERS:
# 784 → 512 → 256 → 128
# (gradually reducing, like a funnel)
#
# OUTPUT LAYER:
# 128 → 10 (one neuron per digit class 0-9)
# Use Softmax → gives probability for each digit

#
# ARCHITECTURE DIAGRAM:
#
# [784 inputs]──>[512 neurons]──>[256 neurons]──>[128 neurons]──>[10 outputs]
#   28×28px       ReLU+BN+DO      ReLU+BN+DO      ReLU+BN+DO     Softmax
#
# ŷ = [0.02, 0.01, 0.03, 0.85, 0.01, 0.02, 0.01, 0.03, 0.01, 0.01]
#       0      1      2      3     4      5      6      7      8      9
#                            ↑
#                       highest prob → predicted class = 3

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()

        self.flatten = nn.Flatten()   # 28×28 → 784

        self.network = nn.Sequential(
            # Layer 1: 784 → 512
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2: 512 → 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 3: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output: 128 → 10
            nn.Linear(128, 10)
            # NOTE: NO Softmax here!
            # CrossEntropyLoss in PyTorch applies Softmax internally
            # Adding it manually would cause errors
        )

    def forward(self, x):
        x = self.flatten(x)     # (batch, 1, 28, 28) → (batch, 784)
        x = self.network(x)     # (batch, 784) → (batch, 10)
        return x

model = MNISTClassifier().to(device)
print(model)
# Output: shows full architecture

# Count parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
# Output: Total parameters: ~567,000




# ============================================================
# STEP 5: LOSS AND OPTIMIZER
# ============================================================

# CrossEntropyLoss = for multi-class classification (10 classes)
# Adam = best general purpose optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,    # reduce LR every 5 epochs
    gamma=0.5       # multiply LR by 0.5 each time
)




# ============================================================
# STEP 6: TRAINING LOOP
# ============================================================

# ANALOGY:
# Training = practicing for an exam 📚
# Each epoch = one full study session
# Each batch  = one set of practice questions
# Loss        = how many questions you got wrong
# Optimizer   = correcting your mistakes after each set

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss    = 0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)              # shape: (64, 10)
        loss    = criterion(outputs, labels) # compare with true labels

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss    += loss.item()
        predicted      = outputs.argmax(dim=1)   # pick class with highest score
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss    = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss    += loss.item()
            predicted      = outputs.argmax(dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy


# ── MAIN TRAINING LOOP ─────────────────────────────────────
EPOCHS = 15

train_losses, val_losses     = [], []
train_accs,   val_accs       = [], []

print("Starting training...\n")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc = evaluate(
        model, test_loader, criterion, device
    )
    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1:2d}/{EPOCHS}] | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    # Output:
    # Epoch [ 1/15] | Train Loss: 0.2134 | Train Acc: 93.45% | Val Loss: 0.1023 | Val Acc: 96.87%
    # Epoch [ 5/15] | Train Loss: 0.0987 | Train Acc: 97.12% | Val Loss: 0.0812 | Val Acc: 97.54%
    # Epoch [10/15] | Train Loss: 0.0756 | Train Acc: 97.89% | Val Loss: 0.0734 | Val Acc: 97.91%
    # Epoch [15/15] | Train Loss: 0.0634 | Train Acc: 98.12% | Val Loss: 0.0698 | Val Acc: 98.05%




# ============================================================
# STEP 7: PLOT TRAINING CURVES
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(train_losses, label='Train Loss', color='blue',   marker='o')
axes[0].plot(val_losses,   label='Val Loss',   color='orange', marker='o')
axes[0].set_title('Loss Curve')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy curve
axes[1].plot(train_accs, label='Train Accuracy', color='blue',   marker='o')
axes[1].plot(val_accs,   label='Val Accuracy',   color='orange', marker='o')
axes[1].set_title('Accuracy Curve')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy %')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
# Output: two smooth curves
# Loss going DOWN, Accuracy going UP = good training!




# ============================================================
# STEP 8: EVALUATE ON TEST SET
# ============================================================

# Get all predictions
all_preds  = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images  = images.to(device)
        outputs = model(images)
        preds   = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# Final accuracy
final_acc = (all_preds == all_labels).mean() * 100
print(f"\n✅ Final Test Accuracy: {final_acc:.2f}%")
# Output: ✅ Final Test Accuracy: ~98%

# Classification report
print("\n📊 Classification Report:")
print(classification_report(
    all_labels, all_preds,
    target_names=[str(i) for i in range(10)]
))
# Output:
#         precision  recall  f1-score
#    0       0.99     0.99     0.99
#    1       0.99     0.99     0.99
#    2       0.98     0.98     0.98
#    ...
#    9       0.98     0.97     0.98




# ============================================================
# STEP 9: CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10),
            yticklabels=range(10))
plt.title('Confusion Matrix — MNIST Digit Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
# Output: 10×10 heatmap
# Diagonal = correct predictions (should be very high)
# Off-diagonal = mistakes (e.g., model confused 4 with 9)




# ============================================================
# STEP 10: VISUALIZE PREDICTIONS
# ============================================================

# See what the model actually predicts on real images
# ANALOGY: open book test — let's see if the model passes! 📖

def show_predictions(model, loader, device, n=12):
    model.eval()
    images, labels = next(iter(loader))
    images_dev     = images.to(device)

    with torch.no_grad():
        outputs = model(images_dev)
        probs   = torch.softmax(outputs, dim=1)   # convert to probabilities
        preds   = outputs.argmax(dim=1)

    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(2, 6, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        pred  = preds[i].item()
        true  = labels[i].item()
        conf  = probs[i][pred].item() * 100
        color = 'green' if pred == true else 'red'
        plt.title(f"P:{pred} T:{true}\n{conf:.1f}%", color=color, fontsize=9)
        plt.axis('off')
    plt.suptitle('Green=Correct  Red=Wrong', fontsize=12)
    plt.tight_layout()
    plt.show()

show_predictions(model, test_loader, device)
# Output: grid of 12 images
# Green title = correct prediction ✅
# Red title   = wrong prediction ❌
# confidence % shown for each




# ============================================================
# STEP 11: PREDICT A SINGLE IMAGE
# ============================================================

def predict_single(model, dataset, idx, device):
    image, true_label = dataset[idx]

    # Add batch dimension
    image_input = image.unsqueeze(0).to(device)  # (1, 28, 28) → (1, 1, 28, 28)

    model.eval()
    with torch.no_grad():
        output = model(image_input)
        probs  = torch.softmax(output, dim=1)
        pred   = output.argmax(dim=1).item()
        conf   = probs[0][pred].item() * 100

    # Show image
    plt.figure(figsize=(4, 4))
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Predicted: {pred} ({conf:.1f}% confident)\nTrue Label: {true_label}",
              fontsize=12)
    plt.axis('off')
    plt.show()

    return pred, true_label, conf

pred, true, conf = predict_single(model, test_dataset, idx=7, device=device)
print(f"\nPredicted: {pred} | True: {true} | Confidence: {conf:.1f}%")
# Output:
# Predicted: 1 | True: 1 | Confidence: 99.8%




# ============================================================
# STEP 12: SAVE THE MODEL
# ============================================================

torch.save(model.state_dict(), 'mnist_ann_model.pth')
print("\n✅ Model saved as mnist_ann_model.pth")

# To load later:
# loaded_model = MNISTClassifier().to(device)
# loaded_model.load_state_dict(torch.load('mnist_ann_model.pth'))
# loaded_model.eval()




# ============================================================
# CHURN vs MNIST — KEY DIFFERENCES
# ============================================================

#
#  FEATURE           CHURN PROJECT          MNIST PROJECT
#  ────────────────────────────────────────────────────────
#  Problem Type      Binary Classification  Multi Classification
#  Output Classes    2 (churn/no churn)     10 (digits 0-9)
#  Loss Function     BCELoss                CrossEntropyLoss
#  Output Activation Sigmoid                Softmax (internal)
#  Input Type        Tabular (numbers)      Image (pixels)
#  Input Size        ~30 features           784 (28×28 flattened)
#  DataLoader        Manual tensor          torchvision datasets
#  Evaluation        Recall focus           Accuracy + confusion
#




# ============================================================
# KEY POINTS TO REMEMBER:
# ============================================================

# 1.  MNIST = 70,000 images, 10 classes, 28×28 pixels each
# 2.  Flatten image 28×28 → 784 before feeding into ANN
# 3.  Use CrossEntropyLoss for multi-class (NOT BCELoss)
# 4.  Do NOT add Softmax manually — CrossEntropyLoss includes it
# 5.  outputs.argmax(dim=1) → gives predicted class
# 6.  DataLoader handles batching and shuffling automatically
# 7.  transforms.Normalize() → zero mean, unit std per channel
# 8.  model.train() during training, model.eval() during testing
# 9.  torch.no_grad() during evaluation → saves memory
# 10. ~98% accuracy is achievable with ANN on MNIST
# 11. CNN will do even better (99%+) — we learn that next!
# 12. Confusion matrix diagonal = correct, off-diagonal = mistakes

# ============================================================
# NEXT TOPIC: Backpropagation in depth
# (the math of how the model learned to get 98% accuracy)
# ============================================================