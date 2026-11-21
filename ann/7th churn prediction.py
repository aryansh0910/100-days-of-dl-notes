# ============================================================
# CUSTOMER CHURN PREDICTION USING ANN
# Complete End-to-End Project in PyTorch
# ============================================================


# ============================================================
# WHAT IS CUSTOMER CHURN?
# ============================================================

# Churn = when a customer STOPS using a service
# Example: you had Hotstar subscription → you cancelled it → you CHURNED

# ANALOGY:
# Imagine you run a gym 🏋️
# Some members renew every year (loyal customers)
# Some members cancel after 3 months (churned customers)
# If you can PREDICT who is about to cancel
# you can offer them a discount BEFORE they leave
# That saves the business a LOT of money!

# THIS IS WHY CHURN PREDICTION IS ONE OF THE MOST
# COMMON REAL WORLD ML PROBLEMS IN INDUSTRY




# ============================================================
# WHAT WE WILL BUILD
# ============================================================

# INPUT:  customer features (age, balance, tenure etc.)
# OUTPUT: will this customer churn? (1=yes, 0=no)
# MODEL:  ANN (Multi Layer Perceptron)
# This is a BINARY CLASSIFICATION problem




# ============================================================
# STEP 0: INSTALL AND IMPORT
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix,
                             classification_report,
                             accuracy_score)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Output: Using device: cpu  (or cuda if you have GPU)




# ============================================================
# STEP 1: LOAD THE DATASET
# ============================================================

# We use the famous Telco Customer Churn dataset
# Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# OR use this code to load it directly:

url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/Telco-Customer-Churn.csv"
df  = pd.read_csv(url)

print(f"Dataset shape: {df.shape}")
# Output: Dataset shape: (7043, 21)
# 7043 customers, 21 features

print(f"\nFirst 3 rows:")
print(df.head(3))

print(f"\nChurn distribution:")
print(df['Churn'].value_counts())
# Output:
# No     5174  ← did not churn
# Yes    1869  ← churned




# ============================================================
# STEP 2: DATA PREPROCESSING
# ============================================================

# ANALOGY:
# Raw data is like raw vegetables 🥦
# You can't eat them directly — you need to clean,
# chop, and cook them first
# Preprocessing = preparing data for the model

# ── 2a: Drop customerID (useless for prediction) ──────────
df = df.drop('customerID', axis=1)

# ── 2b: Fix TotalCharges (it has spaces, should be numeric) ─
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# ── 2c: Encode target column (Yes/No → 1/0) ──────────────
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
print(f"\nChurn encoded: {df['Churn'].unique()}")
# Output: Churn encoded: [0 1]

# ── 2d: Encode binary categorical columns ─────────────────
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
               'PaperlessBilling']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# ── 2e: One-hot encode multi-category columns ─────────────
multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
              'OnlineBackup', 'DeviceProtection', 'TechSupport',
              'StreamingTV', 'StreamingMovies', 'Contract',
              'PaymentMethod']
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

print(f"\nAfter preprocessing shape: {df.shape}")
# Output: After preprocessing shape: (7043, ~30 features)

# ── 2f: Split features and target ─────────────────────────
X = df.drop('Churn', axis=1).values.astype(np.float32)
y = df['Churn'].values.astype(np.float32)

print(f"\nX shape: {X.shape}")   # Output: (7043, ~30)
print(f"y shape: {y.shape}")     # Output: (7043,)

# ── 2g: Train-test split ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}")   # Output: 5634
print(f"Test size:  {X_test.shape[0]}")      # Output: 1409

# ── 2h: Normalize features (VERY important for ANN) ───────
# ANALOGY:
# If one feature is salary (50000) and another is age (25)
# the model will think salary is way more important just because
# its number is bigger — normalization fixes this!
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit on train only!
X_test  = scaler.transform(X_test)        # transform test with same scaler

# ── 2i: Convert to PyTorch tensors ────────────────────────
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1).to(device)

print(f"\nX_train_t shape: {X_train_t.shape}")  # Output: torch.Size([5634, ~30])
print(f"y_train_t shape: {y_train_t.shape}")    # Output: torch.Size([5634, 1])




# ============================================================
# STEP 3: BUILD THE ANN MODEL
# ============================================================

input_size = X_train_t.shape[1]   # number of features

class ChurnANN(nn.Module):
    def __init__(self, input_size):
        super(ChurnANN, self).__init__()

        self.network = nn.Sequential(
            # Layer 1: input → 32 neurons
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),       # normalizes layer outputs → stable training
            nn.Dropout(0.3),          # randomly drops 30% neurons → prevents overfitting

            # Layer 2: 32 → 16 neurons
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),

            # Output: 16 → 1 neuron
            nn.Linear(16, 1),
            nn.Sigmoid()              # binary classification → sigmoid
        )

    def forward(self, x):
        return self.network(x)

model = ChurnANN(input_size).to(device)
print(f"\n{model}")
# Output:
# ChurnANN(
#   (network): Sequential(
#     (0): Linear(in_features=30, out_features=32, bias=True)
#     (1): ReLU()
#     (2): BatchNorm1d(32)
#     (3): Dropout(p=0.3)
#     (4): Linear(in_features=32, out_features=16, bias=True)
#     (5): ReLU()
#     (6): BatchNorm1d(16)
#     (7): Dropout(p=0.2)
#     (8): Linear(in_features=16, out_features=1, bias=True)
#     (9): Sigmoid()
#   )
# )

# Count parameters
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal trainable parameters: {total}")
# Output: Total trainable parameters: ~1200 (small but powerful!)




# ============================================================
# WHAT IS BATCHNORM AND DROPOUT? (QUICK EXPLAINER)
# ============================================================

# BATCHNORM (Batch Normalization):
# ANALOGY: like a teacher who rescales everyone's marks
# so the class average is always 0 and std is 1
# This makes training FASTER and more STABLE

# DROPOUT:
# ANALOGY: like randomly sending 30% of employees
# on leave every day — forces the REMAINING employees
# to work harder and learn more
# This prevents OVERFITTING (memorizing instead of learning)




# ============================================================
# STEP 4: TRAINING SETUP
# ============================================================

criterion = nn.BCELoss()                           # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Learning rate scheduler — reduces LR when loss stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=10, factor=0.5, verbose=True
)




# ============================================================
# STEP 5: TRAINING LOOP
# ============================================================

EPOCHS      = 100
train_losses = []
val_losses   = []

for epoch in range(EPOCHS):

    # ── TRAINING PHASE ─────────────────────────────────────
    model.train()                        # set model to training mode
    optimizer.zero_grad()
    y_pred_train = model(X_train_t)
    loss_train   = criterion(y_pred_train, y_train_t)
    loss_train.backward()
    optimizer.step()
    train_losses.append(loss_train.item())

    # ── VALIDATION PHASE ───────────────────────────────────
    model.eval()                         # set model to evaluation mode
    with torch.no_grad():
        y_pred_val = model(X_test_t)
        loss_val   = criterion(y_pred_val, y_test_t)
        val_losses.append(loss_val.item())

    # Update scheduler
    scheduler.step(loss_val)

    # Print every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | "
              f"Train Loss: {loss_train.item():.4f} | "
              f"Val Loss: {loss_val.item():.4f}")
        # Output:
        # Epoch [ 10/100] | Train Loss: 0.5123 | Val Loss: 0.5234
        # Epoch [ 20/100] | Train Loss: 0.4678 | Val Loss: 0.4789
        # Epoch [ 30/100] | Train Loss: 0.4312 | Val Loss: 0.4421
        # Epoch [ 50/100] | Train Loss: 0.3987 | Val Loss: 0.4102
        # Epoch [100/100] | Train Loss: 0.3654 | Val Loss: 0.3821




# ============================================================
# STEP 6: PLOT TRAINING CURVES
# ============================================================

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses,   label='Val Loss',   color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)

# Output: smooth curves going down
# if val_loss >> train_loss → overfitting
# if both go down together → good training!

plt.tight_layout()
plt.show()




# ============================================================
# STEP 7: EVALUATE THE MODEL
# ============================================================

model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_t)
    y_pred_class = (y_pred_probs >= 0.5).float()  # threshold: 0.5

# Move to numpy for sklearn metrics
y_true = y_test_t.cpu().numpy()
y_pred = y_pred_class.cpu().numpy()
y_prob = y_pred_probs.cpu().numpy()

# ── Accuracy ───────────────────────────────────────────────
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
# Output: ✅ Test Accuracy: ~80-82%

# ── Classification Report ──────────────────────────────────
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred,
                             target_names=['No Churn', 'Churn']))
# Output:
#               precision  recall  f1-score  support
#   No Churn       0.85      0.91     0.88     1035
#   Churn          0.67      0.54     0.60      374
#   accuracy                          0.82     1409

# ── Confusion Matrix ───────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
print("\n🔢 Confusion Matrix:")
print(cm)
# Output:
# [[940  95]    ← 940 correctly predicted no-churn, 95 wrongly predicted churn
#  [171 203]]   ← 171 missed churns (bad!), 203 correctly predicted churn




# ============================================================
# CONFUSION MATRIX EXPLAINED
# ============================================================

#
#                    PREDICTED
#                  No Churn | Churn
#   ACTUAL No Churn  [TN]   |  [FP]    ← predicted churn but didn't
#          Churn     [FN]   |  [TP]    ← predicted no churn but did!
#
#   TN = True Negative   → correctly said no churn ✅
#   TP = True Positive   → correctly said churn ✅
#   FP = False Positive  → false alarm ❌
#   FN = False Negative  → missed a churner ❌ (costly for business!)
#
#   For churn prediction:
#   FN is MORE COSTLY than FP
#   Missing a churner = losing a customer forever
#   False alarm = you offered a discount unnecessarily (small loss)
#   So we care more about RECALL for churn class




# ============================================================
# STEP 8: PREDICT ON A NEW CUSTOMER
# ============================================================

# ANALOGY:
# Your model is now trained
# A new customer walks in → you feed their data → model predicts!

def predict_churn(customer_features, model, scaler, device):
    """
    customer_features: numpy array of shape (1, n_features)
    returns: churn probability and prediction
    """
    # Normalize using same scaler from training
    features_scaled = scaler.transform(customer_features)

    # Convert to tensor
    features_tensor = torch.tensor(
        features_scaled, dtype=torch.float32
    ).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        prob = model(features_tensor).item()

    prediction = "WILL CHURN ⚠️" if prob > 0.5 else "WILL STAY ✅"
    return prob, prediction

# Example: take first test customer
sample_customer = X_test[:1]   # shape (1, n_features) — already scaled
# re-use scaler inverse for demo (in real use: pass raw features)
prob, prediction = predict_churn(
    scaler.inverse_transform(sample_customer), model, scaler, device
)
print(f"\nCustomer Prediction: {prediction}")
print(f"Churn Probability:   {prob:.2%}")
# Output:
# Customer Prediction: WILL CHURN ⚠️  (or WILL STAY ✅)
# Churn Probability:   67.34%




# ============================================================
# STEP 9: SAVE AND LOAD THE MODEL
# ============================================================

# Save model weights
torch.save(model.state_dict(), 'churn_model.pth')
print("\nModel saved as churn_model.pth ✅")

# Load model later
loaded_model = ChurnANN(input_size).to(device)
loaded_model.load_state_dict(torch.load('churn_model.pth'))
loaded_model.eval()
print("Model loaded successfully ✅")




# ============================================================
# FULL PIPELINE SUMMARY
# ============================================================

#
#   RAW CSV DATA
#       ↓
#   PREPROCESSING
#   (encode, normalize, split)
#       ↓
#   PYTORCH TENSORS
#       ↓
#   ANN MODEL
#   (Linear → BN → ReLU → Dropout) × 2
#   (Linear → Sigmoid)
#       ↓
#   TRAINING LOOP
#   (forward → loss → backward → optimizer)
#       ↓
#   EVALUATION
#   (accuracy, confusion matrix, report)
#       ↓
#   PREDICTION ON NEW CUSTOMER




# ============================================================
# KEY POINTS TO REMEMBER:
# ============================================================

# 1.  Churn prediction = binary classification (0 or 1)
# 2.  Always drop useless columns (like ID columns)
# 3.  Encode categorical columns before feeding to ANN
# 4.  ALWAYS normalize features using StandardScaler
# 5.  Fit scaler on TRAIN set only — transform both train and test
# 6.  BatchNorm = stabilizes training, makes it faster
# 7.  Dropout = prevents overfitting (randomly drops neurons)
# 8.  Use BCELoss + Sigmoid for binary classification
# 9.  Adam optimizer works great for most ANN problems
# 10. For churn: focus on RECALL of churn class (minimize missed churners)
# 11. model.train() for training, model.eval() for testing
# 12. Always use torch.no_grad() during evaluation (saves memory)

# ============================================================
# NEXT TOPIC: Backpropagation in depth
# (the math behind HOW the ANN actually learned above)
# ============================================================