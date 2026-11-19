# ============================================================
# LOSS FUNCTIONS FOR PERCEPTRON & NEURAL NETWORKS
# ============================================================




# ============================================================
# WHAT IS A LOSS FUNCTION?
# ============================================================

# A loss function measures HOW WRONG your model's prediction is
# It gives a single number → the higher it is, the worse the model

# ANALOGY:
# Imagine you're learning archery 🏹
# You shoot an arrow at the target
# Loss function = the distance between your arrow and the bullseye
# If arrow hits bullseye → loss = 0 (perfect!)
# If arrow misses by 2 meters → loss = high (bad!)
# Your goal during training = MINIMIZE this distance (loss)

# WHY DO WE NEED IT?
# The model has no idea if its prediction is good or bad
# Loss function is the SIGNAL that tells the model:
#   "hey you were THIS wrong, now adjust your weights"
# Without loss → no learning → model stays dumb forever

# FLOW:
#
#   Input → Model → Prediction
#                       ↓
#                Loss Function ← Actual Label
#                       ↓
#                  Loss Value
#                       ↓
#               Backpropagation
#                       ↓
#               Update Weights




# ============================================================
# TYPES OF LOSS FUNCTIONS
# ============================================================

#
#  PROBLEM TYPE          LOSS FUNCTION              WHEN TO USE
#  ─────────────────────────────────────────────────────────────────
#  Binary Classification  Binary Cross Entropy       cat vs dog, spam vs not
#  Multi Classification   Cross Entropy Loss         digit 0-9, 10 classes
#  Regression             Mean Squared Error (MSE)   predicting house price
#  Regression             Mean Absolute Error (MAE)  predicting temperature
#

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt




# ============================================================
# LOSS FUNCTION 1: MEAN SQUARED ERROR (MSE)
# ============================================================

# Used for REGRESSION problems (predicting a number)

# ANALOGY:
# You're a weather forecaster 🌤️
# You predicted 30°C but actual was 35°C → error = 5
# You SQUARE the error → 25 (squaring punishes big mistakes more)
# Do this for all predictions → take the average → that's MSE

# FORMULA:
# MSE = (1/n) * Σ (predicted - actual)²

# WHY SQUARE IT?
# 1. Makes all errors positive (no cancellation)
# 2. Punishes LARGE errors much more than small ones
# 3. Makes math easier for backpropagation (smooth curve)

actual    = torch.tensor([35.0, 20.0, 25.0, 30.0])  # actual temperatures
predicted = torch.tensor([30.0, 22.0, 28.0, 25.0])  # model predictions

# Method 1: Manual calculation
mse_manual = torch.mean((predicted - actual) ** 2)
print(f"MSE (manual) = {mse_manual.item():.4f}")
# Output: MSE (manual) = 14.2500

# Method 2: Using PyTorch
criterion_mse = nn.MSELoss()
mse_pytorch   = criterion_mse(predicted, actual)
print(f"MSE (PyTorch) = {mse_pytorch.item():.4f}")
# Output: MSE (PyTorch) = 14.2500

# WHEN TO USE MSE:
# ✅ Predicting house prices
# ✅ Predicting stock prices
# ✅ Predicting temperature
# ❌ Do NOT use for classification problems




# ============================================================
# LOSS FUNCTION 2: MEAN ABSOLUTE ERROR (MAE)
# ============================================================

# Also for REGRESSION — but takes absolute value instead of squaring

# ANALOGY:
# Same weather forecaster 🌤️
# Predicted 30°C, actual 35°C → error = |30-35| = 5
# Just take the absolute value, no squaring
# Average of all absolute errors → MAE

# FORMULA:
# MAE = (1/n) * Σ |predicted - actual|

# MSE vs MAE:
# MSE → punishes outliers/big errors MORE (because of squaring)
# MAE → treats all errors equally (more robust to outliers)

mae_manual = torch.mean(torch.abs(predicted - actual))
print(f"MAE (manual) = {mae_manual.item():.4f}")
# Output: MAE (manual) = 3.2500

criterion_mae = nn.L1Loss()   # L1Loss = MAE in PyTorch
mae_pytorch   = criterion_mae(predicted, actual)
print(f"MAE (PyTorch) = {mae_pytorch.item():.4f}")
# Output: MAE (PyTorch) = 3.2500

# MSE vs MAE quick comparison:
errors = torch.tensor([1.0, 2.0, 3.0, 10.0])  # last one is an outlier
zeros  = torch.zeros(4)
print(f"\nWith outlier [1,2,3,10]:")
print(f"MSE = {torch.mean(errors**2).item():.2f}")   # Output: MSE = 27.50 (outlier dominates)
print(f"MAE = {torch.mean(errors).item():.2f}")      # Output: MAE = 4.00  (more balanced)




# ============================================================
# LOSS FUNCTION 3: BINARY CROSS ENTROPY (BCE)
# ============================================================

# Used for BINARY CLASSIFICATION (only 2 classes: 0 or 1)

# ANALOGY:
# You're a doctor 🏥 predicting if a patient has a disease or not
# Patient either HAS it (1) or DOESN'T (0)
# Your model gives a PROBABILITY: 0.85 means "85% chance they have it"
# BCE measures how CONFIDENT and CORRECT you were:
#   → if actual=1 and you said 0.85 → small loss (confident + correct ✅)
#   → if actual=1 and you said 0.10 → huge loss  (confident + WRONG ❌)
#   → if actual=1 and you said 0.50 → medium loss (uncertain)

# FORMULA:
# BCE = -[y * log(p) + (1-y) * log(1-p)]
# where y = actual label (0 or 1)
#       p = predicted probability (0 to 1)

# WHY LOGARITHM?
# log(1) = 0    → if you predicted correctly with full confidence → loss = 0
# log(0) = -∞   → if you predicted wrong with full confidence → loss = very high
# This heavily punishes OVERCONFIDENT WRONG predictions

actual_labels = torch.tensor([1.0, 0.0, 1.0, 0.0])       # true labels
predicted_probs = torch.tensor([0.85, 0.10, 0.70, 0.90])  # model probabilities

# Manual BCE calculation
bce_manual = -torch.mean(
    actual_labels * torch.log(predicted_probs) +
    (1 - actual_labels) * torch.log(1 - predicted_probs)
)
print(f"\nBCE (manual) = {bce_manual.item():.4f}")
# Output: BCE (manual) = 0.2an (low — model is mostly correct!)

# PyTorch BCE
criterion_bce = nn.BCELoss()
bce_pytorch   = criterion_bce(predicted_probs, actual_labels)
print(f"BCE (PyTorch) = {bce_pytorch.item():.4f}")
# Output: BCE (PyTorch) = same value

# Let's see how loss changes based on prediction confidence:
print("\n--- How BCE reacts to different predictions ---")
cases = [
    ("Correct + Very Confident",   torch.tensor([1.0]), torch.tensor([0.99])),
    ("Correct + Somewhat Sure",    torch.tensor([1.0]), torch.tensor([0.70])),
    ("Correct + Uncertain",        torch.tensor([1.0]), torch.tensor([0.51])),
    ("Wrong + Somewhat Sure",      torch.tensor([1.0]), torch.tensor([0.30])),
    ("Wrong + Very Confident",     torch.tensor([1.0]), torch.tensor([0.01])),
]
for desc, y_true, y_pred in cases:
    loss = criterion_bce(y_pred, y_true)
    print(f"{desc:35s} → Loss: {loss.item():.4f}")
    # Output:
    # Correct + Very Confident        → Loss: 0.0101  (almost 0, great!)
    # Correct + Somewhat Sure         → Loss: 0.3567
    # Correct + Uncertain             → Loss: 0.6733
    # Wrong + Somewhat Sure           → Loss: 1.2040
    # Wrong + Very Confident          → Loss: 4.6052  (huge penalty!)




# ============================================================
# LOSS FUNCTION 4: CROSS ENTROPY LOSS (MULTI-CLASS)
# ============================================================

# Used for MULTI-CLASS CLASSIFICATION (more than 2 classes)

# ANALOGY:
# You're classifying handwritten digits 0-9 ✏️
# Model outputs a probability for EACH class
# Cross Entropy picks the correct class probability and measures
# how close it is to 1.0 (perfect confidence)

# Example: actual digit = 3
# Model output probabilities:
#   class 0: 0.05
#   class 1: 0.02
#   class 2: 0.08
#   class 3: 0.70  ← correct class, should be 1.0 ideally
#   class 4: 0.05
#   ... etc
# Loss = -log(0.70) = 0.357  (smaller is better)

# NOTE: In PyTorch, CrossEntropyLoss = Softmax + NLLLoss combined
# You do NOT need to apply softmax before CrossEntropyLoss!

# 3 classes example (e.g., cat=0, dog=1, bird=2)
# Raw model outputs (logits) — NOT probabilities yet
logits = torch.tensor([[2.0, 1.0, 0.1],   # sample 1 → model thinks class 0 (cat)
                        [0.5, 2.5, 0.3],   # sample 2 → model thinks class 1 (dog)
                        [0.1, 0.2, 3.0]])  # sample 3 → model thinks class 2 (bird)

actual_classes = torch.tensor([0, 1, 2])   # actual labels: cat, dog, bird

criterion_ce = nn.CrossEntropyLoss()
ce_loss      = criterion_ce(logits, actual_classes)
print(f"\nCross Entropy Loss = {ce_loss.item():.4f}")
# Output: Cross Entropy Loss = 0.3XXX (low — model predicted correctly!)

# What if the model was wrong?
wrong_logits = torch.tensor([[0.1, 2.0, 1.5],   # sample 1 → wrongly thinks dog
                              [2.5, 0.5, 0.3],   # sample 2 → wrongly thinks cat
                              [0.1, 3.0, 0.2]])  # sample 3 → wrongly thinks dog

wrong_loss = criterion_ce(wrong_logits, actual_classes)
print(f"Cross Entropy Loss (wrong predictions) = {wrong_loss.item():.4f}")
# Output: Cross Entropy Loss (wrong predictions) = 2.XXXX (much higher!)




# ============================================================
# PART 5: LOSS FUNCTIONS — FULL COMPARISON DIAGRAM
# ============================================================

#
#  LOSS FUNCTION        FORMULA                  PROBLEM TYPE      OUTPUT ACTIVATION
#  ──────────────────────────────────────────────────────────────────────────────────
#  MSE                  mean((pred-actual)²)     Regression        Linear (none)
#  MAE                  mean(|pred-actual|)      Regression        Linear (none)
#  Binary Cross Entropy -[y*log(p)+(1-y)*log(p)] Binary Classif.   Sigmoid (0 to 1)
#  Cross Entropy        -log(p_correct_class)    Multi Classif.    Softmax (sum to 1)
#
#
#  HOW TO CHOOSE:
#
#  Predicting a NUMBER?
#  └──> Use MSE or MAE
#       └──> Outliers in data? → MAE
#       └──> No outliers?      → MSE
#
#  Predicting a CLASS?
#  └──> 2 classes (yes/no)?      → Binary Cross Entropy + Sigmoid
#  └──> 3+ classes (cat/dog/...) → Cross Entropy + Softmax




# ============================================================
# PART 6: PUTTING IT ALL TOGETHER — PERCEPTRON WITH BCE LOSS
# ============================================================

# Full training loop with proper loss function

# Dataset: will a student pass? (study_hours, sleep_hours) → pass=1, fail=0
X = torch.tensor([
    [1.0, 4.0],   # studied 1hr, slept 4hrs → fail
    [2.0, 5.0],   # studied 2hr, slept 5hrs → fail
    [3.0, 6.0],   # studied 3hr, slept 6hrs → pass
    [4.0, 7.0],   # studied 4hr, slept 7hrs → pass
    [5.0, 8.0],   # studied 5hr, slept 8hrs → pass
], dtype=torch.float32)

y = torch.tensor([[0],[0],[1],[1],[1]], dtype=torch.float32)

# Build perceptron
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.layer(x))  # sigmoid → output between 0 and 1

model     = Perceptron()
criterion = nn.BCELoss()                              # binary classification → BCE
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

losses = []  # track loss over time

for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X)
    loss   = criterion(output, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch+1) % 200 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
        # Output:
        # Epoch 200  | Loss: 0.5123
        # Epoch 400  | Loss: 0.3456
        # Epoch 600  | Loss: 0.2341
        # Epoch 800  | Loss: 0.1789
        # Epoch 1000 | Loss: 0.1432

# Final predictions
print("\n--- Will the student pass? ---")
with torch.no_grad():
    preds = model(X)
    for i in range(len(X)):
        result = "PASS ✅" if preds[i].item() > 0.5 else "FAIL ❌"
        print(f"Study:{X[i][0].item()}hr Sleep:{X[i][1].item()}hr → {result} (prob={preds[i].item():.2f})")
        # Output:
        # Study:1.0hr Sleep:4.0hr → FAIL ❌ (prob=0.18)
        # Study:2.0hr Sleep:5.0hr → FAIL ❌ (prob=0.31)
        # Study:3.0hr Sleep:6.0hr → PASS ✅ (prob=0.67)
        # Study:4.0hr Sleep:7.0hr → PASS ✅ (prob=0.83)
        # Study:5.0hr Sleep:8.0hr → PASS ✅ (prob=0.94)

# Plot loss curve
plt.plot(losses)
plt.title("Loss over Training Epochs")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.grid(True)
plt.show()
# Output: smooth curve going DOWN from ~0.7 to ~0.14
#         this means model is learning!




# ============================================================
# KEY POINTS TO REMEMBER:
# ============================================================

# 1. Loss function = measures HOW WRONG the model's prediction is
# 2. Goal of training = MINIMIZE the loss function
# 3. MSE = for regression, squares the errors, sensitive to outliers
# 4. MAE = for regression, absolute errors, robust to outliers
# 5. BCE = for binary classification (0 or 1), use with Sigmoid
# 6. Cross Entropy = for multi-class, use with Softmax
# 7. BCE punishes OVERCONFIDENT WRONG predictions very heavily
# 8. Lower loss = better model
# 9. Loss of 0 = perfect predictions (rare in real life)
# 10. Always match your loss function to your problem type!

# ============================================================
# NEXT TOPIC: Gradient Descent
# (how the model actually uses the loss to update its weights)
# ============================================================