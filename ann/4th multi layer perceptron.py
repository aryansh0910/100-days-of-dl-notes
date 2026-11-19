# ============================================================
# MLP — MULTI LAYER PERCEPTRON
# Complete Notes: Architecture → Forward Pass → Training → Real Example
# ============================================================




# ============================================================
# PART 1: WHAT IS AN MLP?
# ============================================================

# MLP = Multi Layer Perceptron
# It is the MOST BASIC form of a deep neural network
# Just multiple perceptrons (neurons) stacked in layers

# ANALOGY:
# Think of a COMPANY with departments  🏢
#
#   HR Dept (Input Layer)
#   → receives raw information (resumes, data)
#
#   Management Dept (Hidden Layers)
#   → processes, finds patterns, extracts insights
#   → each manager looks at different aspects
# 
#   CEO (Output Layer)
#   → makes the FINAL decision based on management reports 
#
# Each EMPLOYEE = one neuron
# Each DEPARTMENT = one layer
# Information flows from HR → Management → CEO


# ============================================================
# PART 2: MLP ARCHITECTURE — VISUAL DIAGRAM
# ============================================================

#
#   INPUT LAYER        HIDDEN LAYER 1      HIDDEN LAYER 2      OUTPUT LAYER
#   (3 neurons)        (4 neurons)         (4 neurons)         (1 neuron)
#
#      [x1]               [h1]                [h5]
#       |  \            /  |  \            /   |  \
#       |   \          /   |   \          /    |   \
#      [x2]  -------->    [h2]  -------->     [h6]  ----> [output]
#       |   /          \   |   /          \    |   /
#       |  /            \  |  /            \   |  /
#      [x3]               [h3]                [h7]
#                          |                   |
#                         [h4]                [h8]
#
#   Every neuron in one layer is connected to EVERY neuron in next layer
#   This is called FULLY CONNECTED or DENSE layer
#
#
#   INSIDE EACH NEURON:
#
#   inputs ──[weights]──> weighted sum ──[+bias]──> activation ──> output
#
#   z      = w1*x1 + w2*x2 + w3*x3 + bias
#   output = activation(z)




# ============================================================
# PART 3: THE 3 TYPES OF LAYERS
# ============================================================

#
#   LAYER TYPE      ROLE                          ANALOGY
#   ─────────────────────────────────────────────────────────────
#   Input Layer   → receives raw data             front door of company
#   Hidden Layer  → learns patterns & features    processing departments
#   Output Layer  → gives final answer            CEO decision
#
#
#   INPUT LAYER:
#   → number of neurons = number of features in your data
#   → example: predicting house price with 5 features → 5 input neurons
#   → NO computation happens here, just passes data forward
#
#   HIDDEN LAYER:
#   → this is where the MAGIC happens
#   → each neuron learns a different pattern/feature
#   → more hidden layers = more complex patterns model can learn
#   → uses activation functions (ReLU mostly)
#
#   OUTPUT LAYER:
#   → number of neurons depends on the PROBLEM TYPE
#   → binary classification  → 1 neuron  (with sigmoid)
#   → multi classification   → N neurons (with softmax, N = num classes)
#   → regression             → 1 neuron  (no activation)




# ============================================================
# PART 4: FORWARD PROPAGATION — HOW DATA FLOWS
# ============================================================

# Forward Propagation = data moving LEFT to RIGHT through the network
# Input → Hidden Layers → Output

# ANALOGY:
# Like an ASSEMBLY LINE in a factory 🏭
# Raw material (input) enters from left
# Each station (layer) processes and transforms it
# Final product (output/prediction) comes out at the right

# STEP BY STEP:
#
#   Step 1: Input enters the network
#   Step 2: Each neuron computes: z = weights * inputs + bias
#   Step 3: Apply activation function: a = activation(z)
#   Step 4: Output of this layer becomes INPUT of next layer
#   Step 5: Repeat until output layer
#   Step 6: Final output = prediction

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Manual forward pass to see exactly what happens inside MLP
torch.manual_seed(42)

# Tiny example: 3 inputs → 3 hidden → 1 output
x     = torch.tensor([[1.0, 2.0, 3.0]])  # one sample, 3 features

# Layer 1 weights and bias (randomly initialized normally)
W1 = torch.randn(3, 3)   # 3 inputs → 3 hidden neurons  (shape: out x in)
b1 = torch.randn(3)

# Layer 2 weights and bias
W2 = torch.randn(1, 3)   # 3 hidden → 1 output
b2 = torch.randn(1)

print("=== Manual Forward Pass ===")
print(f"Input x shape:  {x.shape}")           # Output: torch.Size([1, 3])

# Hidden layer computation
z1 = x @ W1.T + b1                            # matrix multiply + bias
a1 = torch.relu(z1)                           # apply ReLU activation
print(f"After Layer 1 (z1) shape: {z1.shape}")# Output: torch.Size([1, 3])
print(f"After ReLU   (a1) shape: {a1.shape}") # Output: torch.Size([1, 3])
print(f"a1 values: {a1}")                     # Output: tensor with ReLU applied

# Output layer computation
z2 = a1 @ W2.T + b2                           # matrix multiply + bias
a2 = torch.sigmoid(z2)                        # sigmoid for binary output
print(f"After Layer 2 (z2) shape: {z2.shape}")# Output: torch.Size([1, 1])
print(f"Final output  (a2): {a2.item():.4f}") # Output: some value between 0 and 1




# ============================================================
# PART 5: BUILDING MLP IN PYTORCH — CLEAN WAY
# ============================================================

# Now we build the same thing using PyTorch's nn.Module properly

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        # Define all layers
        self.layer1  = nn.Linear(input_size, hidden_size)   # input → hidden1
        self.layer2  = nn.Linear(hidden_size, hidden_size)  # hidden1 → hidden2
        self.layer3  = nn.Linear(hidden_size, output_size)  # hidden2 → output
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass — data flows through each layer
        x = self.relu(self.layer1(x))    # input → hidden1 → ReLU
        x = self.relu(self.layer2(x))    # hidden1 → hidden2 → ReLU
        x = self.sigmoid(self.layer3(x)) # hidden2 → output → Sigmoid
        return x

# Create model
model = MLP(input_size=10, hidden_size=16, output_size=1)
print("\n=== MLP Architecture ===")
print(model)
# Output:
# MLP(
#   (layer1): Linear(in_features=10, out_features=16, bias=True)
#   (layer2): Linear(in_features=16, out_features=16, bias=True)
#   (layer3): Linear(in_features=16, out_features=1, bias=True)
#   (relu): ReLU()
#   (sigmoid): Sigmoid()
# )

# Count total trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
# Output: Total parameters: 465
# these are all the weights + biases the model will learn




# ============================================================
# PART 6: COMPLETE MLP TRAINING — REAL DATASET
# ============================================================

# ANALOGY:
# Training MLP is like coaching a cricket team 🏏
# Practice session 1 (epoch 1) → team is bad → coach gives feedback (loss)
# Practice session 2 (epoch 2) → slightly better → more feedback
# After 100 sessions → team is good → ready for real match!

# Step 1: Create dataset (binary classification)
X, y = make_classification(
    n_samples    = 1000,   # 1000 data points
    n_features   = 10,     # 10 input features
    n_classes    = 2,      # binary: 0 or 1
    random_state = 42
)

# Step 2: Preprocess
scaler  = StandardScaler()
X       = scaler.fit_transform(X)              # normalize features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # shape: [800, 1]
y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)  # shape: [200, 1]

print(f"\nX_train shape: {X_train.shape}")  # Output: torch.Size([800, 10])
print(f"y_train shape: {y_train.shape}")  # Output: torch.Size([800, 1])

# Step 3: Build model, loss, optimizer
model     = MLP(input_size=10, hidden_size=16, output_size=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Training loop
train_losses = []
test_losses  = []

print("\n=== Training MLP ===")
for epoch in range(100):

    # --- TRAINING ---
    model.train()                        # set model to training mode
    optimizer.zero_grad()                # clear old gradients
    train_output = model(X_train)        # forward pass
    train_loss   = criterion(train_output, y_train)  # calculate loss
    train_loss.backward()                # backpropagation
    optimizer.step()                     # update weights
    train_losses.append(train_loss.item())

    # --- EVALUATION ---
    model.eval()                         # set model to eval mode
    with torch.no_grad():                # no gradients needed for eval
        test_output = model(X_test)
        test_loss   = criterion(test_output, y_test)
        test_losses.append(test_loss.item())

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}/100 | Train Loss: {train_loss.item():.4f} | Test Loss: {test_loss.item():.4f}")
        # Output:
        # Epoch  20/100 | Train Loss: 0.5823 | Test Loss: 0.5901
        # Epoch  40/100 | Train Loss: 0.4512 | Test Loss: 0.4678
        # Epoch  60/100 | Train Loss: 0.3821 | Test Loss: 0.3956
        # Epoch  80/100 | Train Loss: 0.3234 | Test Loss: 0.3401
        # Epoch 100/100 | Train Loss: 0.2891 | Test Loss: 0.3102

# Step 5: Evaluate accuracy
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predicted_classes = (predictions > 0.5).float()
    accuracy = (predicted_classes == y_test).float().mean()
    print(f"\nTest Accuracy: {accuracy.item()*100:.2f}%")
    # Output: Test Accuracy: ~87-90%

# Step 6: Plot training curve
plt.figure(figsize=(8,4))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(test_losses,  label='Test Loss',  color='orange')
plt.title('MLP Training — Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Output: two curves going DOWN together — model is learning!




# ============================================================
# PART 7: WHAT HAPPENS INSIDE DURING TRAINING
# ============================================================

#
#  EPOCH 1:
#  ┌─────────────────────────────────────────────────────┐
#  │  Forward Pass  → random weights → bad prediction    │
#  │  Loss          → very high (e.g. 0.69)              │
#  │  Backprop      → calculate gradients                │
#  │  Optimizer     → nudge weights slightly             │
#  └─────────────────────────────────────────────────────┘
#           ↓
#  EPOCH 2:
#  ┌─────────────────────────────────────────────────────┐
#  │  Forward Pass  → slightly better weights            │
#  │  Loss          → slightly lower (e.g. 0.61)         │
#  │  Backprop      → calculate gradients again          │
#  │  Optimizer     → nudge weights again                │
#  └─────────────────────────────────────────────────────┘
#           ↓
#  EPOCH 100:
#  ┌─────────────────────────────────────────────────────┐
#  │  Forward Pass  → well trained weights               │
#  │  Loss          → low (e.g. 0.28)                    │
#  │  Predictions   → mostly correct                     │
#  └─────────────────────────────────────────────────────┘




# ============================================================
# PART 8: USING nn.Sequential — CLEANER WAY TO BUILD MLP
# ============================================================

# Instead of defining each layer separately, we can use nn.Sequential
# It stacks layers in ORDER automatically

# ANALOGY:
# nn.Sequential is like a PIPELINE 🚰
# you just connect the pipes in order → water (data) flows through automatically

model_sequential = nn.Sequential(
    nn.Linear(10, 16),   # input layer
    nn.ReLU(),           # activation
    nn.Linear(16, 16),   # hidden layer
    nn.ReLU(),           # activation
    nn.Linear(16, 1),    # output layer
    nn.Sigmoid()         # output activation
)

print("\n=== Sequential MLP ===")
print(model_sequential)
# Output:
# Sequential(
#   (0): Linear(in_features=10, out_features=16, bias=True)
#   (1): ReLU()
#   (2): Linear(in_features=16, out_features=16, bias=True)
#   (3): ReLU()
#   (4): Linear(in_features=16, out_features=1, bias=True)
#   (5): Sigmoid()
# )

# Test it with a random input
dummy_input = torch.randn(5, 10)               # 5 samples, 10 features
output      = model_sequential(dummy_input)
print(f"Output shape: {output.shape}")         # Output: torch.Size([5, 1])
print(f"Output values:\n{output.detach()}")    # Output: 5 values between 0 and 1




# ============================================================
# PART 9: COMMON MISTAKES TO AVOID IN MLP
# ============================================================

#
#  MISTAKE 1: Forgetting optimizer.zero_grad()
#  → gradients ACCUMULATE across batches → wrong weight updates
#  → ALWAYS call zero_grad() before each forward pass
#
#  MISTAKE 2: Not normalizing input features
#  → features on different scales confuse the model
#  → ALWAYS use StandardScaler or MinMaxScaler before training
#
#  MISTAKE 3: Wrong output activation
#  → Binary classification  → use Sigmoid   + BCELoss
#  → Multi classification   → use Softmax   + CrossEntropyLoss
#  → Regression             → no activation + MSELoss
#
#  MISTAKE 4: Too many layers for small dataset
#  → model memorizes training data → fails on new data (overfitting)
#  → start simple, add layers only if needed
#
#  MISTAKE 5: Forgetting model.eval() during testing
#  → some layers behave differently in train vs eval mode
#  → ALWAYS set model.eval() before testing/inference




# ============================================================
# KEY POINTS TO REMEMBER:
# ============================================================

# 1.  MLP = multiple perceptrons stacked in layers (input → hidden → output)
# 2.  Every neuron in one layer connects to EVERY neuron in next layer
# 3.  Input layer  → receives raw features (no computation)
# 4.  Hidden layer → learns patterns (uses ReLU activation)
# 5.  Output layer → final prediction (activation depends on problem)
# 6.  Forward pass = data flows left to right through all layers
# 7.  Training loop = forward → loss → backward → optimizer step
# 8.  model.train() for training, model.eval() for testing
# 9.  nn.Sequential = cleaner way to stack layers in order
# 10. Always normalize your input data before feeding to MLP

# ============================================================
# NEXT TOPIC: Backpropagation
# (how the model figures out WHICH weights to change and by HOW MUCH)
# ============================================================ 