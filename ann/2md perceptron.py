 # ============================================================
# PERCEPTRON — COMPLETE NOTES
# Single Layer → Multi Layer → Everything You Need to Know
# ============================================================




# ============================================================
# PART 1: WHAT IS A PERCEPTRON?
# ============================================================

# The Perceptron is the SIMPLEST neural network that ever existed
# Just ONE single neuron that makes a decision

# Invented by Frank Rosenblatt in 1958
# It was the FIRST step toward modern deep learning

# ANALOGY:
# Imagine a BOUNCER at a club:
#   - He looks at multiple things: age, dress code, VIP status
#   - He gives each thing a weight (importance)
#   - He adds it all up
#   - If total score crosses his threshold → YOU'RE IN (output = 1)
#   - If not → YOU'RE OUT (output = 0)
# That is exactly what a perceptron does!




# ============================================================
# PART 2: HOW A PERCEPTRON WORKS — VISUAL DIAGRAM
# ============================================================

#
#        INPUTS          WEIGHTS
#
#   x1 (age=1)    ----[w1=0.8]----\
#                                   \
#   x2 (dress=1)  ----[w2=0.5]------[  NEURON  ]----> activation ---> output (0 or 1)
#                                   /    +bias
#   x3 (vip=0)   ----[w3=0.7]----/
#
#
#   STEP 1: z = (x1*w1) + (x2*w2) + (x3*w3) + bias
#   STEP 2: output = activation_function(z)
#
#   For Perceptron → activation = STEP FUNCTION
#   if z >= 0 → output = 1  (let them in)
#   if z <  0 → output = 0  (turn them away)




# ============================================================
# PART 3: PERCEPTRON IN PYTORCH — CODING IT FROM SCRATCH
# ============================================================

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ANALOGY:
# We'll teach the perceptron to act like an AND gate
# AND gate truth table:
#   input1=0, input2=0 → output=0
#   input1=0, input2=1 → output=0
#   input1=1, input2=0 → output=0
#   input1=1, input2=1 → output=1  ← only fires when BOTH are 1

# Step 1: Define the data (AND gate)
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[0],[0],[1]],         dtype=torch.float32)

# Step 2: Build the Perceptron
class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.layer = nn.Linear(2, 1)   # 2 inputs → 1 output neuron

    def forward(self, x):
        z = self.layer(x)              # weighted sum + bias
        output = torch.sigmoid(z)      # smooth version of step function
        return output

# Step 3: Initialize model, loss, optimizer
model     = Perceptron()
criterion = nn.BCELoss()                        # Binary Cross Entropy for 0/1 output
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Step 4: Train the perceptron
for epoch in range(1000):
    optimizer.zero_grad()       # clear old gradients
    output = model(X)           # forward pass
    loss   = criterion(output, y) # calculate loss
    loss.backward()             # backpropagation
    optimizer.step()            # update weights

    if (epoch+1) % 200 == 0:
        print(f"Epoch {epoch+1}/1000 | Loss: {loss.item():.4f}")
        # Output:
        # Epoch 200/1000 | Loss: 0.3521
        # Epoch 400/1000 | Loss: 0.2134
        # Epoch 600/1000 | Loss: 0.1456
        # Epoch 800/1000 | Loss: 0.1098
        # Epoch 1000/1000 | Loss: 0.0876

# Step 5: Test the perceptron
print("\n--- Perceptron Predictions (AND Gate) ---")
with torch.no_grad(): 
    predictions = model(X)
    for i in range(len(X)):
        pred = 1 if predictions[i].item() > 0.5 else 0
        print(f"Input: {X[i].tolist()} → Predicted: {pred} | Actual: {int(y[i].item())}")
        # Output:
        # Input: [0.0, 0.0] → Predicted: 0 | Actual: 0  ✓
        # Input: [0.0, 1.0] → Predicted: 0 | Actual: 0  ✓
        # Input: [1.0, 0.0] → Predicted: 0 | Actual: 0  ✓
        # Input: [1.0, 1.0] → Predicted: 1 | Actual: 1  ✓




# ============================================================
# PART 4: THE BIG LIMITATION OF SINGLE LAYER PERCEPTRON
# ============================================================

# ANALOGY:
# A single perceptron can ONLY draw ONE straight line to separate data
# Like a bouncer who only has ONE rule: "must be above 18"
# But what if the rule is complex? He can't handle it alone

# This is called LINEAR SEPARABILITY
# Perceptron works ONLY if data can be separated by a straight line

#
#   AND gate → linearly separable ✅
#
#   0  0  |  plotted:
#   0  1  |     .  .
#   1  0  |     .  ✓      ← only [1,1] is class 1
#   1  1  |
#          → one line CAN separate this
#
#
#   XOR gate → NOT linearly separable ❌
#
#   0  0 → 0
#   0  1 → 1
#   1  0 → 1
#   1  1 → 0
#
#   plotted:
#       1  .  ✓          ← class 1 and class 0 are MIXED
#       0  ✓  .          ← NO single straight line can separate them!
#
#   → single perceptron FAILS on XOR
#   → we need MULTIPLE LAYERS to solve this!




# ============================================================
# PART 5: MULTI-LAYER PERCEPTRON (MLP) — THE SOLUTION
# ============================================================

# ANALOGY:
# One bouncer (single perceptron) can only apply ONE rule
# But a TEAM of bouncers (MLP) can apply MULTIPLE rules together
# Bouncer 1: checks age
# Bouncer 2: checks dress code
# Bouncer 3: checks VIP list
# Manager (output): combines all their reports → final decision

# MLP = multiple perceptrons stacked in layers
# This is what we call an ANN (Artificial Neural Network)!

# VISUAL DIAGRAM:
#
#   INPUT LAYER    HIDDEN LAYER    OUTPUT LAYER
#
#      x1 --------> [N1] \
#                         \
#      x2 --------> [N2] ---> [N5] ----> output
#                         /
#      x3 --------> [N3] /
#             \
#              -----> [N4] /
#
#   Each N = one perceptron (neuron)
#   Hidden layer neurons learn INTERMEDIATE features
#   Output neuron makes the FINAL decision




# ============================================================
# PART 6: MLP SOLVING XOR — WHAT SINGLE PERCEPTRON COULDN'T
# ============================================================

# XOR gate:
#   [0,0] → 0
#   [0,1] → 1
#   [1,0] → 1
#   [1,1] → 0

X_xor = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y_xor = torch.tensor([[0],[1],[1],[0]],          dtype=torch.float32)

# Build MLP with a hidden layer
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(2, 4)    # 2 inputs → 4 hidden neurons
        self.output = nn.Linear(4, 1)    # 4 hidden → 1 output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x)) # hidden layer with activation
        x = self.sigmoid(self.output(x)) # output layer with activation
        return x

mlp       = MLP()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.5)

# Train MLP on XOR
for epoch in range(5000):
    optimizer.zero_grad()
    output = mlp(X_xor)
    loss   = criterion(output, y_xor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 1000 == 0:
        print(f"Epoch {epoch+1}/5000 | Loss: {loss.item():.4f}")
        # Output:
        # Epoch 1000/5000 | Loss: 0.2934
        # Epoch 2000/5000 | Loss: 0.1423
        # Epoch 3000/5000 | Loss: 0.0812
        # Epoch 4000/5000 | Loss: 0.0534
        # Epoch 5000/5000 | Loss: 0.0389

# Test MLP on XOR
print("\n--- MLP Predictions (XOR Gate) ---")
with torch.no_grad():
    predictions = mlp(X_xor)
    for i in range(len(X_xor)):
        pred = 1 if predictions[i].item() > 0.5 else 0
        print(f"Input: {X_xor[i].tolist()} → Predicted: {pred} | Actual: {int(y_xor[i].item())}")
        # Output:
        # Input: [0.0, 0.0] → Predicted: 0 | Actual: 0  ✓
        # Input: [0.0, 1.0] → Predicted: 1 | Actual: 1  ✓
        # Input: [1.0, 0.0] → Predicted: 1 | Actual: 1  ✓
        # Input: [1.0, 1.0] → Predicted: 0 | Actual: 0  ✓
        # MLP solves XOR perfectly — single perceptron could NOT!




# ============================================================
# PART 7: SINGLE LAYER vs MULTI LAYER — SIDE BY SIDE
# ============================================================

#
#  SINGLE LAYER PERCEPTRON          MULTI LAYER PERCEPTRON (MLP)
#  ========================          ============================
#
#  INPUT → [NEURON] → OUTPUT        INPUT → [HIDDEN] → [HIDDEN] → OUTPUT
#
#  • Only 1 neuron                  • Many neurons in many layers
#  • Only learns linear patterns    • Learns complex non-linear patterns
#  • Fails on XOR                   • Solves XOR easily
#  • No hidden layers               • Has 1 or more hidden layers
#  • Good for: simple problems      • Good for: real world problems
#  • Example: AND, OR gate          • Example: image, text, speech
#
#
#  DEPTH COMPARISON:
#
#  Single Layer:
#  [x1]──┐
#  [x2]──┤──>[neuron]──> output
#  [x3]──┘
#
#  Multi Layer (3 layer ANN):
#  [x1]──┐       ┌──[h1]──┐
#  [x2]──┼──────>├──[h2]──┼──>[output]
#  [x3]──┘       └──[h3]──┘
#         input    hidden    output




# ============================================================
# PART 8: HOW DOES MLP ACTUALLY LEARN? (INTUITION)
# ============================================================

# ANALOGY:
# Imagine you're learning darts:
# Round 1: you throw → miss by a lot → someone tells you HOW FAR you missed (loss)
# Round 2: you adjust your aim based on the feedback (backpropagation)
# Round 3: throw again → miss less → adjust again
# After 1000 rounds → you hit bullseye consistently!

# That's EXACTLY how MLP learns:
# 1. FORWARD PASS   → throw the dart (make a prediction)
# 2. LOSS           → measure how far you missed
# 3. BACKPROP       → figure out which muscles to adjust (gradients)
# 4. OPTIMIZER      → actually adjust your aim (update weights)
# 5. REPEAT         → thousands of times until accurate

# Printing model weights to see what the MLP learned:
print("\n--- What MLP Learned (Weights) ---")
for name, param in mlp.named_parameters():
    print(f"{name}: {param.data}")
    # Output: shows the learned weights and biases
    # hidden.weight: tensor([[ 4.23, -3.89], ...])
    # hidden.bias:   tensor([-1.92, ...])
    # output.weight: tensor([[ 5.12, ...]])
    # output.bias:   tensor([-2.34])




# ============================================================
# PART 9: KEY TERMS CHEAT SHEET
# ============================================================

#
#  TERM               MEANING                         ANALOGY
#  ────────────────────────────────────────────────────────────
#  Input              raw data fed to network         ingredients in a recipe
#  Weight             importance of each input        how much salt vs sugar
#  Bias               shifts the decision boundary    your default mood
#  Activation         decides if neuron fires         light switch threshold
#  Forward Pass       data flows input → output       cooking the recipe
#  Loss               how wrong the prediction was    how bad the dish tastes
#  Backpropagation    find which weights caused loss   finding which ingredient ruined it
#  Optimizer          updates the weights             adjusting the recipe
#  Epoch              one full pass through all data  one round of practice
#  Hidden Layer       intermediate processing layer   kitchen prep team




# ============================================================
# KEY POINTS TO REMEMBER:
# ============================================================

# 1. Perceptron = single neuron → simplest neural network ever
# 2. It does: weighted sum + bias → activation → output
# 3. Single layer perceptron can ONLY solve linearly separable problems
# 4. It FAILS on XOR because XOR is not linearly separable
# 5. MLP = many perceptrons in layers → solves complex problems
# 6. Hidden layers are what give MLP its POWER
# 7. MLP learns through: forward pass → loss → backprop → optimizer
# 8. More hidden layers = more complex patterns it can learn
# 9. Single perceptron uses step function, MLP uses sigmoid/relu
# 10. MLP is also called ANN (Artificial Neural Network)

# ============================================================
# NEXT TOPIC: Forward Propagation in depth
# (how data travels through the network mathematically)
# ============================================================