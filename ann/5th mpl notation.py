# ============================================================
# MLP NOTATION — SEPARATE REFERENCE NOTES
# ============================================================
# Save this as: 04b_mlp_notation.py
# Use this as a CHEAT SHEET whenever you see MLP math
# ============================================================




# ============================================================
# WHY LEARN NOTATION?
# ============================================================

# When you read research papers, textbooks, or watch lectures
# you will see symbols like W², δ³, a⁽²⁾ etc.
# Without knowing the notation → everything looks like alien language
# With notation → you can read ANY neural network math easily

# ANALOGY:
# Notation is like learning cricket scoreboard symbols
# Once you know what W, R, 4, 6 mean on a scoreboard
# you can read ANY cricket match scorecard in the world!




# ============================================================
# PART 1: LAYER NOTATION
# ============================================================

#
#   Superscript in square brackets = LAYER NUMBER
#
#   L        = total number of layers in the network
#   [l]      = refers to layer number l
#   [0]      = input layer
#   [1]      = first hidden layer
#   [2]      = second hidden layer
#   [L]      = output layer (last layer)
#
#
#   EXAMPLE — a 3 layer MLP:
#
#   Layer [0] → Input Layer      (not counted in depth usually)
#   Layer [1] → Hidden Layer 1
#   Layer [2] → Hidden Layer 2
#   Layer [3] → Output Layer     ← L = 3
#
#
#   DIAGRAM:
#
#   [0]        [1]         [2]         [3]
#   INPUT   HIDDEN-1    HIDDEN-2    OUTPUT
#
#   x ──────> a[1] ─────> a[2] ─────> a[3] = ŷ
#




# ============================================================
# PART 2: NEURON AND LAYER SIZE NOTATION
# ============================================================

#
#   n[l]   = number of neurons in layer l
#
#   EXAMPLE:
#   n[0] = 3   → input layer has 3 features
#   n[1] = 4   → hidden layer 1 has 4 neurons
#   n[2] = 4   → hidden layer 2 has 4 neurons
#   n[3] = 1   → output layer has 1 neuron
#
#
#   DIAGRAM with sizes:
#
#   [0]         [1]          [2]         [3]
#   n=3         n=4          n=4         n=1
#
#   (x1)      (h1)(h2)     (h5)(h6)
#   (x2)  →   (h3)(h4)  →  (h7)(h8) →  (out)
#   (x3)
#




# ============================================================
# PART 3: WEIGHT MATRIX NOTATION
# ============================================================

#
#   W[l]   = weight matrix for layer l
#            (connects layer l-1 to layer l)
#
#   SHAPE of W[l]:
#   W[l].shape = (n[l], n[l-1])
#                 ↑       ↑
#              neurons  neurons in
#              in layer  previous layer
#
#
#   EXAMPLE:
#   W[1].shape = (n[1], n[0]) = (4, 3)  → 4 neurons, 3 inputs each
#   W[2].shape = (n[2], n[1]) = (4, 4)  → 4 neurons, 4 inputs each
#   W[3].shape = (n[3], n[2]) = (1, 4)  → 1 neuron,  4 inputs
#
#
#   Individual weight:
#   W[l]_(j,k) = weight connecting neuron k (layer l-1)
#                                  to neuron j (layer l)
#
#   DIAGRAM:
#
#   Layer [0]     Layer [1]
#
#   x1 ──W[1]_(1,1)──> h1
#   x1 ──W[1]_(2,1)──> h2
#   x2 ──W[1]_(1,2)──> h1
#   x2 ──W[1]_(2,2)──> h2
#

import torch
import torch.nn as nn

# See weight shapes in PyTorch
model = nn.Sequential(
    nn.Linear(3, 4),   # W[1]: shape (4,3)
    nn.ReLU(),
    nn.Linear(4, 4),   # W[2]: shape (4,4)
    nn.ReLU(),
    nn.Linear(4, 1),   # W[3]: shape (1,4)
    nn.Sigmoid()
)

for i, layer in enumerate(model):
    if isinstance(layer, nn.Linear):
        print(f"W[{i//2 + 1}] shape = {layer.weight.shape} | b[{i//2 + 1}] shape = {layer.bias.shape}")
        # Output:
        # W[1] shape = torch.Size([4, 3]) | b[1] shape = torch.Size([4])
        # W[2] shape = torch.Size([4, 4]) | b[2] shape = torch.Size([4])
        # W[3] shape = torch.Size([1, 4]) | b[3] shape = torch.Size([1])




# ============================================================
# PART 4: BIAS NOTATION
# ============================================================

#
#   b[l]   = bias vector for layer l
#   shape  = (n[l], 1)   → one bias per neuron in that layer
#
#   EXAMPLE:
#   b[1].shape = (4, 1)  → 4 biases for 4 neurons in layer 1
#   b[2].shape = (4, 1)  → 4 biases for 4 neurons in layer 2
#   b[3].shape = (1, 1)  → 1 bias for 1 output neuron
#
#   ANALOGY:
#   Bias is like a BASE SALARY for each neuron
#   Even if all inputs are 0, the neuron still has its base salary
#   It shifts the activation threshold up or down




# ============================================================
# PART 5: PRE-ACTIVATION AND ACTIVATION NOTATION
# ============================================================

#
#   z[l]   = pre-activation (weighted sum BEFORE activation function)
#   a[l]   = activation     (output AFTER activation function)
#
#   FORMULA:
#   z[l] = W[l] · a[l-1] + b[l]     ← matrix multiplication + bias
#   a[l] = g[l]( z[l] )              ← apply activation function g
#
#   where:
#   a[0]  = X  (input layer output = raw input)
#   a[L]  = ŷ  (output layer output = final prediction)
#   g[l]  = activation function at layer l (ReLU, Sigmoid etc.)
#
#
#   FULL FLOW DIAGRAM:
#
#   a[0]=X
#     │
#     ▼
#   z[1] = W[1]·a[0] + b[1]  ──> a[1] = ReLU(z[1])
#                                          │
#                                          ▼
#                              z[2] = W[2]·a[1] + b[2]  ──> a[2] = ReLU(z[2])
#                                                                     │
#                                                                     ▼
#                                                         z[3] = W[3]·a[2] + b[3]
#                                                                     │
#                                                                     ▼
#                                                         a[3] = Sigmoid(z[3]) = ŷ

# Manually computing forward pass using notation
torch.manual_seed(0)

# Input (a[0])
a0 = torch.tensor([[0.5, 0.8, 0.2]])  # shape (1, 3) → 1 sample, 3 features

# Layer 1 weights and bias
W1 = torch.randn(4, 3)  # shape (n[1], n[0]) = (4, 3)
b1 = torch.randn(4, 1)  # shape (n[1], 1)    = (4, 1)

# Layer 2 weights and bias
W2 = torch.randn(4, 4)
b2 = torch.randn(4, 1)

# Layer 3 weights and bias
W3 = torch.randn(1, 4)
b3 = torch.randn(1, 1)

# Forward pass using notation
z1 = W1 @ a0.T + b1           # z[1] = W[1] · a[0] + b[1]
a1 = torch.relu(z1)            # a[1] = ReLU(z[1])

z2 = W2 @ a1 + b2             # z[2] = W[2] · a[1] + b[2]
a2 = torch.relu(z2)            # a[2] = ReLU(z[2])

z3 = W3 @ a2 + b3             # z[3] = W[3] · a[2] + b[3]
a3 = torch.sigmoid(z3)         # a[3] = Sigmoid(z[3]) = ŷ

print(f"z[1] shape = {z1.shape}")   # Output: z[1] shape = torch.Size([4, 1])
print(f"a[1] shape = {a1.shape}")   # Output: a[1] shape = torch.Size([4, 1])
print(f"z[2] shape = {z2.shape}")   # Output: z[2] shape = torch.Size([4, 1])
print(f"a[2] shape = {a2.shape}")   # Output: a[2] shape = torch.Size([4, 1])
print(f"z[3] shape = {z3.shape}")   # Output: z[3] shape = torch.Size([1, 1])
print(f"a[3] = ŷ  = {a3.item():.4f}")  # Output: a[3] = ŷ = 0.XXXX




# ============================================================
# PART 6: FULL NOTATION SUMMARY DIAGRAM
# ============================================================

#
#   COMPLETE MLP NOTATION AT A GLANCE
#   ════════════════════════════════════════════════════════
#
#   SYMBOL     MEANING                          EXAMPLE
#   ────────────────────────────────────────────────────────
#   L          total layers                     L = 3
#   [l]        layer index                      [1], [2], [3]
#   n[l]       neurons in layer l               n[1]=4, n[2]=4
#   W[l]       weight matrix of layer l         shape: (n[l], n[l-1])
#   b[l]       bias vector of layer l           shape: (n[l], 1)
#   z[l]       pre-activation of layer l        z[l] = W[l]·a[l-1]+b[l]
#   a[l]       activation of layer l            a[l] = g(z[l])
#   g[l]       activation function of layer l   ReLU, Sigmoid, etc.
#   a[0]       input to network                 = X (raw features)
#   a[L]       output of network                = ŷ (prediction)
#   ŷ          predicted output                 value between 0 and 1
#   y          actual true label                0 or 1
#   m          number of training samples       m = 1000
#   (i)        refers to i-th sample            x(i), y(i)
#
#
#   FORWARD PASS SUMMARY:
#   ════════════════════════════════════════════════════════
#
#   for l = 1 to L:
#       z[l] = W[l] · a[l-1] + b[l]
#       a[l] = g[l]( z[l] )
#
#   final output: ŷ = a[L]
#
#
#   SHAPE RULES (SUPER IMPORTANT):
#   ════════════════════════════════════════════════════════
#
#   W[l]  →  (n[l],   n[l-1])
#   b[l]  →  (n[l],   1)
#   z[l]  →  (n[l],   m)        ← m = number of samples
#   a[l]  →  (n[l],   m)
#




# ============================================================
# PART 7: NOTATION IN PYTORCH — WHERE YOU SEE IT
# ============================================================

# In PyTorch, nn.Linear(in, out) maps directly to notation:
# nn.Linear(n[l-1], n[l]) creates W[l] and b[l] automatically

class NotatedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer [1]: n[0]=3 inputs → n[1]=4 neurons
        self.W1 = nn.Linear(3, 4)   # W[1] shape = (4,3), b[1] shape = (4,)

        # Layer [2]: n[1]=4 inputs → n[2]=4 neurons
        self.W2 = nn.Linear(4, 4)   # W[2] shape = (4,4), b[2] shape = (4,)

        # Layer [3]: n[2]=4 inputs → n[3]=1 neuron
        self.W3 = nn.Linear(4, 1)   # W[3] shape = (1,4), b[3] shape = (1,)

    def forward(self, x):
        a0 = x                             # a[0] = input
        z1 = self.W1(a0)                   # z[1] = W[1]·a[0] + b[1]
        a1 = torch.relu(z1)                # a[1] = ReLU(z[1])

        z2 = self.W2(a1)                   # z[2] = W[2]·a[1] + b[2]
        a2 = torch.relu(z2)                # a[2] = ReLU(z[2])

        z3 = self.W3(a2)                   # z[3] = W[3]·a[2] + b[3]
        a3 = torch.sigmoid(z3)             # a[3] = ŷ = Sigmoid(z[3])
        return a3

notated_model = NotatedMLP()
sample = torch.tensor([[0.5, 0.8, 0.2]])
output = notated_model(sample)
print(f"\nFinal output ŷ = {output.item():.4f}")  # Output: ŷ = 0.XXXX




# ============================================================
# KEY POINTS TO REMEMBER:
# ============================================================

# 1.  [l] in superscript = layer number
# 2.  n[l] = number of neurons in layer l
# 3.  W[l] shape = (n[l], n[l-1]) — always rows=current, cols=previous
# 4.  b[l] shape = (n[l], 1)
# 5.  z[l] = pre-activation = W[l]·a[l-1] + b[l]
# 6.  a[l] = post-activation = g(z[l])
# 7.  a[0] = X (raw input), a[L] = ŷ (final prediction)
# 8.  g[l] = activation function (ReLU for hidden, Sigmoid for output)
# 9.  In PyTorch: nn.Linear(n[l-1], n[l]) handles W and b automatically
# 10. Learning this notation = reading ANY research paper becomes easy!

# ============================================================
# NEXT TOPIC: Forward Propagation in depth
# (step by step math + code of how data flows through MLP)
# ============================================================