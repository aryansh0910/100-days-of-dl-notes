# ============================================================
# FORWARD PROPAGATION — IN DEPTH
# ============================================================




# ============================================================
# WHAT IS FORWARD PROPAGATION?
# ============================================================

# Forward Propagation = data flowing FORWARD through the network
# from Input Layer → Hidden Layers → Output Layer
# to produce a PREDICTION

# ANALOGY:
# Think of an ASSEMBLY LINE in a factory 🏭
#
#   Raw Material (Input)
#       ↓
#   Machine 1 — cuts the material       (Hidden Layer 1)
#       ↓
#   Machine 2 — shapes it               (Hidden Layer 2)
#       ↓
#   Machine 3 — polishes it             (Hidden Layer 3)
#       ↓
#   Final Product (Output = Prediction)
#
# Each machine does its job and passes
# the result to the next machine
# Data flows in ONE direction only → forward
# That is forward propagation!


# ============================================================
# THE MATH — STEP BY STEP
# ============================================================

# For each layer l:
#
#   STEP 1: z[l] = W[l] · a[l-1] + b[l]
#           ↑ weighted sum (linear transformation)
#
#   STEP 2: a[l] = g( z[l] )
#           ↑ apply activation function (non-linear transformation)
#
# These 2 steps repeat for EVERY layer
# Final output: ŷ = a[L]
#
#
# WHY TWO STEPS?
#
# Step 1 (linear)    → combines all inputs with weights
#                      like mixing ingredients
#
# Step 2 (non-linear)→ activation adds complexity
#                      like cooking the mixture
#                      without this, stacking layers does NOTHING
#                      (linear + linear = still linear!)


# ============================================================
# VISUAL: FULL FORWARD PASS THROUGH A 3-LAYER MLP
# ============================================================

#
#   INPUT          LAYER 1           LAYER 2          OUTPUT
#   a[0]=X         z[1],a[1]         z[2],a[2]        z[3],a[3]=ŷ
#
#   ┌────┐      ┌──────────┐      ┌──────────┐      ┌────────┐
#   │ x1 │──┐   │          │      │          │      │        │
#   │ x2 │──┼──>│ z[1]=    │──┐   │ z[2]=    │──┐   │ z[3]=  │
#   │ x3 │──┘   │W[1]·a[0] │  │   │W[2]·a[1] │  │   │W[3]·   │
#   └────┘      │  +b[1]   │  │   │  +b[2]   │  │-->│a[2]+b3 │
#               │          │  │   │          │  │   │        │
#               │ a[1]=    │  │   │ a[2]=    │  │   │ ŷ=     │
#               │ ReLU(z1) │──┘   │ ReLU(z2) │──┘   │Sigmoid │
#               └──────────┘      └──────────┘      └────────┘
#
#   ══════════════════════════════════════════════════════
#   a[0] → z[1] → a[1] → z[2] → a[2] → z[3] → a[3]=ŷ
#   ══════════════════════════════════════════════════════




import torch
import torch.nn as nn




# ============================================================
# PART 1: FORWARD PROP — MANUAL STEP BY STEP (FROM SCRATCH)
# ============================================================

# We will do EVERYTHING manually
# No nn.Module, no black box
# Just raw matrix math so you SEE exactly what happens

torch.manual_seed(42)

# ── INPUT ──────────────────────────────────────────────────
# 1 sample, 3 features
# a[0] = X
a0 = torch.tensor([[0.5, 0.8, 0.2]])   # shape: (1, 3)
print(f"a[0] (input) shape     = {a0.shape}")   # Output: torch.Size([1, 3])

# ── LAYER 1 ────────────────────────────────────────────────
# n[0]=3 inputs, n[1]=4 neurons
W1 = torch.randn(4, 3)   # W[1] shape: (n[1], n[0]) = (4, 3)
b1 = torch.randn(4, 1)   # b[1] shape: (n[1], 1)    = (4, 1)

# Step 1: pre-activation
z1 = W1 @ a0.T + b1      # z[1] = W[1] · a[0]ᵀ + b[1]
                          # shape: (4,3)@(3,1) + (4,1) = (4,1)

# Step 2: activation (ReLU for hidden layer)
a1 = torch.relu(z1)      # a[1] = ReLU(z[1])

print(f"\n── Layer 1 ──")
print(f"W[1] shape             = {W1.shape}")   # Output: torch.Size([4, 3])
print(f"b[1] shape             = {b1.shape}")   # Output: torch.Size([4, 1])
print(f"z[1] shape             = {z1.shape}")   # Output: torch.Size([4, 1])
print(f"a[1] shape             = {a1.shape}")   # Output: torch.Size([4, 1])
print(f"z[1] values            = {z1.T}")       # Output: tensor([[...]])
print(f"a[1] after ReLU        = {a1.T}")       # Output: negatives become 0

# ── LAYER 2 ────────────────────────────────────────────────
# n[1]=4 inputs, n[2]=4 neurons
W2 = torch.randn(4, 4)   # W[2] shape: (n[2], n[1]) = (4, 4)
b2 = torch.randn(4, 1)   # b[2] shape: (n[2], 1)    = (4, 1)

z2 = W2 @ a1 + b2        # z[2] = W[2] · a[1] + b[2]
a2 = torch.relu(z2)      # a[2] = ReLU(z[2])

print(f"\n── Layer 2 ──")
print(f"W[2] shape             = {W2.shape}")   # Output: torch.Size([4, 4])
print(f"z[2] shape             = {z2.shape}")   # Output: torch.Size([4, 1])
print(f"a[2] shape             = {a2.shape}")   # Output: torch.Size([4, 1])

# ── OUTPUT LAYER ───────────────────────────────────────────
# n[2]=4 inputs, n[3]=1 output neuron
W3 = torch.randn(1, 4)   # W[3] shape: (n[3], n[2]) = (1, 4)
b3 = torch.randn(1, 1)   # b[3] shape: (n[3], 1)    = (1, 1)

z3 = W3 @ a2 + b3        # z[3] = W[3] · a[2] + b[3]
a3 = torch.sigmoid(z3)   # a[3] = Sigmoid(z[3]) = ŷ

print(f"\n── Output Layer ──")
print(f"W[3] shape             = {W3.shape}")       # Output: torch.Size([1, 4])
print(f"z[3] shape             = {z3.shape}")       # Output: torch.Size([1, 1])
print(f"ŷ = a[3]               = {a3.item():.4f}") # Output: some value 0 to 1




# ============================================================
# PART 2: FORWARD PROP — WHAT ReLU ACTUALLY DOES
# ============================================================

# ANALOGY:
# ReLU is like a DOOR with a one-way valve 🚪
# Positive signal → passes through unchanged ✅
# Negative signal → blocked, set to 0 ❌
#
# WHY DO WE NEED ACTIVATION?
# Without activation: stacking layers = still ONE linear function
# With activation:    network can learn CURVES and COMPLEX patterns
#
# Without ReLU:
# a[2] = W[2]·(W[1]·x + b[1]) + b[2]
#       = (W[2]·W[1])·x + (W[2]·b[1]+b[2])
#       = W_new · x + b_new   ← still just a straight line!
#
# With ReLU: the non-linearity BREAKS this collapse
# Each layer genuinely learns something NEW

z_example = torch.tensor([-3.0, -1.0, 0.0, 1.0, 2.0, 4.0])
a_relu    = torch.relu(z_example)
a_sigmoid = torch.sigmoid(z_example)

print("\n── What activations do to z values ──")
print(f"z values  = {z_example.tolist()}")
print(f"ReLU(z)   = {a_relu.tolist()}")
# Output: ReLU(z)    = [0.0, 0.0, 0.0, 1.0, 2.0, 4.0]
# negative values → 0, positive values → same

print(f"Sigmoid(z)= {[round(x,3) for x in a_sigmoid.tolist()]}")
# Output: Sigmoid(z) = [0.047, 0.269, 0.5, 0.731, 0.880, 0.982]
# all values squashed between 0 and 1




# ============================================================
# PART 3: FORWARD PROP WITH BATCH OF SAMPLES
# ============================================================

# In real training we don't pass 1 sample at a time
# We pass a BATCH of samples together (faster + more stable)

# ANALOGY:
# Instead of washing 1 shirt at a time (1 sample)
# you put 32 shirts in the washing machine together (batch of 32)
# Same process, much more efficient!

# When using batches, shapes change:
# a[0] shape = (m, n[0])   ← m = batch size
# z[l] shape = (m, n[l])
# a[l] shape = (m, n[l])

batch_size = 4   # 4 samples at once
torch.manual_seed(0)

# Batch input: 4 samples, 3 features each
X_batch = torch.randn(batch_size, 3)   # shape: (4, 3)
print(f"\nX_batch shape = {X_batch.shape}")   # Output: torch.Size([4, 3])

# Layer 1
W1 = torch.randn(3, 4)    # NOTE: in nn.Linear PyTorch transposes internally
b1 = torch.randn(4)

z1_batch = X_batch @ W1 + b1    # shape: (4,3)@(3,4) + (4,) = (4, 4)
a1_batch = torch.relu(z1_batch) # shape: (4, 4)

print(f"z[1] batch shape = {z1_batch.shape}") # Output: torch.Size([4, 4])
print(f"a[1] batch shape = {a1_batch.shape}") # Output: torch.Size([4, 4])
# Each ROW = one sample's activations


# ============================================================
# PART 4: FULL FORWARD PROP USING nn.Module (CLEAN VERSION)
# ============================================================

# Now let's see the SAME forward prop but using PyTorch properly
# PyTorch handles all the matrix math internally

class MLP(nn.Module):
    def __init__(self, n0, n1, n2, n3):
        # n0=input size, n1=hidden1, n2=hidden2, n3=output
        super().__init__()
        self.layer1 = nn.Linear(n0, n1)   # W[1], b[1]
        self.layer2 = nn.Linear(n1, n2)   # W[2], b[2]
        self.layer3 = nn.Linear(n2, n3)   # W[3], b[3]

    def forward(self, x):
        # ── Layer 1 ──────────────────────────
        z1 = self.layer1(x)          # z[1] = W[1]·a[0] + b[1]
        a1 = torch.relu(z1)          # a[1] = ReLU(z[1])

        # ── Layer 2 ──────────────────────────
        z2 = self.layer2(a1)         # z[2] = W[2]·a[1] + b[2]
        a2 = torch.relu(z2)          # a[2] = ReLU(z[2])

        # ── Output Layer ─────────────────────
        z3 = self.layer3(a2)         # z[3] = W[3]·a[2] + b[3]
        a3 = torch.sigmoid(z3)       # ŷ    = Sigmoid(z[3])

        return a3

# Build model: 3 inputs → 4 hidden → 4 hidden → 1 output
model = MLP(n0=3, n1=4, n2=4, n3=1)

# Single sample forward pass
x_single = torch.tensor([[0.5, 0.8, 0.2]])
y_hat    = model(x_single)
print(f"\nSingle sample prediction ŷ = {y_hat.item():.4f}")
# Output: ŷ = 0.XXXX (some value between 0 and 1)

# Batch forward pass
x_batch  = torch.randn(8, 3)   # 8 samples, 3 features
y_hat_batch = model(x_batch)
print(f"Batch predictions shape     = {y_hat_batch.shape}")
# Output: torch.Size([8, 1])
print(f"Batch predictions           = {y_hat_batch.detach().T}")
# Output: tensor([[0.XX, 0.XX, 0.XX, 0.XX, 0.XX, 0.XX, 0.XX, 0.XX]])


# ============================================================
# PART 5: INSPECTING EVERY LAYER DURING FORWARD PASS
# ============================================================

# Useful for debugging — see what each layer is outputting

class MLP_Debug(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 4)
        self.layer2 = nn.Linear(4, 4)
        self.layer3 = nn.Linear(4, 1)

    def forward(self, x):
        print(f"\n  a[0] input shape    = {x.shape}")

        z1 = self.layer1(x)
        a1 = torch.relu(z1)
        print(f"  z[1] shape          = {z1.shape}")
        print(f"  a[1] shape          = {a1.shape}")
        print(f"  a[1] values         = {a1.detach()}")

        z2 = self.layer2(a1)
        a2 = torch.relu(z2)
        print(f"  z[2] shape          = {z2.shape}")
        print(f"  a[2] shape          = {a2.shape}")

        z3 = self.layer3(a2)
        a3 = torch.sigmoid(z3)
        print(f"  z[3] shape          = {z3.shape}")
        print(f"  ŷ = a[3]            = {a3.detach()}")

        return a3

debug_model = MLP_Debug()
sample      = torch.tensor([[1.0, 0.5, 0.3]])

print("=== Forward Pass Debug ===")
output = debug_model(sample)
# Output:
#   a[0] input shape    = torch.Size([1, 3])
#   z[1] shape          = torch.Size([1, 4])
#   a[1] shape          = torch.Size([1, 4])
#   a[1] values         = tensor([[0.0, 0.XX, 0.0, 0.XX]])
#   z[2] shape          = torch.Size([1, 4])
#   a[2] shape          = torch.Size([1, 4])
#   z[3] shape          = torch.Size([1, 1])
#   ŷ = a[3]            = tensor([[0.XXXX]])




# ============================================================
# PART 6: FORWARD PROP — COMPLETE FLOW SUMMARY
# ============================================================

#
#   COMPLETE FORWARD PROPAGATION CHEAT SHEET
#   ════════════════════════════════════════════════════════════
#
#   INPUT:
#   X shape = (m, n[0])     m=batch size, n[0]=features
#
#   FOR EACH LAYER l = 1 to L:
#   ┌─────────────────────────────────────────────────────┐
#   │  z[l] = a[l-1] · W[l]ᵀ + b[l]                      │
#   │  a[l] = g[l]( z[l] )                                │
#   └─────────────────────────────────────────────────────┘
#
#   OUTPUT:
#   ŷ = a[L]                final prediction
#
#
#   ACTIVATION FUNCTION GUIDE:
#   ════════════════════════════════════════════════════════════
#
#   LAYER TYPE     ACTIVATION     WHY
#   ────────────────────────────────────────────────────────────
#   Hidden Layer   ReLU           Fast, avoids vanishing gradient
#   Output Binary  Sigmoid        Squashes to 0-1 (probability)
#   Output Multi   Softmax        Probabilities that sum to 1
#   Output Regress None/Linear    Raw number prediction
#
#
#   SHAPE CHEAT SHEET:
#   ════════════════════════════════════════════════════════════
#
#   W[l]  →  (n[l],  n[l-1])     weight matrix
#   b[l]  →  (n[l],)             bias vector
#   z[l]  →  (m,     n[l])       pre-activation (batch)
#   a[l]  →  (m,     n[l])       post-activation (batch)
#




# ============================================================
# KEY POINTS TO REMEMBER:
# ============================================================

# 1.  Forward prop = data flowing INPUT → HIDDEN → OUTPUT
# 2.  Two steps per layer: z[l] = linear, a[l] = activation
# 3.  z[l] = weighted sum + bias (linear transformation)
# 4.  a[l] = activation function applied to z[l]
# 5.  Without activation → stacking layers = useless (still linear)
# 6.  ReLU for hidden layers, Sigmoid/Softmax for output
# 7.  a[0] = raw input X, a[L] = final prediction ŷ
# 8.  Batch processing = multiple samples in one forward pass
# 9.  Shape rule: z[l] and a[l] both have shape (m, n[l])
# 10. Forward prop produces ŷ → then loss is calculated → then backprop

# ============================================================
# NEXT TOPIC: Backpropagation in depth
# (how the network learns from its mistakes using gradients)
# ============================================================