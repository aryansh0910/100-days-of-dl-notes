# ============================================================
# DEEP LEARNING FOUNDATIONS — COMPLETE NOTES
# Neuron → Perceptron → Neural Network → Types of Neural Networks
# ============================================================




# ============================================================
# PART 1: WHAT IS DEEP LEARNING?
# ============================================================

# Classical ML = you manually tell the model WHAT features to look at
# Example: "hey model, look at age, salary, location to predict churn"

# Deep Learning = the model FIGURES OUT the features BY ITSELF
# Example: you just give it raw images, it learns edges → shapes → faces on its own

# WHY "DEEP"?
# Because it uses MANY LAYERS stacked on top of each other
# Each layer learns something deeper/more abstract than the previous one

# ANALOGY:
# Think of it like learning to recognize a dog:
# Layer 1 → sees edges and lines
# Layer 2 → sees ears, paws, tail shapes
# Layer 3 → puts it together → "that's a dog!"

# Deep Learning needs:
# 1. Lots of DATA
# 2. Lots of COMPUTE (GPU)
# 3. The right ARCHITECTURE (ANN, CNN, RNN etc.)


# ============================================================
# PART 2: THE BIOLOGICAL NEURON (THE INSPIRATION)
# ============================================================

# Our brain has ~86 billion neurons
# Each neuron:
#   - receives signals from other neurons (INPUTS)
#   - processes the signal
#   - fires or doesn't fire (OUTPUT)

# ANALOGY:
# A neuron is like a decision maker in an office:
#   - It receives emails (inputs) from multiple people
#   - It reads and weighs how important each email is
#   - If the total importance crosses a threshold → it sends a reply (fires)
#   - If not → it stays quiet (doesn't fire)

# Parts of a biological neuron:
# Dendrites   → receive input signals
# Cell Body   → processes the signals
# Axon        → sends the output signal
# Synapse     → connection strength between neurons



# ============================================================
# PART 3: THE ARTIFICIAL NEURON (MATHEMATICAL VERSION)
# ============================================================

# An artificial neuron does the SAME thing mathematically:
# Step 1 → Take inputs (x1, x2, x3...)
# Step 2 → Multiply each input by a WEIGHT (w1, w2, w3...)
#           weights = how important is each input
# Step 3 → Add them all up + add a BIAS
#           bias = shifts the decision threshold
# Step 4 → Pass through ACTIVATION FUNCTION
#           activation = decides whether neuron "fires" or not


# FORMULA:
# output = activation( (x1*w1 + x2*w2 + x3*w3) + bias )

# ANALOGY:
# You're deciding whether to go to a party:
# x1 = is your best friend going?     weight = 0.8 (very important)
# x2 = is the venue nearby?           weight = 0.5 (somewhat important)
# x3 = is it raining?                 weight = -0.7 (negative, reduces chance)
# bias = your general mood today
# If the total score crosses your threshold → you go (output = 1)
# If not → you don't go (output = 0)

import torch

# Single artificial neuron in PyTorch
x = torch.tensor([1.0, 1.0, 0.0])   # inputs: friend going=yes, nearby=yes, raining=no
w = torch.tensor([0.8, 0.5, -0.7])  # weights
b = torch.tensor(0.1)               # bias

# Step 1: weighted sum
z = torch.dot(x, w) + b
print(f"Weighted sum z = {z.item():.4f}")  # Output: Weighted sum z = 1.4000

# Step 2: activation function (sigmoid here — squashes to 0-1)
output = torch.sigmoid(z)
print(f"Neuron output = {output.item():.4f}")  # Output: Neuron output = 0.8022
# Output close to 1 → neuron fires → you GO to the party!




# ============================================================
# PART 4: THE PERCEPTRON
# ============================================================

# The Perceptron is the SIMPLEST neural network — just ONE artificial neuron
# Invented by Frank Rosenblatt in 1958
# It is the BUILDING BLOCK of all neural networks

# HOW IT WORKS:
# Takes inputs → multiplies by weights → adds bias → passes through activation
# For perceptron, the activation is a STEP FUNCTION:
#   if z >= 0 → output = 1
#   if z <  0 → output = 0

# ANALOGY:
# A perceptron is like a light switch:
#   - if the total signal is strong enough → switch ON (output 1)
#   - if not strong enough → switch OFF (output 0)

# LIMITATION of Perceptron:
# It can only solve LINEARLY SEPARABLE problems
# Example: it can separate cats vs dogs IF they are cleanly separable
# It CANNOT solve XOR problem (we need multiple layers for that)

import torch
import torch.nn as nn

# Building a Perceptron using PyTorch
class Perceptron(nn.Module):
    def __init__(self, input_size):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # one neuron

    def forward(self, x):
        z = self.linear(x)           # weighted sum + bias
        output = torch.sigmoid(z)    # activation function
        return output

# Create perceptron with 3 inputs
perceptron = Perceptron(input_size=3)

# Pass one sample through it
sample = torch.tensor([[1.0, 0.5, -1.0]])
result = perceptron(sample)
print(f"Perceptron output = {result.item():.4f}")  # Output: some value between 0 and 1

# The perceptron learned nothing yet — weights are random
# Training will adjust weights to give correct outputs




# ============================================================
# PART 5: WHAT IS A NEURAL NETWORK (ANN)?
# ============================================================

# A Neural Network = MANY neurons connected in LAYERS

# ANALOGY:
# Think of a company:
#   - Interns (Input Layer)     → receive raw data/tasks
#   - Managers (Hidden Layers)  → process, analyze, extract patterns
#   - CEO (Output Layer)        → makes the final decision

# 3 TYPES OF LAYERS:
# 1. Input Layer  → receives raw data (pixels, numbers, text etc.)
# 2. Hidden Layer → learns patterns (can be 1 to 100s of layers)
# 3. Output Layer → gives final answer (cat/dog, price, yes/no)

# HOW LEARNING HAPPENS:
# Step 1 → Forward Pass: data flows from input → hidden → output
# Step 2 → Calculate Loss: how wrong was the prediction?
# Step 3 → Backward Pass (Backpropagation): calculate gradients
# Step 4 → Update Weights using Optimizer
# Repeat thousands of times → model gets better

# Building a simple ANN in PyTorch
class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.layer1 = nn.Linear(3, 4)    # input layer:  3 inputs  → 4 neurons
        self.layer2 = nn.Linear(4, 4)    # hidden layer: 4 neurons → 4 neurons
        self.layer3 = nn.Linear(4, 1)    # output layer: 4 neurons → 1 output
        self.relu   = nn.ReLU()          # activation function for hidden layers
        self.sigmoid = nn.Sigmoid()      # activation for output (binary classification)

    def forward(self, x):
        x = self.relu(self.layer1(x))    # input → hidden1
        x = self.relu(self.layer2(x))    # hidden1 → hidden2
        x = self.sigmoid(self.layer3(x)) # hidden2 → output
        return x

model = SimpleANN()
print(model)
# Output:
# SimpleANN(
#   (layer1): Linear(in_features=3, out_features=4, bias=True)
#   (layer2): Linear(in_features=4, out_features=4, bias=True)
#   (layer3): Linear(in_features=4, out_features=1, bias=True)
#   (relu): ReLU()
#   (sigmoid): Sigmoid()
# )

# Test forward pass
sample = torch.tensor([[1.0, 2.0, 3.0]])
output = model(sample)
print(f"ANN output = {output.item():.4f}")  # Output: some value between 0 and 1




# ============================================================
# PART 6: TYPES OF NEURAL NETWORKS
# ============================================================

# Different problems need different architectures
# ANALOGY:
# Just like different tools for different jobs:
#   hammer = nails, screwdriver = screws, wrench = bolts
#   ANN = tabular data, CNN = images, RNN = sequences

# -------------------------------------------------------
# TYPE 1: ANN — Artificial Neural Network (Feedforward)
# -------------------------------------------------------
# → Data flows in ONE direction: input → hidden → output
# → Used for: tabular/structured data
# → Examples: predicting house prices, customer churn, fraud detection
# → What you just built above is an ANN!

# -------------------------------------------------------
# TYPE 2: CNN — Convolutional Neural Network
# -------------------------------------------------------
# → Specially designed for IMAGES and SPATIAL data
# → Uses filters/kernels to detect edges, shapes, patterns
# → ANALOGY: like a magnifying glass sliding over an image,
#             looking for specific patterns in each region
# → Used for: image classification, object detection, face recognition
# → Examples: Google Photos, self-driving cars, medical imaging

import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)  # 1 channel in, 16 filters
        self.pool  = nn.MaxPool2d(2, 2)               # reduce spatial size
        self.fc1   = nn.Linear(16 * 13 * 13, 10)     # flatten → output 10 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # conv → activation → pool
        x = x.view(x.size(0), -1)                 # flatten
        x = self.fc1(x)                           # fully connected
        return x

# -------------------------------------------------------
# TYPE 3: RNN — Recurrent Neural Network
# -------------------------------------------------------
# → Designed for SEQUENTIAL data (order matters)
# → Has a MEMORY — output from previous step feeds into next step
# → ANALOGY: like reading a sentence word by word,
#             remembering previous words to understand the next one
# → Used for: time series, text, speech, stock prices
# → Examples: Google Translate (old), speech recognition, autocomplete

class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=10, hidden_size=20, batch_first=True)
        self.fc  = nn.Linear(20, 1)

    def forward(self, x):
        out, hidden = self.rnn(x)   # out has output at each time step
        out = self.fc(out[:, -1, :]) # take last time step output
        return out

# -------------------------------------------------------
# TYPE 4: LSTM — Long Short-Term Memory
# -------------------------------------------------------
# → Improved version of RNN — solves the VANISHING GRADIENT problem
# → RNN forgets long-term dependencies, LSTM remembers them
# → ANALOGY: RNN is like short-term memory (forgets quickly)
#             LSTM is like taking notes — you decide what to remember,
#             what to forget, and what to write down
# → Used for: same as RNN but for LONGER sequences
# → Examples: language models, sentiment analysis, music generation

# -------------------------------------------------------
# TYPE 5: Transformer
# -------------------------------------------------------
# → The MOST POWERFUL architecture today
# → Uses ATTENTION mechanism — looks at ALL parts of input at once
# → ANALOGY: instead of reading word by word (RNN),
#             a Transformer reads the WHOLE sentence at once
#             and decides which words are most important for each word
# → Used for: ChatGPT, BERT, LLaMA, Google Search, GitHub Copilot
# → This is what YOU will learn in Phase 3!

# -------------------------------------------------------
# TYPE 6: Autoencoder
# -------------------------------------------------------
# → Compresses data into a smaller representation, then reconstructs it
# → ANALOGY: like zipping a file → unzipping it
# → Used for: anomaly detection, image denoising, dimensionality reduction

# -------------------------------------------------------
# TYPE 7: GAN — Generative Adversarial Network
# -------------------------------------------------------
# → Two networks competing: Generator vs Discriminator
# → Generator creates fake data, Discriminator tries to catch fakes
# → ANALOGY: a counterfeiter (Generator) vs a detective (Discriminator)
#             both keep getting better by competing with each other
# → Used for: generating fake images, deepfakes, image-to-image translation
# → Examples: DALL-E (early), StyleGAN, face generation




# ============================================================
# SUMMARY TABLE
# ============================================================

# | Type        | Best For              | Famous Use Case          |
# |-------------|-----------------------|--------------------------|
# | ANN         | Tabular/structured    | Fraud detection          |
# | CNN         | Images, spatial data  | Face recognition         |
# | RNN         | Short sequences       | Autocomplete             |
# | LSTM        | Long sequences        | Language modeling        |
# | Transformer | Text, images, audio   | ChatGPT, BERT, LLaMA     |
# | Autoencoder | Compression, anomaly  | Image denoising          |
# | GAN         | Generating new data   | AI image generation      |




# ============================================================
# KEY POINTS TO REMEMBER:
# ============================================================

# 1. A Neuron = takes inputs → weights them → adds bias → activation → output
# 2. A Perceptron = simplest neural network (1 neuron), only solves linear problems
# 3. ANN = many neurons in layers (input → hidden → output)
# 4. Learning = forward pass → calculate loss → backprop → update weights
# 5. CNN = best for images (uses filters/kernels)
# 6. RNN/LSTM = best for sequences (text, time series)
# 7. Transformer = king of modern AI (attention mechanism)
# 8. GAN = two networks competing to generate realistic data
# 9. In this course you will go deep into ANN → CNN → RNN → LSTM
# 10. Transformers come in Phase 3 of your roadmap

# ============================================================
# NEXT TOPIC: ANN in depth — Forward Propagation
# ============================================================