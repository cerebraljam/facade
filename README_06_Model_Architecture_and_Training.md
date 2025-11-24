# Document 6: Model Architecture and Training

## Introduction

This document explains Facade's neural network architecture and training process. We'll cover:
1. How features are processed into embeddings
2. The two-tower architecture implementation
3. The contrastive loss function
4. Training dynamics and optimization

---

## 1. High-Level Architecture Overview

### 1.1 The Complete Data Flow

```
Input: tf.SequenceExample (from ContextualizedActions)
  ↓
Token Lookup Embedders
  - Map tokens (b"alice") to embedding vectors
  ↓
Segment Embedders  
  - Process bags-of-weighted-tokens into segment vectors
  - Weighted averaging of token embeddings
  ↓
Concatenation
  - Combine all segment vectors
  ↓
SNN (Siamese Neural Network)
  - Feed-forward layers with ReLU
  ↓
Transformation
  - Softplus → L2 normalization
  ↓
Final Embeddings
  - Context embedding: 32-dim vector
  - Action embeddings: 32-dim vectors (one per action)
  ↓
Scoring
  - Dot product: score = <context_emb, action_emb>
  ↓
Loss Computation
  - Pairwise contrastive loss
  ↓
Backpropagation
  - Update embedding tables and neural network weights
```

---

## 2. Token Embedding Layers

### 2.1 Purpose

**Transform tokens** (bytes) **into dense vectors** (embeddings)

**Example**:
```
Token: b"alice"  →  Embedding: [0.23, -0.15, 0.67, ..., 0.42]  (16-dim)
Token: b"bob"    →  Embedding: [0.19, 0.31, -0.22, ..., 0.15]  (16-dim)
```

### 2.2 Vocabulary Building

**From training data**, collect all unique tokens:
```python
vocab = Vocabulary(
  embedding_name="context_username",
  tokens=[b"alice", b"bob", b"carol", b"diana", ...]
)
```

**Create embedding table**:
```python
embedding_table = tf.keras.layers.Embedding(
  input_dim=len(vocab.tokens) + num_oov_indices,  # e.g., 1000 + 1
  output_dim=16,  # From config: dimensions: 16
  name="context_username"
)
```

**Dimensions**:
- `input_dim`: Vocabulary size + OOV tokens
- `output_dim`: Embedding dimension (from config)

### 2.3 Token Lookup Process

**Input feature**:
```python
Feature(
  name="code_review/change_number_f/t",
  bag_of_weighted_words=WeightedTokens(
    tokens=[
      WeightedToken(token=b"bob", weight=1.0),
      WeightedToken(token=b"carol", weight=1.0)
    ]
  )
)
```

**Step 1: Map tokens to indices**
```python
token_to_index = {"alice": 0, "bob": 1, "carol": 2, ...}

tokens = [b"bob", b"carol"]
indices = [1, 2]
```

**Step 2: Lookup embeddings**
```python
embeddings = embedding_table(indices)
# Result shape: [2, 16]  (2 tokens, each 16-dim)

embeddings = [
  [0.19, 0.31, -0.22, ..., 0.15],  # bob's embedding
  [0.11, -0.05, 0.44, ..., 0.29]   # carol's embedding
]
```

### 2.4 Separate Embedding Tables

**Why separate tables for action vs context?**

**Same token, different meanings**:
- In action context: "alice" represents alice as a **resource accessor**
- In context: "alice" represents alice as a **collaborator/peer**

**Example**:
```python
# config.textproto
token_embedding_name_to_config {
  key: "action_username"
  value { dimensions: 16 }
}
token_embedding_name_to_config {
  key: "context_username"
  value { dimensions: 16 }
}
```

**Two separate embedding tables**:
- `action_username`: Used for `doc_access/prin/t` features
- `context_username`: Used for `code_review/change_number_f/t` features

**Result**: Token "alice" has different embeddings depending on context

---

## 3. Segment Embedding Layers

### 3.1 Purpose

**Transform bag-of-weighted-tokens into a single vector**

**Input**: Two features (tokens + weights)
```python
tokens_feature:   [bob, carol, diana]
weights_feature:  [bob:0.6, carol:0.3, diana:0.1]
```

**Output**: Single segment vector (16-dim)

### 3.2 The SegmentEmbedder Layer

**Configuration** (from config.textproto):
```textproto
segment_reductions {
  segment_weight_scaling: WS_IDENTITY
  segment_weight_normalization: WN_L2
  token_embedding_name: "context_username"
  token_feature_name: "code_review/change_number_f/t"
  intensity_feature_name: "code_review/change_number_f/w"
}
```

**Process**:

**Step 1: Lookup token embeddings**
```python
tokens = [bob, carol, diana]
token_embeddings = embedding_table.lookup(tokens)
# Shape: [3, 16]
```

**Step 2: Extract weights**
```python
weights = [0.6, 0.3, 0.1]  # From intensity_feature
```

**Step 3: Apply weight scaling** (if configured)
```python
if scaling == WS_IDENTITY:
  scaled_weights = weights  # [0.6, 0.3, 0.1]
elif scaling == WS_SQRT:
  scaled_weights = sqrt(weights)  # [0.77, 0.55, 0.32]
elif scaling == WS_LOG:
  scaled_weights = log(1 + weights)  # [0.47, 0.26, 0.10]
```

**Step 4: Apply weight normalization**
```python
if normalization == WN_L2:
  norm = sqrt(0.6^2 + 0.3^2 + 0.1^2) = 0.67
  normalized_weights = [0.6/0.67, 0.3/0.67, 0.1/0.67]
  normalized_weights = [0.89, 0.45, 0.15]
elif normalization == WN_SUM:
  sum = 0.6 + 0.3 + 0.1 = 1.0
  normalized_weights = [0.6, 0.3, 0.1]
elif normalization == WN_NONE:
  normalized_weights = weights
```

**Step 5: Compute weighted average**
```python
segment_embedding = (
  0.89 * embedding_bob +
  0.45 * embedding_carol +
  0.15 * embedding_diana
) / (0.89 + 0.45 + 0.15)

# Result shape: [16]  (single vector)
```

### 3.3 Dropout (Training Only)

**Token dropout** (configured via `dropout_tokens: 0.05`):
```python
# Randomly drop 5% of tokens
if training:
  keep_mask = tf.random.uniform([num_tokens]) > 0.05
  tokens = tokens[keep_mask]
  weights = weights[keep_mask]
```

**Effect**: Forces model to not rely on any single token

---

## 4. The Two Towers

### Why Two Separate Towers?

**The core insight**: Context and actions represent fundamentally different aspects of a security event.

- **Context**: "Who is this person and what do they normally do?" (independent of what they're accessing right now)
- **Action**: "What is this resource and who typically accesses it?" (independent of who's accessing it right now)

**Why separate processing matters**:
1. **Modularity**: You can update context features (e.g., add social network data) without retraining action tower
2. **Efficiency**: At inference, you can cache context embeddings (they don't change when evaluating multiple actions)
3. **Semantic clarity**: Forces the model to learn "what's normal for this person" vs "what's normal for this resource" separately

**The key question Facade answers**: "Does this person's normal behavior overlap with this resource's normal access patterns?" If yes → low score (normal). If no → high score (anomalous).

**Why this works for security**:
- Legitimate access: Alice (dev on Project X) accesses design docs for Project X → high overlap → normal
- Anomalous access: Alice suddenly accesses finance spreadsheets → no overlap → suspicious

---

### 4.1 Context Tower

**Purpose**: Process context features → context embedding

**What it does**: Transforms "who you work with" into a single vector that captures "what kind of work you do"

**Input**: Context features from ContextualizedActions
```python
context_features = {
  "code_review/change_number_f/t": [bob, carol],
  "code_review/change_number_f/w": [bob:0.6, carol:0.4],
  "code_review/change_number_b/t": [alice, diana],
  "code_review/change_number_b/w": [alice:0.7, diana:0.3]
}
```

**Architecture**:
```python
class ConcatenateThenSNNTower:
  def __init__(self):
    # One SegmentEmbedder per feature pair
    self.segment_embedders = {
      "code_review/change_number_f/t": SegmentEmbedder(...),
      "code_review/change_number_b/t": SegmentEmbedder(...)
    }
    # Feed-forward network
    self.snn = SNN(layer_sizes=[24], output_dim=32)
    # Transformations
    self.transformation = [Softplus, L2Normalize]
```

**Forward pass**:
```python
def call(context_features, training=False):
  segments = []
  
  # Process each feature pair
  for feature_name, embedder in segment_embedders.items():
    tokens = context_features[feature_name]
    weights = context_features[feature_name.replace("/t", "/w")]
    segment = embedder(tokens, weights, training=training)
    segments.append(segment)  # Each is 16-dim
  
  # Concatenate all segments
  concatenated = tf.concat(segments, axis=-1)
  # Shape: [batch_size, 32]  (2 segments × 16 dims)
  
  # Feed through SNN
  hidden = snn(concatenated, training=training)
  # Shape: [batch_size, 32]
  
  # Apply transformations
  output = softplus(hidden)
  output = l2_normalize(output)
  # Shape: [batch_size, 32]
  
  return output  # Context embedding
```

**Example**:
```python
Input: 
  Segment 1 (forward peers): [0.2, 0.3, ..., 0.1]  (16-dim)
  Segment 2 (backward peers): [0.1, 0.5, ..., 0.3]  (16-dim)

Concatenated: [0.2, 0.3, ..., 0.1, 0.1, 0.5, ..., 0.3]  (32-dim)

After SNN: [0.4, 0.2, ..., 0.7]  (32-dim)

After Softplus: [0.6, 0.4, ..., 0.9]  (all positive)

After L2-norm: [0.15, 0.10, ..., 0.22]  (unit length)

Final context embedding: [0.15, 0.10, ..., 0.22]  (32-dim, unit length)
```

### 4.2 Action Tower

**Purpose**: Process action features → action embeddings

**Key difference from context**: Produces **multiple** action embeddings (one per action)

**Input**: Action features (ragged/variable-length)
```python
actions = [
  {
    "doc_access/prin/t": [alice, bob],
    "doc_access/prin/w": [alice:2.0, bob:1.0]
  },
  {
    "doc_access/prin/t": [carol],
    "doc_access/prin/w": [carol:3.0]
  }
]
# 2 actions in this batch
```

**Architecture**: Same as context tower (SegmentEmbedders + SNN)

**Forward pass**:
```python
def call(action_features, training=False):
  all_action_embeddings = []
  
  for action in action_features:
    segments = []
    for feature_name, embedder in segment_embedders.items():
      segment = embedder(action[feature_name], action[weights], training)
      segments.append(segment)
    
    concatenated = tf.concat(segments, axis=-1)
    embedding = snn(concatenated, training=training)
    embedding = transform(embedding)  # Softplus + L2-norm
    all_action_embeddings.append(embedding)
  
  return tf.ragged.constant(all_action_embeddings)
  # Shape: RaggedTensor with shape [num_contexts, None, 32]
```

**Example output**:
```python
Context 1 (Alice) has 3 actions:
  action_1_embedding: [0.12, 0.34, ..., 0.18]
  action_2_embedding: [0.21, 0.15, ..., 0.29]
  action_3_embedding: [0.09, 0.42, ..., 0.11]

Context 2 (Bob) has 1 action:
  action_4_embedding: [0.31, 0.08, ..., 0.25]

Result: RaggedTensor([
  [[0.12, 0.34, ..., 0.18],  # Alice's actions
   [0.21, 0.15, ..., 0.29],
   [0.09, 0.42, ..., 0.11]],
  [[0.31, 0.08, ..., 0.25]]   # Bob's action
])
```

### 4.3 The SNN (Siamese Neural Network)

**Purpose**: Learn non-linear transformations of the concatenated segments into the final embedding space

**Why "Siamese"?**: The same SNN architecture is used for both context and action towers (they share the structure, not the weights). This ensures embeddings from both towers live in comparable spaces.

**Architecture**: Feed-forward network with ReLU activations

**Why this architecture?**

1. **ReLU activation**: Introduces non-linearity without vanishing gradients
   - Allows learning complex feature interactions (e.g., "if forward_peers AND backward_peers overlap, then...")
   - Fast to compute (just max(0, x))

2. **Hidden layers**: Create intermediate representations
   - Input: Raw concatenated segments (mechanical combination)
   - Hidden: Learned feature combinations (semantic patterns)
   - Output: Final embedding (behavioral signature)

**Configuration** (from config.textproto):
```textproto
snn {
  layer_sizes: 24
}
```

**Expands to**:
```python
layers = [
  Dense(24, activation='relu'),  # Hidden layer
  Dense(32, activation=None)      # Output layer (embedding_dims)
]
```

**With dropout**:
```python
layers = [
  Dense(24, activation='relu'),
  Dropout(0.05),  # dropout_neurons from config
  Dense(32, activation=None)
]
```

**Multiple hidden layers** (if configured):
```textproto
snn {
  layer_sizes: 48
  layer_sizes: 24
}
```

**Becomes**:
```python
layers = [
  Dense(48, activation='relu'),
  Dropout(0.05),
  Dense(24, activation='relu'),
  Dropout(0.05),
  Dense(32, activation=None)  # Always ends with embedding_dims
]
```

**What happens if you change the architecture?**

**Adding more layers** (e.g., [48, 24] instead of [24]):
- ✅ **Pros**: Can learn more complex patterns, higher model capacity
- ❌ **Cons**: More parameters → slower training, risk of overfitting, needs more data
- **When to use**: If you have complex feature interactions and lots of training data

**Removing layers** (e.g., no hidden layers):
- ✅ **Pros**: Fast training, less overfitting risk, works with less data
- ❌ **Cons**: Can only learn linear combinations (weighted sum of segments)
- **When to use**: If segments already capture most information, or limited training data

**Changing layer size** (e.g., 48 instead of 24):
- Larger (48): More capacity, can capture finer distinctions → better if not overfitting
- Smaller (12): Simpler model, forces learning of only most important patterns → better if overfitting

**Rule of thumb**: Start with one hidden layer of size = (input_dim + output_dim) / 2
- Input: 32-dim (concatenated segments)
- Output: 32-dim (embedding)
- Suggested hidden: 32 dims
- Facade uses 24 dims (slightly smaller → regularization effect)

**Why Facade uses a shallow network**:
- Segments already contain rich information (peer networks)
- Deep networks risk overfitting (limited malicious data)
- Shallow networks are easier to interpret and debug

---

## 5. Scoring Function

### Why Dot Product for Similarity?

**Geometric intuition**: Embeddings are points on a hypersphere (due to L2 normalization). Dot product measures the angle between them.

**Visual analogy in 2D**:
```
         context
            ↑
            |  ← small angle
            | ↗ action
            |/
  ──────────┼──────────
            |
```
- Small angle → high dot product → similar behaviors → **NORMAL**
- Large angle (90°) → zero dot product → unrelated behaviors → **ANOMALOUS**
- Opposite (180°) → negative dot product → contradictory behaviors → **VERY ANOMALOUS**

**Why this works for insider threat detection**:

1. **Alice (SWE) accesses code repository**:
   - Context embedding: Points toward "software development" region
   - Action embedding: Points toward "software development" region
   - Small angle → high similarity → normal

2. **Alice (SWE) accesses HR salary database**:
   - Context embedding: Points toward "software development" region
   - Action embedding: Points toward "HR/finance" region
   - Large angle → low similarity → anomalous

**The embedding space learns behavioral clusters**:
- Region 1: Software engineering activities
- Region 2: Finance/HR activities
- Region 3: Marketing activities
- etc.

Normal access = context and action point to the same region
Anomalous access = context and action point to different regions

---

### 5.1 Computing Scores

**Facade uses SF_OMDOT** (One Minus Dot product) **by default**:

```python
context_embedding = [0.15, 0.10, 0.22, ...]  # 32-dim, unit length
action_embedding = [0.12, 0.34, 0.18, ...]   # 32-dim, unit length

# Compute dot product (cosine similarity)
dot_product = context_embedding · action_embedding
# dot_product = 0.15*0.12 + 0.10*0.34 + 0.22*0.18 + ...
# dot_product ∈ [-1, 1]  (because both are unit vectors)

# Apply SF_OMDOT scoring (configured in config.textproto)
score = 1 - dot_product
# score ∈ [0, 2]
```

**Interpretation** (with SF_OMDOT):
- **score close to 0**: Embeddings aligned (dot product = 1) → similar → **NORMAL access**
- **score close to 1**: Embeddings orthogonal (dot product = 0) → dissimilar → **ANOMALOUS**
- **score close to 2**: Embeddings opposite (dot product = -1) → very dissimilar → **VERY ANOMALOUS**

**Why SF_OMDOT?**
- Standard anomaly detection convention: **high score = anomaly**
- Implements cosine distance (distance in embedding space)
- Intuitive: "anomaly score" should increase with anomalousness
- Mathematically: score = 1 - cos(θ) where θ is the angle between embeddings

**Alternative**: SF_DOT (vanilla dot product) where high score = normal, low = anomalous
- Less intuitive for security (low score = bad)
- But mathematically equivalent (just flipped threshold)

### 5.2 Batch Scoring

**Context tower output**:
```python
context_embeddings = [..., ..., ...]  # Shape: [batch_size, 32]
# batch_size = 100 (from config)
```

**Action tower output**:
```python
action_embeddings = RaggedTensor([...])  # Shape: [batch_size, None, 32]
# Total actions across batch: e.g., 250
action_embeddings_flat = action_embeddings.flat_values  # Shape: [250, 32]
```

**Match contexts to actions**:
```python
# Which context does each action belong to?
row_ids = action_embeddings.value_rowids()
# row_ids = [0, 0, 0, 1, 2, 2, ...]  (action_1, action_2, action_3 belong to context_0, etc.)

# Get corresponding context embeddings
context_for_each_action = tf.gather(context_embeddings, row_ids)
# Shape: [250, 32]

# Compute scores
scores = tf.reduce_sum(
  context_for_each_action * action_embeddings_flat,
  axis=-1
)
# Shape: [250]  (one score per action)
```

---

## 6. Loss Function: Pairwise Contrastive Learning

### Why Contrastive Learning?

**The fundamental challenge**: We don't have labeled examples of "anomalous" behavior (insider attacks are rare and varied).

**Traditional approaches won't work**:
- ❌ **Supervised learning**: Needs labeled anomalies (which we don't have)
- ❌ **Autoencoder**: Learns to reconstruct inputs (but can also reconstruct anomalies well)
- ❌ **One-class SVM**: Draws boundary around normal (but struggles with high-dimensional data)

**Contrastive learning solution**: Learn what's normal by comparing normal examples

**Key insight**: If Alice actually accessed Document X, then:
- ✅ (Alice's context, Document X) should be **similar** (they matched in reality)
- ❌ (Alice's context, Random Document Y) should be **dissimilar** (unlikely to happen)

**Why this works**:
- We have unlimited normal examples (all logged accesses)
- We can synthesize negative examples (random mismatches)
- The model learns: "normal = context matches action, anomalous = context doesn't match action"

**At inference**: Real anomalous access looks like a synthetic negative pair → high score!

---

### 6.1 The Training Objective

**Goal**: Learn embeddings such that:
- **Positive pairs** (context, action that actually occurred) have **high similarity** (low score)
- **Negative pairs** (context, action that didn't occur) have **low similarity** (high score)

### 6.2 Generating Training Pairs

**From a minibatch**:

**Positive pairs** (actual occurrences):
```
(alice_context, alice_action_1)  ← Alice actually did this
(alice_context, alice_action_2)
(bob_context, bob_action_1)
```

**Negative pairs** (synthetic):
```
(alice_context, bob_action_1)    ← Alice didn't do Bob's action
(bob_context, alice_action_1)    ← Bob didn't do Alice's action
```

**Configuration** (from config.textproto):
```textproto
synthetic_positives_strategy {
  random_sample_within_minibatch {
    contrastive_scores_per_query: 4
    positive_instances_weight_factor: 1.0
  }
}
```

**contrastive_scores_per_query = 4**: For each action, sample 4 negative contexts

**Example**:
```
Action: alice_action_1

Positive pair:
  (alice_context, alice_action_1)  score = 0.85, label = 1

Negative pairs (sampled from batch):
  (bob_context, alice_action_1)    score = 0.23, label = 0
  (carol_context, alice_action_1)  score = 0.15, label = 0
  (diana_context, alice_action_1)  score = 0.31, label = 0
  (eve_context, alice_action_1)    score = 0.08, label = 0
```

### 6.3 Pairwise Huber Loss

**What is a margin?**

Imagine we want the model to predict positive pairs have high similarity (low score) and negative pairs have low similarity (high score). But by *how much* should they differ?

**Margin = the minimum acceptable gap between positive and negative scores**

Example with soft_margin = 0.05:
```
Positive score: 0.20
Negative score: 0.30
Gap: 0.30 - 0.20 = 0.10 > 0.05 ✓ (margin satisfied, no penalty)

Positive score: 0.28
Negative score: 0.30
Gap: 0.30 - 0.28 = 0.02 < 0.05 ✗ (margin violated, apply penalty!)
```

**Why margins matter**:
- Without margin: Model might make positive=0.499, negative=0.501 → technically correct but useless
- With margin: Forces positive < negative - 0.05 → clear separation → robust decisions

**Configuration**:
```textproto
loss_function {
  pairwise_huber {
    soft_margin: 0.05      # Minimum gap to aim for
    hard_margin: 0.02      # Transition point for Huber loss
    norm_push: 1.0         # Weight for normalization penalty
  }
}
```

**Concept**: Penalize violations of the margin, with Huber smoothing

**For each pair** (positive vs negative):
```python
positive_score = 0.85
negative_score = 0.23

violation = soft_margin - (positive_score - negative_score)
violation = 0.05 - (0.85 - 0.23) = 0.05 - 0.62 = -0.57

if violation > 0:  # Margin violated
  if violation < hard_margin:
    loss = 0.5 * violation^2  # Quadratic (Huber)
  else:
    loss = hard_margin * violation - 0.5 * hard_margin^2  # Linear
else:  # Margin satisfied
  loss = 0
```

**In this case**: violation < 0 → loss = 0 (margin satisfied)

**Example with violation**:
```python
positive_score = 0.52
negative_score = 0.50

violation = 0.05 - (0.52 - 0.50) = 0.05 - 0.02 = 0.03

violation > 0 and violation > hard_margin (0.02):
  loss = 0.02 * 0.03 - 0.5 * 0.02^2
  loss = 0.0006 - 0.0002 = 0.0004
```

**Total loss**: Sum over all pairs in minibatch

### 6.4 Why Pairwise Huber?

**Why "Huber"?** Named after Peter Huber, combines best of squared and absolute error:

```
Loss
  ↑
  |     ╱  (linear - robust to outliers)
  |    ╱
  |   ╱╲   (quadratic - smooth gradients)
  |  ╱  ╲
  | ╱    ╲___
  |╱_________╲___
  └──────────────→ Violation size
       ↑
   hard_margin
```

**Comparison with alternatives**:

**Squared error** (e.g., (positive - negative + margin)²):
- ✅ Smooth, nice gradients
- ❌ Huge penalties for outliers → unstable training
- ❌ Example: One bad negative pair (violation=1.0) dominates over 100 good pairs

**Absolute error** (e.g., |positive - negative + margin|):
- ✅ Robust to outliers
- ❌ Non-smooth at zero → unstable gradients → slow convergence

**Huber loss** (quadratic → linear transition):
- ✅ Smooth gradients for small violations (fast learning on most examples)
- ✅ Robust to outliers (large violations don't dominate)
- ✅ Best of both worlds!

**Why this matters for Facade**:
- Training data has noisy labels (some "normal" pairs might actually be anomalous)
- Outliers shouldn't derail training
- But we still want smooth optimization for typical examples

**Advantages**:
1. **Smooth gradients** (quadratic for small violations) → fast convergence
2. **Robust to outliers** (linear for large violations) → stable training
3. **Margin-based** (only penalizes when margin violated) → efficient learning
4. **Relative scoring** (learns to rank, not absolute values) → generalizes better

### 6.5 Principal Compatibility

**Important constraint**: Don't create negative pairs from same principal

**Configuration** checks:
```python
if action_principal == context_principal:
  # Don't use as negative pair
  # (Alice's context shouldn't be negative for Alice's action)
  weight = 0.0  # Exclude from loss
else:
  weight = 1.0  # Include in loss
```

**Why**: Alice accessing resource X under Alice's context is always a valid occurrence (even if rare)

---

## 7. Training Process

### 7.1 Data Loading

**Input files**:
```
train.tfrecord  (ContextualizedActions → tf.SequenceExample)
vocab.tfrecord  (Vocabulary)
```

**Dataset pipeline**:
```python
dataset = tf.data.TFRecordDataset("train.tfrecord")
dataset = dataset.map(parse_sequence_example)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(batch_size=100)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

### 7.2 Training Loop

**Configuration**:
```textproto
training_hyperparameters {
  batch_size: 100
  training_examples: 80000
}
```

**Compute epochs**:
```python
dataset_size = 10000  # Total training examples
training_examples = 80000
epochs = 80000 / 10000 = 8
```

**Per epoch**:
```python
for batch in dataset:
  # Forward pass
  context_embeddings = context_tower(batch.context_features, training=True)
  action_embeddings = action_tower(batch.action_features, training=True)
  
  # Compute scores
  scores = score_function(context_embeddings, action_embeddings)
  
  # Compute loss
  loss = pairwise_huber_loss(scores, labels)
  
  # Backward pass
  gradients = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(gradients, trainable_variables))
```

### 7.3 Optimizer: AdamW

**Configuration**:
```textproto
optimizer {
  adam_w {
    weight_decay: 0.0004
    beta_1: 0.9
    beta_2: 0.999
    epsilon: 1e-07
    global_clipnorm: 10
  }
}
```

**AdamW**: Adam with decoupled weight decay (regularization)

**Parameters**:
- **weight_decay**: L2 regularization strength (prevents overfitting)
- **beta_1**: Momentum for gradients (0.9 standard)
- **beta_2**: Momentum for squared gradients (0.999 standard)
- **global_clipnorm**: Gradient clipping (prevents explosions)

### 7.4 Learning Rate Schedule: OneCycle

**Configuration**:
```textproto
learning_rate_schedule {
  one_cycle {
    peak_learning_rate: 0.001
    learning_rate_rampup_factor: 917.7064
    learning_rate_rampdown_factor: 6626420000.0
    rampup: 0.12393822
    interpolation: I_LINEAR
  }
}
```

**OneCycle schedule**:
```
LR
 ^
 |     ╱╲
 |    ╱  ╲___
 |   ╱       ╲___
 |  ╱            ╲___
 | ╱                 ╲___
 |╱____________________╲___
 └────────────────────────> Steps
   ↑                    ↑
 rampup             training_examples
```

**Phases**:
1. **Warmup** (12.4% of training): LR increases from ~0.000001 to 0.001
2. **Peak** (brief): LR = 0.001
3. **Decay** (87.6% of training): LR decreases from 0.001 to ~0.0000002

**Why OneCycle?**

**Phase 1 - Warmup**: Start with tiny learning rate, gradually increase
- **Why**: Random initialization means gradients are unreliable early on
- **Effect**: Avoids divergence, lets model find general direction
- **Analogy**: Walking slowly in the dark until your eyes adjust

**Phase 2 - Peak**: High learning rate for exploring the loss landscape
- **Why**: Large steps help escape local minima, find better regions
- **Effect**: Rapid improvement in loss
- **Analogy**: Running once you can see where you're going

**Phase 3 - Decay**: Gradually decrease learning rate
- **Why**: Fine-tune the solution, settle into a good minimum
- **Effect**: Smooth convergence, better final performance
- **Analogy**: Slowing down as you approach your destination

**Benefits compared to constant learning rate**:
- ✅ Faster training (aggressive peak phase)
- ✅ Better final performance (fine-tuning decay phase)
- ✅ More stable (warmup prevents early divergence)
- ✅ Often eliminates need for manual LR tuning

**Benefits compared to step decay** (common alternative):
- ✅ Smoother optimization (no sudden jumps)
- ✅ Fewer hyperparameters (no need to choose step points)
- ✅ Works well with limited training time (one cycle = one training run)

---

## 8. Evaluation Metrics

### 8.1 Metrics Configuration

```textproto
evaluation {
  metrics_fpr_thresholds: 1.0
  metrics_fpr_thresholds: 0.1
  metrics_fpr_thresholds: 0.01
}
```

**Metrics computed**:
- **TPR @ FPR**: True positive rate at specific false positive rates
- **AUC @ FPR**: Area under ROC curve up to specific FPR
- **Precision @ FPR**: Precision at specific FPR thresholds

**Why these metrics?**: 
- Insider threat detection requires **low FPR** (< 1%)
- Traditional accuracy not meaningful (vast majority of accesses are normal)

### 8.2 Validation Loop

**Every N training steps**:
```python
for validation_batch in validation_dataset:
  # Forward pass (training=False, no dropout)
  context_emb = context_tower(batch.context, training=False)
  action_emb = action_tower(batch.actions, training=False)
  
  # Generate synthetic negatives
  positive_pairs, negative_pairs = generate_contrastive_pairs(...)
  
  # Compute scores
  scores = score_function(...)
  
  # Update metrics (no gradient computation)
  metrics.update_state(labels, scores)

# Compute final metrics
tpr_at_1pct_fpr = metrics.tpr_at_fpr(0.01)
```

---

## 9. Model Saving

### 9.1 SavedModel Format

**After training completes**:
```python
model.save("sample/model/export/final/")
```

**Saved components**:
```
sample/model/export/final/
├── saved_model.pb         # Model architecture
├── variables/
│   ├── variables.data-00000-of-00001
│   └── variables.index
└── assets/
```

**Includes**:
- Embedding table weights
- SNN layer weights
- Model configuration
- Serving signature (for inference)

### 9.2 Checkpoint Saving

**During training** (every N epochs):
```python
callbacks = [
  SaveLatestModels(every_n_epochs=5)
]

model.fit(dataset, callbacks=callbacks)
```

**Creates**:
```
sample/model/intermediate_saved_model/
├── epoch_5/
├── epoch_10/
├── epoch_15/
...
```

**Use case**: Resume training from checkpoint if interrupted

---

## 10. Key Training Insights

### 10.1 Why Metric Learning Works

**The fundamental difference from classification**:

**Traditional classification** (e.g., logistic regression, decision trees):
- Question: "Is this access normal or anomalous?" (binary decision)
- How: Learn boundary in feature space separating normal from anomalous
- Problem: Needs labeled anomalies to learn the boundary
- **Facade's challenge**: We don't have labeled anomalies!

**Metric learning** (what Facade uses):
- Question: "How similar is this context to this action?" (similarity score)
- How: Learn embedding space where similar things are close, dissimilar things are far
- Training: Use normal examples to learn what "similar" means
- Inference: Anomalies naturally score high (dissimilar from normal patterns)

**Why this enables zero-shot generalization**:

Example: New resource appears (Document Z, never seen in training)
- **Classification approach**: Can't classify (never learned about Document Z)
- **Metric learning approach**: Embed Document Z based on who accessed it → compare to contexts → works!

**Concrete example**:
```
Training: Alice (SWE) accesses repo1, repo2, repo3
Model learns: Alice's context ≈ "software development"

Inference: New repo4 appears
- Who accessed it? Other SWEs → Embedded near "software development"
- Alice accesses repo4 → High similarity → Low score → Normal ✓

Inference: Alice accesses finance_db (new)
- Who accessed it? Finance team → Embedded near "finance"
- Alice accesses finance_db → Low similarity → High score → Anomalous! ✓
```

**Key insight**: Model doesn't need to have seen Document Z during training. It just needs to understand "what kind of document is this" (from access patterns) and "what kind of person is Alice" (from social context).

### 10.2 The Role of Negative Sampling

**Why are negative examples necessary?**

Imagine training with only positive pairs (real accesses):
- Model learns: "Make all (context, action) pairs similar"
- Trivial solution: Map everything to the same point!
- Result: All scores = 0 → Can't detect anomalies

**Visual analogy**:
```
Training with only positives:
  All embeddings collapse to one point
  ●●●●●●●●●  →  ●
  (can't distinguish anything)

Training with positives + negatives:
  Embeddings spread out into meaningful clusters

     SWE cluster      Finance cluster
        ●●●●              ○○○○
        ●●●●              ○○○○

  SWE context near SWE actions → low score
  SWE context far from Finance actions → high score
```

**What negative sampling does**:

1. **Creates contrast**: "Alice's context should be similar to docs she accesses, but dissimilar to random docs"

2. **Structures the space**: Forces the model to learn meaningful distances
   - Positive pairs → pull embeddings together
   - Negative pairs → push embeddings apart
   - Balance → organized embedding space with semantic clusters

3. **Teaches "normal" implicitly**: By seeing what doesn't match, model learns what does match

**Concrete training example**:
```
Positive pair: (Alice_context, repo_she_accessed) → minimize distance
Negative pairs:
  (Alice_context, finance_doc) → maximize distance
  (Alice_context, hr_doc) → maximize distance
  (Alice_context, marketing_doc) → maximize distance

Result: Alice's context embedding moves toward "software development" region
        and away from "finance", "hr", "marketing" regions
```

**Why random sampling works**: Most random pairs are truly dissimilar
- Probability that random (context, action) match is tiny
- Even if some negative pairs should be positive (noise), vast majority are correct
- Huber loss makes training robust to this noise

### 10.3 Social Network Features Enable Transfer

**The cold start problem with history-only features**:

**Scenario**: Alice accesses a new document she's never seen before
```
Using only Alice's access history:
  "Has Alice accessed this document before?" → No
  "Has Alice accessed similar documents before?" → Can't tell (doc is new)
  Model: ¯\_(ツ)_/¯ (no information to make decision)
```

This is catastrophic because:
- New resources appear constantly in a corporate environment
- Attackers often target resources they've never accessed before
- Cold start = blind spot for detection

**The social network solution**:

Instead of asking "What did Alice access before?", ask:
1. "Who does Alice work with?" → Bob, Carol (her peers)
2. "What do Bob and Carol access?" → Software repos, design docs
3. "Does this new resource look like what Bob/Carol access?" → Yes/No

**Concrete example**:
```
Context: Alice works with Bob, Carol (both SWEs)
         Bob accesses repos X, Y, Z
         Carol accesses repos A, B, C

New resource: repo_new
- Who accessed it? Other SWEs (similar to Bob/Carol)
- Alice accesses repo_new → Consistent with peer behavior → Low score ✓

New resource: finance_spreadsheet
- Who accessed it? Finance team (dissimilar from Bob/Carol)
- Alice accesses it → Inconsistent with peer behavior → High score! ✓
```

**Why this works**:
- **Guilt by association** (positive): Alice probably needs similar access as her collaborators
- **Peer-based inference**: New resource characterized by "who uses it" (not just its name)
- **Robust to novelty**: Works even when both Alice AND the resource are new to the system

**The key insight**: Social networks provide *context* that's independent of specific resources
- Traditional: "Does Alice access docs like this?" (resource-dependent)
- Facade: "Do people like Alice access docs like this?" (role-dependent)

This is why Document 4 emphasizes bipartite graphs so heavily - they're the mechanism that extracts these peer relationships!

---

## 11. Summary

This document explained Facade's neural architecture and training process. Here are the key takeaways:

### Why These Architectural Choices?

1. **Two separate towers**: Context = "who is this person", Action = "what is this resource"
   - Allows independent updates and caching
   - Forces semantic separation (person vs resource features)

2. **Concatenation + SNN**: Combines segments, learns interactions
   - Concatenation: mechanical combination of features
   - SNN: learns semantic patterns (e.g., "peer overlap indicates similar role")
   - Shallow network: sufficient for rich peer-based features, avoids overfitting

3. **Softplus + L2 normalization**: Standardizes embeddings
   - Softplus: ensures positive values (interpretable as "strengths")
   - L2 norm: projects to hypersphere (only direction matters, not magnitude)
   - Prevents shortcuts (can't use magnitude to cheat)

4. **Dot product scoring**: Measures angular similarity
   - Embeddings are directions in behavior space
   - Similar directions (small angle) = normal access
   - Different directions (large angle) = anomalous access

5. **Contrastive learning with Huber loss**: Learns from normal data only
   - Positive pairs (real accesses) pull embeddings together
   - Negative pairs (synthetic mismatches) push embeddings apart
   - Huber: smooth gradients + robust to outliers

6. **OneCycle learning rate**: Efficient training in one pass
   - Warmup: prevents early divergence
   - Peak: rapid learning and exploration
   - Decay: fine-tuning and convergence

### The Big Picture

**Facade learns an embedding space** where:
- Each point represents a "behavioral role" (SWE, finance, HR, etc.)
- Contexts and actions from the same role cluster together
- Normal access = context near action (low score)
- Anomalous access = context far from action (high score)

**Why this works for insider threat detection**:
- No labeled anomalies needed (learns from normal data)
- Generalizes to new resources (uses peer-based features)
- Single-event precision (embedding-level similarity, not volume)
- Robust over time (role-based, not resource-specific)

### What You Can Tune

**For better accuracy** (if not overfitting):
- Add SNN hidden layers or increase layer size
- Increase embedding dimensions
- Train longer

**For better generalization** (if overfitting):
- Remove SNN layers or decrease layer size
- Decrease embedding dimensions
- Increase dropout rates
- Use stronger weight decay

**For faster training**:
- Decrease embedding dimensions
- Remove SNN hidden layers
- Reduce batch size (but may hurt quality)

### Next Steps

**Document 7** will explain how to use the trained model for inference - loading the model, generating scores, and interpreting results.

**Document 8** will provide a step-by-step tutorial for recreating the entire Facade pipeline from scratch.

**Ready for Document 7: Inference and Scoring**
