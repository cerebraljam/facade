# Document 1: Overview and Conceptual Foundation

## Introduction

Facade is a deep learning-based insider threat detection system that answers a deceptively simple question:

> **"Is it normal for Principal A to access Resource K at this moment in time, given A's historical behavioral context?"**

This document establishes the conceptual foundation for understanding how Facade works and why its approach is effective.

---

## 1. The Core Problem: Detecting Anomalous Access

### 1.1 What Makes Access "Anomalous"?

Traditional security systems rely on rules and signatures:
- "User X should never access database Y"
- "Access outside business hours is suspicious"
- "Downloads exceeding N gigabytes are flagged"

These approaches fail because:
1. **Static rules can't capture behavioral nuance** - A researcher might legitimately access unusual data
2. **High false positive rates** - Many legitimate behaviors look suspicious in isolation
3. **Easy to evade** - Attackers can stay under thresholds
4. **Don't scale** - Requires manual rule creation for each resource/user combination

### 1.2 Facade's Approach: Learning What's Normal

Instead of defining rules, Facade learns from historical data:
- **What resources does Principal A typically access?**
- **Who are Principal A's peers and collaborators?**
- **What do people in A's social network typically do?**
- **Is this new access consistent with A's established patterns?**

The key insight: **Anomalies are deviations from learned behavioral norms, not violations of explicit rules.**

---

## 2. The Social Network Hypothesis

### 2.1 Why Social Networks Matter

Facade's effectiveness comes from a critical observation:

> **People who work together behave similarly.**

Consider these scenarios:

**Scenario 1: Normal Access**
- Alice is a data scientist on the ML team
- Alice regularly collaborates with Bob and Carol (code reviews, meetings)
- Bob and Carol frequently access dataset X for their projects
- **When Alice accesses dataset X, it's probably normal** - she's doing similar work to her peers

**Scenario 2: Anomalous Access**
- Dave works in HR
- Dave's collaborators (Eve, Frank) work on employee records and payroll
- Dave's social network NEVER accesses engineering datasets
- **When Dave suddenly accesses dataset X, it's highly unusual** - this is outside his peer group's normal behavior

### 2.2 The Bipartite Graph Foundation

Facade models social relationships as **bipartite graphs**:

```
Principals         "Middle Nodes"        Principals
   (People)        (Shared Activities)    (People)

    Alice  -------- Change #123 -------- Bob
     |  \                                /  |
     |   \--------  Change #456 --------/   |
     |                                      |
     \------------- Change #789 ----------- Carol
```

**Two-hop traversal** finds peers:
- Alice → Change #123 → Bob (Alice's reviewer)
- Alice → Change #456 → Bob (another shared review)
- Alice → Change #789 → Carol (yet another connection)

**Result**: Bob and Carol are Alice's "code review peers" - they form her social network in that context.

### 2.3 Multiple Context Sources

Facade can combine multiple social graphs:
- **Code reviews** → engineering collaborators
- **Calendar events** → meeting participants
- **Document access** → information-sharing relationships
- **Project memberships** → team affiliations

Each graph provides a different lens on "who is similar to whom."

---

## 3. The Two-Tower Architecture

### 3.1 Overview

Facade uses a **metric learning** approach with two neural networks (towers):

```
                    ┌─────────────────────┐
                    │  CONTEXT TOWER      │
                    │  (Who is Principal A│
                    │   and their social  │
Input: Context      │   network?)         │
Features      ─────>│                     │──────> Context
(Social network,    │  Deep Neural Net    │       Embedding
peer info)          └─────────────────────┘       (vector)
                                                      │
                                                      │
                                                      ▼
                                                  ┌────────┐
                                                  │SCORING │
Input: Action                                     │FUNCTION│◄─── Score
Features      ─────>┌─────────────────────┐      │(dot    │    (how similar?)
(What resource      │  ACTION TOWER       │      │product)│
 is accessed?)      │  (What is being     │──>   └────────┘
                    │   accessed?)        │    Action
                    │                     │    Embedding
                    │  Deep Neural Net    │    (vector)
                    └─────────────────────┘
```

### 3.2 What Are Embeddings?

**Embeddings** are vector representations that capture meaning:
- A context embedding represents "who Principal A is" (their role, team, behavior patterns)
- An action embedding represents "what is being accessed" (the resource and its characteristics)

**Key property**: Similar contexts and actions map to nearby points in embedding space.

**Example in 2D** (real embeddings are higher dimensional):
```
Context Embedding Space:

    Data Scientists
         o o
        o Alice o
         o o
                      
                              Accountants
                                o o
                               o o o


Action Embedding Space:

    ML Datasets              Financial Reports
      x x x                      x x
      x Dataset_X x              x x x
      x x x
```

### 3.3 The Scoring Function

Facade uses **SF_OMDOT** (One Minus Dot) scoring:

```
score = 1 - <context_embedding, action_embedding>

Where <·,·> is the dot product and both vectors are L2-normalized
```

**This is cosine distance**, a standard anomaly detection metric.

**Interpretation**:
- **High score** (close to 1.0): Context and action are dissimilar → **ANOMALOUS access**
- **Low score** (close to 0.0): Context and action are similar → **NORMAL access**

**Geometric interpretation**: 
- Cosine similarity (dot product) close to 1 → vectors aligned → similar → **distance near 0** → normal
- Cosine similarity close to 0 → vectors orthogonal → dissimilar → **distance near 1** → anomalous
- Cosine similarity close to -1 → vectors opposite → very dissimilar → **distance near 2** → very anomalous

**Why this convention?**
- Matches standard anomaly detection: high score = alarm
- Cosine distance is interpretable: distance in embedding space
- Avoids confusion: "anomaly score" should be high for anomalies

**Alternative**: Some configurations use **SF_DOT** (vanilla dot product) where high scores mean normal, but Facade's sample config uses **SF_OMDOT**

---

## 4. How Facade Learns "Normal"

### 4.1 Training Data Generation

Facade observes historical access patterns:

```
Time: Jan 1 - Mar 31 (Training Period)

Alice (Data Scientist):
  - Accessed ML_dataset_1, ML_dataset_2, ML_dataset_3
  - Code reviews with Bob, Carol (also data scientists)
  
Dave (HR):
  - Accessed payroll_db, employee_records
  - Meetings with Eve, Frank (also HR)
```

### 4.2 The Contrastive Loss Function

Training uses **metric learning with contrastive loss**:

**Positive pairs** (should have high scores):
- Alice's context + ML_dataset_1 (she actually accessed it)
- Bob's context + ML_dataset_2 (he actually accessed it)

**Negative pairs** (should have low scores):
- Alice's context + payroll_db (she never accessed it)
- Dave's context + ML_dataset_1 (he never accessed it)

The model learns to:
- **Pull together** contexts and actions that co-occur in history
- **Push apart** contexts and actions that don't co-occur

### 4.3 Why This Works

After training, the embedding space has structure:

```
Embedding Space (conceptual):

  [Data Science Cluster]
      Alice's context ← close together
      Bob's context
      ML_dataset_1
      ML_dataset_2
      
                                [HR Cluster]
                                    Dave's context ← close together
                                    Eve's context
                                    payroll_db
                                    employee_records
```

**At inference time**:
- New action: Alice accesses ML_dataset_3 (new dataset, not seen in training)
- Alice's context embedding is in the "Data Science Cluster"
- If ML_dataset_3 is similar to other ML datasets → its embedding will be nearby
- **High score** → flagged as normal
- If Alice suddenly accesses payroll_db → embeddings are far apart
- **Low score** → flagged as anomalous

---

## 5. The Role of Social Networks (Critical!)

### 5.1 Beyond Individual History

**Naive approach**: "Has Alice accessed this resource before?"
- Problem: Doesn't generalize to new resources
- Problem: Can't detect privilege escalation (accessing resources you *could* access but normally don't)

**Facade's approach**: "Do people in Alice's peer network access resources like this?"
- Alice has never seen ML_dataset_3, BUT
- Alice's code review peers (Bob, Carol) regularly access similar ML datasets
- The model learns: "Alice is like Bob and Carol" (via social network features)
- Therefore: "Alice accessing ML_dataset_3 is consistent with her peer group's behavior"

### 5.2 How Social Features Enter the Model

The **context features** include:

1. **Direct features**: Alice's own historical access patterns
2. **Social features**: Who are Alice's peers? (from bipartite graph traversal)

Example context feature for Alice:
```
Context features at time T:
  - code_review_peers: [Bob, Carol, Diana]  (from 2-hop graph walk)
  - These peers are represented as weighted embeddings
  - The weights reflect how "close" they are (based on collaboration frequency)
```

The **Context Tower** processes these features:
1. Embeds each peer (Bob, Carol, Diana)
2. Computes weighted average: `0.5*Bob_emb + 0.3*Carol_emb + 0.2*Diana_emb`
3. Combines with other features
4. Produces final context embedding for Alice

**Result**: Alice's context embedding captures not just what Alice does, but what her social network does.

### 5.3 Generalization Power

This is why Facade generalizes:

**Training scenario**:
- Bob accessed ML_dataset_1 → model learns Bob's context is similar to ML_dataset_1
- Carol accessed ML_dataset_2 → model learns Carol's context is similar to ML_dataset_2
- Alice collaborates with Bob and Carol → Alice's context includes Bob and Carol features

**Inference scenario** (new situation):
- New dataset: ML_dataset_3 (never seen before)
- ML_dataset_3 is similar to ML_dataset_1 and ML_dataset_2
- Alice's context includes Bob and Carol features
- **The model infers**: Since Alice is like Bob/Carol, and Bob/Carol access ML datasets, Alice accessing ML_dataset_3 is normal

**Without social features**: The model would have no basis to judge - Alice never accessed ML_dataset_3 before.

**With social features**: The model knows Alice's peer group accesses ML datasets, so it's probably fine.

---

## 6. Metric Learning: The Mathematical Foundation

### 6.1 What is Metric Learning?

**Metric learning** trains a model to learn a distance function where:
- Similar items are close together
- Dissimilar items are far apart

In Facade:
- **Items**: (context, action) pairs
- **Distance metric**: Negative dot product (or equivalently, cosine distance)
- **Learning objective**: Make the distance between positive pairs small, negative pairs large

### 6.2 The Pairwise Loss

Facade uses a **pairwise loss** function:

For each training example:
1. Take a positive pair: (Alice's context, resource Alice accessed)
2. Generate negative pairs: (Alice's context, resources Alice didn't access)
3. Compute scores for all pairs
4. **Optimize**: Maximize positive score, minimize negative scores

**Huber loss variant**: Combines benefits of:
- **L2 loss**: Smooth gradients for small errors
- **L1 loss**: Robust to outliers
- **Soft margin**: Only penalize violations beyond a threshold

### 6.3 Why Metric Learning Instead of Classification?

**Classification approach** (doesn't scale):
- Output layer: one neuron per possible resource
- Problem: New resources require retraining
- Problem: Millions of possible resources → huge output layer

**Metric learning approach** (scales):
- Output: embeddings (fixed-size vectors, e.g., 32 dimensions)
- New resources: Just compute their embedding, no retraining needed
- Inference: Compare embeddings using dot product
- **Zero-shot generalization**: Works for resources never seen during training

---

## 7. Why Facade is High-Precision

### 7.1 Precision vs. Recall Trade-off

**Precision**: Of all alerts raised, how many are true threats?
**Recall**: Of all true threats, how many did we catch?

Traditional systems struggle with precision:
- Rule-based systems: High false positives (low precision)
- Analysts overwhelmed by alerts
- Real threats lost in noise

### 7.2 Facade's High-Precision Design

Facade achieves high precision through:

1. **Rich context modeling**
   - Social networks capture nuanced behavioral patterns
   - Multiple context sources (code reviews, meetings, etc.)
   - Time-aware features (recent activity weighted more)

2. **Metric learning generalization**
   - Learns patterns, not memorizes access lists
   - New resources scored accurately without training data
   - Reduces false positives on legitimate novel accesses

3. **Threshold tuning**
   - Can set very conservative thresholds
   - e.g., alert only on bottom 0.01% of scores
   - High precision: most alerts are real anomalies
   - Trade-off: May miss some attacks (lower recall)

4. **Ensemble of signals**
   - Multiple action types (doc access, DB queries, etc.)
   - Multiple context sources combined
   - Anomalous only if deviant across multiple dimensions

### 7.3 Use Cases

Given high precision, Facade works well as:

- **Last line of defense**: Final check before critical access
- **ACL recommendation**: Suggest access controls based on learned patterns
- **Account compromise detection**: Spot when credentials are used abnormally
- **Insider threat triage**: Prioritize investigations for security analysts

---

## 8. Summary: Key Takeaways

### Core Concepts

1. **Facade detects anomalies by learning behavioral norms**, not rule violations
2. **Social networks are central**: Your peers define what's normal for you
3. **Bipartite graphs** model relationships through shared activities
4. **Two-tower architecture**: Separate embeddings for context and actions
5. **Metric learning**: Learn a similarity function, not a classifier
6. **High precision** through rich context and generalization

### Why It Works

- **Generalization**: Works for new resources never seen in training
- **Nuance**: Captures complex behavioral patterns via social features
- **Scalability**: Fixed-size embeddings regardless of number of resources
- **Adaptability**: Learns from your organization's specific patterns

### The Secret Sauce

> **The bipartite graph random walk is what makes Facade special.**

Without it: Just "has Alice accessed this before?" (doesn't generalize)

With it: "Do people like Alice (based on collaboration) access things like this?" (generalizes powerfully)

### Next Steps

- **Document 2**: Understand the data model (protos)
- **Document 3**: Learn to configure Facade (directive & config files)
- **Document 4**: Deep dive into bipartite graph algorithms (**the most important**)
- **Document 5-7**: Feature engineering, model architecture, inference
- **Document 8**: Hands-on reconstruction tutorial
- **Document 9**: Implement the resource-centric extension

---

## Conceptual Questions to Test Understanding

Before moving to Document 2, ensure you can answer:

1. **Why are social networks important in Facade?**
   - Answer: They enable generalization to new resources by capturing "people like you access things like this"

2. **What is a bipartite graph in Facade's context?**
   - Answer: A graph connecting principals through shared activities (code reviews, meetings, etc.)

3. **What does the two-tower architecture do?**
   - Answer: Creates separate embeddings for contexts and actions, then scores via dot product

4. **How does metric learning differ from classification?**
   - Answer: Learns a similarity function (embeddings), not per-class decisions; enables zero-shot generalization

5. **Why is Facade high-precision?**
   - Answer: Rich context (social + temporal), metric learning generalization, conservative thresholds

6. **How would a resource-centric view complement the current principal-centric approach?**
   - Answer: Detects when a principal accesses resources outside their typical scope, even if consistent with peer behavior

---

**Ready for Document 2: Data Model and Proto Schemas**
