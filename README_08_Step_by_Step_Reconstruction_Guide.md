# Document 8: Step-by-Step Reconstruction Guide

## Introduction

This document provides a **complete, hands-on tutorial** for rebuilding Facade from scratch. We'll walk through an end-to-end example with actual data, showing you exactly how to:

1. Prepare raw action and context data
2. Create configuration files
3. Generate training datasets
4. Train the model
5. Run inference
6. Interpret results

By the end, you'll have a working Facade system and understand every step of the pipeline.

---

## 1. Prerequisites

### 1.1 Environment Setup

**Python version**: >= 3.10

**Install dependencies**:
```bash
pip install -r requirements.in
```

**Key packages**:
- TensorFlow >= 2.13
- Protocol Buffers
- NumPy, Pandas
- absl-py

**Verify installation**:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

### 1.2 Repository Structure

```
facade/
├── action/              # Action featurization
├── context/             # Context featurization (bipartite graphs)
├── batch/               # Dataset generation, training, inference
├── model/               # Neural network architecture
├── pipelines/           # Feature engineering pipelines
├── protos/              # Protocol buffer definitions
├── common/              # Utilities
└── sample/              # Example data and configs
    ├── action.tfrecord
    ├── context.tfrecord
    ├── directive.textproto
    └── config.textproto
```

---

## 2. Example Scenario

### 2.1 The Organization

**Company**: TechCorp (50 employees)
**Systems**: Document management, code review platform
**Threat**: Detect unauthorized document access and code exfiltration

### 2.2 Raw Data Available

**Action logs** (document access):
```
Timestamp: 2024-07-01 09:15:00, User: alice@techcorp.com, Action: read, Resource: design_doc_123
Timestamp: 2024-07-01 10:30:00, User: bob@techcorp.com, Action: read, Resource: design_doc_123
Timestamp: 2024-07-02 14:20:00, User: alice@techcorp.com, Action: read, Resource: salary_data_456
```

**Context logs** (code reviews):
```
Timestamp: 2024-06-15, Change: 789, Author: alice@techcorp.com, Reviewers: [bob@techcorp.com, carol@techcorp.com]
Timestamp: 2024-06-20, Change: 790, Author: bob@techcorp.com, Reviewers: [alice@techcorp.com, diana@techcorp.com]
```

---

## 3. Step 1: Create Action Protos

### 3.1 Action Data Format

**Goal**: Convert raw logs to Action protos

**Action proto structure** (review from Document 2):
```protobuf
message Action {
  string principal = 1;
  string resource_id = 2;
  Timestamp time = 3;
  bytes id = 4;
  string source_type = 5;
  repeated Feature features = 6;
}
```

### 3.2 Create Actions Script

**Create**: `scripts/create_actions.py`

```python
import tensorflow as tf
from datetime import datetime
from protos import action_pb2
from protos import timestamp_pb2
from common import source_data

def create_action(principal, resource_id, timestamp, source_type):
    """Create an Action proto from raw data."""
    action = action_pb2.Action()
    action.principal = principal
    action.resource_id = resource_id
    action.source_type = source_type
    
    # Convert timestamp
    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    action.time.seconds = int(dt.timestamp())
    
    # Generate unique ID
    action.id = f"{principal}_{resource_id}_{timestamp}".encode('utf-8')
    
    # Create features
    # Feature 1: Principal
    prin_feature = action.features.add()
    prin_feature.name = f"{source_type}/prin/t"
    token = prin_feature.bag_of_weighted_words.tokens.add()
    token.token = principal.encode('utf-8')
    token.weight = 1.0
    
    # Feature 2: Resource
    res_feature = action.features.add()
    res_feature.name = f"{source_type}/res/t"
    token = res_feature.bag_of_weighted_words.tokens.add()
    token.token = resource_id.encode('utf-8')
    token.weight = 1.0
    
    return action

# Example usage
actions = [
    create_action("alice@techcorp.com", "design_doc_123", "2024-07-01 09:15:00", "doc_access"),
    create_action("bob@techcorp.com", "design_doc_123", "2024-07-01 10:30:00", "doc_access"),
    create_action("alice@techcorp.com", "salary_data_456", "2024-07-02 14:20:00", "doc_access"),
    create_action("carol@techcorp.com", "code_change_789", "2024-07-01 11:00:00", "code_review"),
    create_action("diana@techcorp.com", "code_change_790", "2024-07-02 15:30:00", "code_review"),
]

# Write to TFRecord
with tf.io.TFRecordWriter('data/action.tfrecord') as writer:
    for action in actions:
        writer.write(action.SerializeToString())

print(f"Created {len(actions)} actions")
```

**Run**:
```bash
python scripts/create_actions.py
```

**Output**: `data/action.tfrecord` containing Action protos

### 3.3 Verify Actions

```python
# Read back and verify
import tensorflow as tf
from protos import action_pb2

reader = tf.data.TFRecordDataset(['data/action.tfrecord'])
for raw_record in reader.take(1):
    action = action_pb2.Action()
    action.ParseFromString(raw_record.numpy())
    print(action)
```

**Expected output**:
```
principal: "alice@techcorp.com"
resource_id: "design_doc_123"
time { seconds: 1719824100 }
id: "alice@techcorp.com_design_doc_123_2024-07-01 09:15:00"
source_type: "doc_access"
features {
  name: "doc_access/prin/t"
  bag_of_weighted_words {
    tokens { token: "alice@techcorp.com" weight: 1.0 }
  }
}
features {
  name: "doc_access/res/t"
  bag_of_weighted_words {
    tokens { token: "design_doc_123" weight: 1.0 }
  }
}
```

---

## 4. Step 2: Create Context Protos

### 4.1 Context Data Format

**Goal**: Convert collaboration data to Context protos with PeerAttributes

**Context proto structure**:
```protobuf
message Context {
  string principal = 1;
  Timestamp snapshot_time = 2;
  bytes id = 3;
  string source_type = 4;
  repeated PeerAttribute peer_attributes = 5;
}

message PeerAttribute {
  string name = 1;
  repeated bytes peer_ids = 2;
  repeated float intensities = 3;
  Timestamp timestamp = 4;
}
```

### 4.2 Create Contexts Script

**IMPORTANT**: You only need to record individual participation (e.g., "Alice attended meeting X"), **NOT** explicit relationships (e.g., "Alice met with Bob"). The bipartite graph algorithm discovers peer relationships automatically!

**Create**: `scripts/create_contexts.py`

```python
import tensorflow as tf
from datetime import datetime
from protos import context_pb2
from protos import timestamp_pb2

def create_context_simple(principal, snapshot_time, source_type, activities):
    """Create a Context proto from individual activity participation.

    You only record WHAT the principal did, not WHO they interacted with.
    The graph algorithm discovers peer relationships automatically!

    PeerAttribute structure:
    - name: The attribute type (e.g., "meeting_id", "change_number")
    - peer_ids: The attribute VALUES (e.g., [b"meeting_789"], [b"CL_12345"])
    - intensities: Weights for each value (e.g., [1.0])
    - timestamp: When this activity occurred
    """
    context = context_pb2.Context()
    context.principal = principal
    context.source_type = source_type

    # Snapshot time
    dt = datetime.strptime(snapshot_time, "%Y-%m-%d %H:%M:%S")
    context.snapshot_time.seconds = int(dt.timestamp())

    # Unique ID
    context.id = f"{principal}_{snapshot_time}".encode('utf-8')

    # Add peer attributes (individual activities only!)
    for activity in activities:
        peer_attr = context.peer_attributes.add()
        peer_attr.name = activity['type']  # e.g., "meeting_id", "change_number"

        # CRITICAL: peer_ids contains the attribute VALUES, not other people!
        # The graph algorithm will find people who share these values
        peer_attr.peer_ids.append(activity['value'])  # e.g., b"meeting_789"
        peer_attr.intensities.append(activity.get('weight', 1.0))

        # Add timestamp
        event_dt = datetime.strptime(activity['timestamp'], "%Y-%m-%d %H:%M:%S")
        peer_attr.timestamp.seconds = int(event_dt.timestamp())

        # Direction (for asymmetric relationships like author vs reviewer)
        if 'direction' in activity:
            peer_attr.direction = activity['direction']

    return context

# Example: Meeting attendance (symmetric relationship)
# Alice, Bob, and Carol all attended meeting_789
# We create 3 separate contexts (one per person)
# The graph algorithm will automatically discover they're all peers!

alice_context = create_context_simple(
    principal="alice@techcorp.com",
    snapshot_time="2024-07-01 00:00:00",
    source_type="calendar",
    activities=[
        {
            'type': 'meeting_id',
            'value': b'meeting_789',  # Alice attended this meeting
            'timestamp': "2024-06-30 14:00:00",
            'weight': 1.0
        }
    ]
)

bob_context = create_context_simple(
    principal="bob@techcorp.com",
    snapshot_time="2024-07-01 00:00:00",
    source_type="calendar",
    activities=[
        {
            'type': 'meeting_id',
            'value': b'meeting_789',  # Bob attended the same meeting
            'timestamp': "2024-06-30 14:00:00",
            'weight': 1.0
        }
    ]
)

carol_context = create_context_simple(
    principal="carol@techcorp.com",
    snapshot_time="2024-07-01 00:00:00",
    source_type="calendar",
    activities=[
        {
            'type': 'meeting_id',
            'value': b'meeting_789',  # Carol attended the same meeting
            'timestamp': "2024-06-30 14:00:00",
            'weight': 1.0
        }
    ]
)

# When the bipartite graph algorithm runs, it will automatically discover:
# - Alice's peers: Bob, Carol (they shared meeting_789)
# - Bob's peers: Alice, Carol (they shared meeting_789)
# - Carol's peers: Alice, Bob (they shared meeting_789)
# You never had to explicitly write "Alice met with Bob"!

# Example: Code review (asymmetric relationship)
# Alice authored CL_12345, Bob reviewed it
# Use D_FORWARD for author, D_BACKWARD for reviewer

alice_code_review = create_context_simple(
    principal="alice@techcorp.com",
    snapshot_time="2024-07-01 00:00:00",
    source_type="code_review",
    activities=[
        {
            'type': 'change_number',
            'value': b'CL_12345',  # Alice authored this CL
            'timestamp': "2024-06-15 10:00:00",
            'weight': 1.0,
            'direction': context_pb2.PeerAttribute.D_FORWARD  # Author
        }
    ]
)

bob_code_review = create_context_simple(
    principal="bob@techcorp.com",
    snapshot_time="2024-07-01 00:00:00",
    source_type="code_review",
    activities=[
        {
            'type': 'change_number',
            'value': b'CL_12345',  # Bob reviewed this CL
            'timestamp': "2024-06-15 10:00:00",
            'weight': 2.0,  # Higher weight (Bob reviewed it thoroughly)
            'direction': context_pb2.PeerAttribute.D_BACKWARD  # Reviewer
        }
    ]
)

# The graph algorithm will discover:
# - Alice's forward peers: Bob (people who review Alice's code)
# - Bob's backward peers: Alice (people whose code Bob reviews)
# This happens because both Alice and Bob have CL_12345 in their peer_ids!

contexts = [alice_context, bob_context, carol_context, alice_code_review, bob_code_review]

# Write to TFRecord
with tf.io.TFRecordWriter('data/context.tfrecord') as writer:
    for context in contexts:
        writer.write(context.SerializeToString())

print(f"Created {len(contexts)} contexts")
print("The bipartite graph algorithm will automatically discover peer relationships!")
```

**Run**:
```bash
python scripts/create_contexts.py
```

**Output**: `data/context.tfrecord`

### 4.3 Understanding How the Bipartite Graph Works

**What we created** (individual participation records):

**Alice's context**:
```python
peer_attributes: [
  {name: "meeting_id", peer_ids: [b"meeting_789"], intensities: [1.0]}
]
```

**Bob's context**:
```python
peer_attributes: [
  {name: "meeting_id", peer_ids: [b"meeting_789"], intensities: [1.0]}
]
```

**Carol's context**:
```python
peer_attributes: [
  {name: "meeting_id", peer_ids: [b"meeting_789"], intensities: [1.0]}
]
```

**What the bipartite graph algorithm discovers** (during `fold.py` execution):

1. **Build bipartite graph**:
   ```
   Alice ←→ meeting_789 ←→ Bob
   Alice ←→ meeting_789 ←→ Carol
   Bob ←→ meeting_789 ←→ Carol
   ```

2. **Two-hop random walk** (Alice's perspective):
   ```
   Alice → meeting_789 → Bob    (Alice's peer: Bob)
   Alice → meeting_789 → Carol  (Alice's peer: Carol)
   ```

3. **Result**: Alice's peer network = {Bob, Carol}

**Why this is powerful**:
- You only record "Alice attended meeting_789" (simple fact)
- The algorithm discovers "Alice's collaborators are Bob, Carol" (relationship)
- When Alice accesses a document, the model checks if Bob/Carol accessed similar documents
- **No manual relationship tracking needed!**

---

## 5. Step 3: Create directive.textproto

### 5.1 Basic Configuration

**Create**: `config/directive.textproto`

```textproto
# Action sources
action_sources {
  source_type: "doc_access"
  
  # History features: What docs did this user access?
  token_features {
    # Principal (username)
    token_feature_name: "doc_access/prin/t"
    weight_feature_name: "doc_access/prin/w"
    history_settings {
      count_transform: CT_IDENTITY
    }
  }
  token_features {
    # Resource (document ID)
    token_feature_name: "doc_access/res/t"
    weight_feature_name: "doc_access/res/w"
    history_settings {
      count_transform: CT_IDENTITY
    }
  }
}

# Context sources (for bipartite graph)
context_sources {
  source_type: "code_review"
  
  # Bipartite graph settings
  token_features {
    token_feature_name: "code_review/change_number_f/t"  # Forward: Who did Alice work with?
    weight_feature_name: "code_review/change_number_f/w"
    graph_settings {
      peer_attribute_name: "change_number"
      graph_walk_mode: GWM_FORWARD
      node_distance: 2
      pruning {
        top_k: 20
      }
    }
  }
  token_features {
    token_feature_name: "code_review/change_number_b/t"  # Backward: Who worked with Alice?
    weight_feature_name: "code_review/change_number_b/w"
    graph_settings {
      peer_attribute_name: "change_number"
      graph_walk_mode: GWM_BACKWARD
      node_distance: 2
      pruning {
        top_k: 20
      }
    }
  }
}

# Dataset parameters
dataset_parameters {
  max_action_sources: 10
  max_feature_size: 100
  max_features_per_source: 2
  max_history_tokens_per_feature: 50
  max_context_tokens_per_feature: 50
  vocab_min_count: 2
}
```

### 5.2 Key Configuration Choices

**History features**:
- `doc_access/prin/t`: Track which user performed actions
- `doc_access/res/t`: Track which resources were accessed

**Graph features** (the innovation):
- `change_number_f`: Alice's code review collaborators → their doc accesses
- `change_number_b`: People who collaborated with Alice → their doc accesses

**Pruning**:
- `top_k: 20`: Keep top 20 peers by intensity

---

## 6. Step 4: Create config.textproto

### 6.1 Model Configuration

**Create**: `config/config.textproto`

```textproto
# Action tower architecture
action_name_to_architecture {
  key: "doc_access"
  value {
    # Map features to embeddings
    segment_feature_names: "doc_access/prin/t"
    segment_feature_names: "doc_access/res/t"
  }
}

# Context tower architecture (uses bipartite graph features)
context_architecture {
  segment_feature_names: "code_review/change_number_f/t"
  segment_feature_names: "code_review/change_number_b/t"
}

# Token embeddings
token_embedding_name_to_config {
  key: "action_username"
  value { dimensions: 16 }
}
token_embedding_name_to_config {
  key: "action_resource"
  value { dimensions: 16 }
}
token_embedding_name_to_config {
  key: "context_username"
  value { dimensions: 16 }
}

# Segment embedders
segment_reductions {
  segment_weight_scaling: WS_IDENTITY
  segment_weight_normalization: WN_L2
  token_embedding_name: "action_username"
  token_feature_name: "doc_access/prin/t"
  intensity_feature_name: "doc_access/prin/w"
}
segment_reductions {
  segment_weight_scaling: WS_IDENTITY
  segment_weight_normalization: WN_L2
  token_embedding_name: "action_resource"
  token_feature_name: "doc_access/res/t"
  intensity_feature_name: "doc_access/res/w"
}
segment_reductions {
  segment_weight_scaling: WS_IDENTITY
  segment_weight_normalization: WN_L2
  token_embedding_name: "context_username"
  token_feature_name: "code_review/change_number_f/t"
  intensity_feature_name: "code_review/change_number_f/w"
}
segment_reductions {
  segment_weight_scaling: WS_IDENTITY
  segment_weight_normalization: WN_L2
  token_embedding_name: "context_username"
  token_feature_name: "code_review/change_number_b/t"
  intensity_feature_name: "code_review/change_number_b/w"
}

# SNN architecture
snn {
  layer_sizes: 24
}

# Embeddings dimension
embedding_dims: 32

# Scoring
scoring_function: SF_OMDOT

# Loss function
loss_function {
  pairwise_huber {
    soft_margin: 0.05
    hard_margin: 0.02
    norm_push: 1.0
  }
}

# Synthetic negatives
synthetic_positives_strategy {
  random_sample_within_minibatch {
    contrastive_scores_per_query: 4
    positive_instances_weight_factor: 1.0
  }
}

# Optimizer
optimizer {
  adam_w {
    weight_decay: 0.0004
    beta_1: 0.9
    beta_2: 0.999
    epsilon: 1e-07
    global_clipnorm: 10
  }
}

# Learning rate
learning_rate_schedule {
  one_cycle {
    peak_learning_rate: 0.001
    learning_rate_rampup_factor: 917.7064
    learning_rate_rampdown_factor: 6626420000.0
    rampup: 0.12393822
    interpolation: I_LINEAR
  }
}

# Training parameters
training_hyperparameters {
  batch_size: 100
  training_examples: 10000
}

# Evaluation
evaluation {
  metrics_fpr_thresholds: 1.0
  metrics_fpr_thresholds: 0.1
  metrics_fpr_thresholds: 0.01
}
```

### 6.2 Configuration Summary

**Embeddings**: 16-dim token embeddings → 32-dim segment embeddings
**Architecture**: 2 segments per tower → concatenate (32-dim) → SNN (24 hidden) → 32-dim output
**Training**: 100 examples/batch, 10000 total examples = 100 epochs
**Loss**: Pairwise Huber with 0.05 soft margin

---

## 7. Step 5: Generate Training Dataset

### 7.1 Run Dataset Maker

```bash
python -m batch.dataset_maker_main \
  --directive=config/directive.textproto \
  --start_time="2024-07-01 00:00:00" \
  --end_time="2024-07-08 00:00:00" \
  --action_path=data/action.tfrecord \
  --context_path=data/context.tfrecord \
  --output_dir=data/training/
```

### 7.2 What Happens

**Step 1**: Load actions and contexts
```
Loading actions from data/action.tfrecord...
Loaded 5 actions
Loading contexts from data/context.tfrecord...
Loaded 2 contexts
```

**Step 2**: History featurization
```
Computing history features for each principal...
alice@techcorp.com: 2 actions
bob@techcorp.com: 1 action
carol@techcorp.com: 1 action
diana@techcorp.com: 1 action
```

**Step 3**: Bipartite graph computation
```
Building bipartite graph from peer_attributes...
Graph nodes: 4 principals
Graph edges: 4 relationships
Computing 2-hop random walks...
alice@techcorp.com forward neighbors: bob, carol, diana (via Bob)
bob@techcorp.com forward neighbors: alice, diana
```

**Step 4**: Merger
```
Merging context features with actions...
Created 5 ContextualizedActions
```

**Step 5**: Vocabulary building
```
Building vocabularies...
action_username: 4 unique tokens
action_resource: 5 unique tokens
context_username: 4 unique tokens
```

**Step 6**: Write outputs
```
Writing to data/training/train.tfrecord...
Writing vocabularies to data/training/vocab.tfrecord...
Done!
```

### 7.3 Verify Output

**Check files**:
```bash
ls -lh data/training/
```

**Expected**:
```
train.tfrecord      # ContextualizedActions → tf.SequenceExample
vocab.tfrecord      # Vocabularies
```

**Read one example**:
```python
import tensorflow as tf

dataset = tf.data.TFRecordDataset(['data/training/train.tfrecord'])
for raw_record in dataset.take(1):
    example = tf.train.SequenceExample()
    example.ParseFromString(raw_record.numpy())
    print(example)
```

**Expected structure**:
```
context {
  feature {
    key: "principal"
    value { bytes_list { value: "alice@techcorp.com" } }
  }
}
feature_lists {
  feature_list {
    key: "doc_access/prin/t"
    value { feature { bytes_list { value: ["alice@techcorp.com"] } } }
  }
  feature_list {
    key: "doc_access/res/t"
    value { feature { bytes_list { value: ["design_doc_123"] } } }
  }
  feature_list {
    key: "code_review/change_number_f/t"
    value { feature { bytes_list { value: ["bob@techcorp.com", "carol@techcorp.com"] } } }
  }
  feature_list {
    key: "code_review/change_number_f/w"
    value { feature { float_list { value: [2.0, 1.0] } } }
  }
  ...
}
```

**This shows**: Alice's action with her peer network features!

---

## 8. Step 6: Train the Model

### 8.1 Run Training

```bash
python -m model.train_main \
  --train_file=data/training/train.tfrecord \
  --vocab_file=data/training/vocab.tfrecord \
  --model_config=config/config.textproto \
  --model_dir=models/facade_v1/
```

### 8.2 Training Output

**Initialization**:
```
Loading config from config/config.textproto...
Loading vocabularies from data/training/vocab.tfrecord...
Vocabulary sizes:
  action_username: 4
  action_resource: 5
  context_username: 4
Creating model...
Total parameters: 1,234
```

**Training loop**:
```
Epoch 1/100
Loss: 0.4523, TPR@1%FPR: 0.12
Epoch 2/100
Loss: 0.3891, TPR@1%FPR: 0.23
...
Epoch 50/100
Loss: 0.1234, TPR@1%FPR: 0.78
...
Epoch 100/100
Loss: 0.0567, TPR@1%FPR: 0.92
```

**Final metrics**:
```
Training complete!
Final metrics:
  TPR @ 1% FPR: 0.92
  TPR @ 0.1% FPR: 0.87
  AUC @ 1% FPR: 0.0095
```

**Model saved**:
```
Saving model to models/facade_v1/export/final/
Model export complete!
```

### 8.3 Monitor Training (Optional)

**If you want TensorBoard**:
```bash
tensorboard --logdir=models/facade_v1/logs/
```

**View in browser**: http://localhost:6006

**Metrics to watch**:
- Loss should decrease
- TPR@1%FPR should increase
- Learning rate should follow OneCycle pattern

---

## 9. Step 7: Run Inference

### 9.1 Prepare Test Data

**Create new actions** (for July 8-15):
```python
# scripts/create_test_actions.py
test_actions = [
    create_action("alice@techcorp.com", "design_doc_123", "2024-07-08 09:00:00", "doc_access"),  # Normal
    create_action("alice@techcorp.com", "salary_data_999", "2024-07-08 23:45:00", "doc_access"),  # Anomalous!
    create_action("bob@techcorp.com", "design_doc_123", "2024-07-09 10:00:00", "doc_access"),    # Normal
]

with tf.io.TFRecordWriter('data/test_action.tfrecord') as writer:
    for action in test_actions:
        writer.write(action.SerializeToString())
```

**Use same contexts** (or updated ones with more recent data)

### 9.2 Run Inference

```bash
python -m batch.inference_main \
  --directive=config/directive.textproto \
  --start_time="2024-07-08 00:00:00" \
  --end_time="2024-07-15 00:00:00" \
  --action_path=data/test_action.tfrecord \
  --context_path=data/context.tfrecord \
  --output_file=data/scores.tfrecord \
  --model_config=config/config.textproto \
  --model_dir=models/facade_v1/
```

### 9.3 Inference Output

```
Loading model from models/facade_v1/export/final/...
Computing features...
Running inference...
Processed 3 actions
Writing scores to data/scores.tfrecord...
Done!
```

---

## 10. Step 8: Analyze Scores

### 10.1 Read Scores

```bash
python -m batch.read_scores_main \
  --score_file=data/scores.tfrecord \
  --top_n=10
```

**Output**:
```
principal: "alice@techcorp.com"
action_type: "doc_access"
resource_id: "salary_data_999"
action_id: "alice@techcorp.com_salary_data_999_2024-07-08 23:45:00"
score: 0.89

principal: "alice@techcorp.com"
action_type: "doc_access"
resource_id: "design_doc_123"
action_id: "alice@techcorp.com_design_doc_123_2024-07-08 09:00:00"
score: 0.12

principal: "bob@techcorp.com"
action_type: "doc_access"
resource_id: "design_doc_123"
action_id: "bob@techcorp.com_design_doc_123_2024-07-09 10:00:00"
score: 0.08
```

### 10.2 Interpretation

**Score 0.89** (Alice accessing salary_data_999):
- **Why high?**: 
  - Alice never accessed salary data before (history)
  - Alice's peers (Bob, Carol) never accessed it (social network)
  - Late night access (23:45) - unusual time
- **Conclusion**: **ALERT** - Potential insider threat

**Score 0.12** (Alice accessing design_doc_123):
- **Why low?**:
  - Alice accessed design_doc_123 before (in training data)
  - Bob accessed it (peer similarity)
  - Normal work hours
- **Conclusion**: Normal behavior

**Score 0.08** (Bob accessing design_doc_123):
- **Why low?**:
  - Bob's history includes design docs
  - Bob collaborated with Alice who accessed this doc
- **Conclusion**: Normal behavior

### 10.3 Set Threshold

**Based on scores**:
```python
threshold = 0.70  # Alert on scores > 0.70

alerts = [s for s in scores if s.score > threshold]
print(f"Generated {len(alerts)} alerts")
# Output: Generated 1 alert (Alice's salary_data access)
```

---

## 11. Advanced: Adding More Features

### 11.1 Time-of-Day Feature

**Modify action creation**:
```python
def create_action_with_time_features(principal, resource_id, timestamp, source_type):
    action = create_action(principal, resource_id, timestamp, source_type)
    
    # Add hour-of-day feature
    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    hour = dt.hour
    
    time_feature = action.features.add()
    time_feature.name = f"{source_type}/hour/t"
    token = time_feature.bag_of_weighted_words.tokens.add()
    token.token = f"hour_{hour}".encode('utf-8')
    token.weight = 1.0
    
    return action
```

**Update directive.textproto**:
```textproto
action_sources {
  source_type: "doc_access"
  
  token_features {
    token_feature_name: "doc_access/hour/t"
    weight_feature_name: "doc_access/hour/w"
    history_settings {
      count_transform: CT_IDENTITY
    }
  }
  # ... existing features
}
```

**Update config.textproto**:
```textproto
action_name_to_architecture {
  key: "doc_access"
  value {
    segment_feature_names: "doc_access/prin/t"
    segment_feature_names: "doc_access/res/t"
    segment_feature_names: "doc_access/hour/t"  # New!
  }
}

token_embedding_name_to_config {
  key: "action_time"
  value { dimensions: 8 }
}

segment_reductions {
  token_embedding_name: "action_time"
  token_feature_name: "doc_access/hour/t"
  intensity_feature_name: "doc_access/hour/w"
  # ... other settings
}
```

**Now**: Model learns "normal" hours for each user

---

## 12. Common Issues and Solutions

### 12.1 Issue: "No actions found in time range"

**Cause**: Start/end times don't match your data

**Solution**: Check action timestamps
```python
for action in actions:
    print(f"Action time: {action.time.seconds}")
```

### 12.2 Issue: "Vocabulary too small"

**Cause**: `vocab_min_count` too high, filters out rare tokens

**Solution**: Lower `vocab_min_count` in directive
```textproto
dataset_parameters {
  vocab_min_count: 1  # Was 2
}
```

### 12.3 Issue: "All scores are 0.5"

**Cause**: Model didn't train properly

**Solutions**:
- Increase `training_examples`
- Check that you have enough data diversity
- Verify loss is decreasing during training

### 12.4 Issue: "Graph computation slow"

**Cause**: Too many peers, large graphs

**Solutions**:
- Reduce `top_k` in graph pruning
- Reduce `node_distance` from 2 to 1
- Filter contexts to recent time windows

---

## 13. Production Checklist

### 13.1 Before Deployment

- [ ] **Sufficient training data**: At least 10,000 examples
- [ ] **Diverse action types**: Multiple resource types
- [ ] **Rich context data**: Multiple peer relationship types
- [ ] **Validation metrics**: TPR@1%FPR > 0.7
- [ ] **Threshold tuning**: Based on validation data, not training data
- [ ] **Alert investigation process**: SOC team trained

### 13.2 Monitoring

- [ ] **Score distribution tracking**: Daily mean/median
- [ ] **Alert volume**: Number of alerts per day
- [ ] **False positive rate**: Based on analyst feedback
- [ ] **Model staleness**: Retrain monthly

### 13.3 Scaling Considerations

**For large organizations** (1000+ employees):
- **Data volume**: May need distributed processing (Spark, Beam)
- **Graph computation**: Cache frequently-used subgraphs
- **Model serving**: Use TensorFlow Serving or similar
- **Real-time inference**: Stream processing (Kafka, Flink)

---

## 14. Complete Example: End-to-End Script

### 14.1 Unified Script

**Create**: `run_full_pipeline.sh`

```bash
#!/bin/bash
set -e

echo "=== Facade Full Pipeline ==="

# Step 1: Create data
echo "Step 1: Creating action and context data..."
python scripts/create_actions.py
python scripts/create_contexts.py

# Step 2: Generate training dataset
echo "Step 2: Generating training dataset..."
python -m batch.dataset_maker_main \
  --directive=config/directive.textproto \
  --start_time="2024-07-01 00:00:00" \
  --end_time="2024-07-08 00:00:00" \
  --action_path=data/action.tfrecord \
  --context_path=data/context.tfrecord \
  --output_dir=data/training/

# Step 3: Train model
echo "Step 3: Training model..."
python -m model.train_main \
  --train_file=data/training/train.tfrecord \
  --vocab_file=data/training/vocab.tfrecord \
  --model_config=config/config.textproto \
  --model_dir=models/facade_v1/

# Step 4: Create test data
echo "Step 4: Creating test data..."
python scripts/create_test_actions.py

# Step 5: Run inference
echo "Step 5: Running inference..."
python -m batch.inference_main \
  --directive=config/directive.textproto \
  --start_time="2024-07-08 00:00:00" \
  --end_time="2024-07-15 00:00:00" \
  --action_path=data/test_action.tfrecord \
  --context_path=data/context.tfrecord \
  --output_file=data/scores.tfrecord \
  --model_config=config/config.textproto \
  --model_dir=models/facade_v1/

# Step 6: Display results
echo "Step 6: Top anomalous actions:"
python -m batch.read_scores_main \
  --score_file=data/scores.tfrecord \
  --top_n=5

echo "=== Pipeline Complete! ==="
```

**Run everything**:
```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

---

## 15. Next Steps

### 15.1 Expand Your Dataset

**More action types**:
- Email sends
- File downloads
- VPN logins
- Database queries

**More context sources**:
- Email communication networks
- Project membership
- Organizational hierarchy

### 15.2 Tune Your Model

**Experiment with**:
- Different embedding dimensions
- More SNN layers
- Different loss margins
- Learning rate schedules

### 15.3 Evaluate Thoroughly

**Metrics to compute**:
- Precision-recall curves
- Per-user performance
- Per-resource-type performance
- Temporal stability

---

## 16. Summary

### What We Built

1. **Raw data** → Action and Context protos
2. **Configuration** → directive.textproto + config.textproto
3. **Dataset generation** → ContextualizedActions with bipartite graph features
4. **Training** → Neural network learning embeddings
5. **Inference** → Scoring new actions
6. **Analysis** → Identifying anomalies

### Key Takeaways

- **Bipartite graphs** enable generalization to new resources
- **Metric learning** learns similarity, not classification
- **Configuration** controls feature engineering and model architecture
- **End-to-end pipeline** from logs to alerts

### You Now Know How To

✅ Create Action and Context protos from raw logs
✅ Configure directive.textproto for feature engineering
✅ Configure config.textproto for model architecture
✅ Generate training datasets with bipartite graph features
✅ Train a Facade model with TensorFlow
✅ Run inference and interpret scores
✅ Deploy a production insider threat detection system

---

## Document 9 Preview

The final document will explore **extending Facade** with a resource-centric view:

**Question**: "Given Resource K, is it normal for Principal A to access it, given the history of other Principals who accessed it previously?"

**Approach**: Construct a second bipartite graph (Principal ↔ Resource ↔ Principal) and combine with the existing context-based approach.

**Ready for Document 9: Extending Facade - Resource-Centric View**
