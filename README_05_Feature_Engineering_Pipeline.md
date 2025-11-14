# Document 5: Feature Engineering Pipeline

## Introduction

This document explains how Facade transforms raw Action and Context protos into training-ready ContextualizedActions protos with features.

**Three main pipelines**:
1. **History Featurizer**: Builds features from action history
2. **Context Featurizer**: Builds features from bipartite graphs (covered in Document 4)
3. **Merger**: Combines contexts and actions by snapshot time

---

## 1. Overview: The Complete Pipeline

### 1.1 Data Flow

```
Input:
  action.tfrecord (Action protos)
  context.tfrecord (Context protos)
  directive.textproto (configuration)

     ↓

Pipeline Stage 1: History Featurizer
  For each action:
    - Look back in time (history_duration)
    - Find who accessed this resource before
    - Create features: "prin/t" (tokens), "prin/w" (weights)

     ↓

Pipeline Stage 2: Context Featurizer
  For each snapshot time:
    - Collect contexts in lookback window
    - Build bipartite graphs (Document 4)
    - Create features: "change_number_f/t", "change_number_b/t", etc.

     ↓

Pipeline Stage 3: Merger
  For each snapshot time:
    - Match contexts to principals
    - Find actions that occurred after this snapshot
    - Combine into ContextualizedActions protos

     ↓

Output:
  train.tfrecord (ContextualizedActions protos)
  vocab.tfrecord (Vocabulary for token embeddings)
```

### 1.2 Code Entry Point

```python
# From batch/dataset_maker_main.py
contextualized_actions = batch_lib.compute_contextualized_actions(
    directive=directive,
    pipeline_type=PipelineType.TRAINING,  # or INFERENCE
    start_time=datetime(2024, 4, 1),
    end_time=datetime(2024, 7, 1),
    action_file_path="action.tfrecord",
    context_file_path="context.tfrecord"
)
```

---

## 2. History Featurizer Pipeline

### 2.1 Purpose

**Question**: "Who has accessed this resource historically?"

**Output**: Features showing historical accessors and their access frequency

### 2.2 Input

**Action protos** for a specific action type (e.g., `doc_access`):
```python
Action(
  type="doc_access",
  id=b"action_1",
  resource_id="doc_ml_guide",
  principal="alice",
  occurred_at=Timestamp(2024-06-01 10:00:00),
  history_key=b"doc_ml_guide"
)
Action(
  type="doc_access",
  id=b"action_2",
  resource_id="doc_ml_guide",
  principal="bob",
  occurred_at=Timestamp(2024-05-15 14:00:00),
  history_key=b"doc_ml_guide"
)
Action(
  type="doc_access",
  id=b"action_3",
  resource_id="doc_ml_guide",
  principal="alice",
  occurred_at=Timestamp(2024-05-01 09:00:00),
  history_key=b"doc_ml_guide"
)
```

### 2.3 Configuration (from directive)

```textproto
action_sources {
  type: "doc_access"
  history_duration { seconds: 7776000 }  # 90 days
  action_deduplication_period { seconds: 7200 }  # 2 hours
  max_values_per_feature: 1000
  min_values_per_feature: 1
}
```

### 2.4 Algorithm: Event-Based Processing

The history featurizer uses an **event-based timeline** approach:

```python
def build_history_features(actions, featurize_at_or_after, featurize_before, 
                           history_duration, deduplication_period, 
                           max_tokens_per_segment):
    # Group actions by history_key (usually = resource_id)
    actions_by_key = group_by(actions, key=lambda a: a.history_key)
    
    for history_key, action_list in actions_by_key.items():
        # Process timeline for this resource
        process_timeline(action_list, ...)
```

### 2.5 Timeline Processing Example

**Scenario**: Featurize actions on `doc_ml_guide` between 2024-06-01 and 2024-07-01

**History window**: 90 days before featurization period = from 2024-03-03

**Actions** (sorted by time):
```
2024-03-15 10:00 - alice accessed doc_ml_guide
2024-04-01 14:00 - bob accessed doc_ml_guide
2024-04-02 09:00 - alice accessed doc_ml_guide
2024-04-02 09:30 - alice accessed doc_ml_guide  ← Within dedup period
2024-05-01 11:00 - carol accessed doc_ml_guide
2024-06-15 15:00 - alice accessed doc_ml_guide  ← TO FEATURIZE
```

**Events created**:
```
Event 1: time=2024-03-15 10:00, type=ACCUMULATE, payload=[(prin, alice, +1.0)]
Event 2: time=2024-04-01 14:00, type=ACCUMULATE, payload=[(prin, bob, +1.0)]
Event 3: time=2024-04-02 09:00, type=ACCUMULATE, payload=[(prin, alice, +1.0)]
Event 4: time=2024-04-02 09:30, type=ACCUMULATE, payload=[] ← DEDUPLICATED (within 2 hours of prev)
Event 5: time=2024-05-01 11:00, type=ACCUMULATE, payload=[(prin, carol, +1.0)]
Event 6: time=2024-06-15 15:00, type=FEATURIZE, payload=action_to_featurize

# Expiration events (after history_duration)
Event 7: time=2024-06-13 10:00 (2024-03-15 + 90 days), type=ACCUMULATE, payload=[(prin, alice, -1.0)]
```

**Process events in chronological order**:

```
Time: 2024-03-15 10:00
  Event: ACCUMULATE (prin, alice, +1.0)
  State: {alice: 1.0}

Time: 2024-04-01 14:00
  Event: ACCUMULATE (prin, bob, +1.0)
  State: {alice: 1.0, bob: 1.0}

Time: 2024-04-02 09:00
  Event: ACCUMULATE (prin, alice, +1.0)
  State: {alice: 2.0, bob: 1.0}

Time: 2024-04-02 09:30
  Event: ACCUMULATE (deduplicated, no payload)
  State: {alice: 2.0, bob: 1.0} ← UNCHANGED

Time: 2024-05-01 11:00
  Event: ACCUMULATE (prin, carol, +1.0)
  State: {alice: 2.0, bob: 1.0, carol: 1.0}

Time: 2024-06-13 10:00
  Event: ACCUMULATE (prin, alice, -1.0) ← EXPIRATION
  State: {alice: 1.0, bob: 1.0, carol: 1.0}

Time: 2024-06-15 15:00
  Event: FEATURIZE
  Current state BEFORE this action: {alice: 1.0, bob: 1.0, carol: 1.0}
  
  Generate feature:
    name: "doc_access/prin/t"
    tokens: [alice, bob, carol]
    
    name: "doc_access/prin/w"  
    tokens: [alice, bob, carol]
    weights: [1.0, 1.0, 1.0]
```

**Key insight**: The feature for action_6 shows historical state BEFORE that action occurred (doesn't include the current action).

### 2.6 Temporal Deduplication

**Purpose**: Prevent over-counting of rapid repeated accesses

**Example** (deduplication_period = 2 hours):
```
10:00 - Alice accesses doc_X  ✓ COUNTED
10:15 - Alice accesses doc_X  ✗ SKIPPED (within 2 hours)
10:45 - Alice accesses doc_X  ✗ SKIPPED (within 2 hours)
12:01 - Alice accesses doc_X  ✓ COUNTED (> 2 hours since 10:00)
```

**Implementation**:
```python
class HistoryFeaturizer:
    def __init__(self, deduplication_period, max_tokens):
        self.last_seen = {}  # {(key, token): last_seen_time}
        self.segments = {}   # {key: OnlineHeaviestItems}
    
    def accumulate(self, event_time, tokens):
        for key, token in tokens:
            lookup_key = token.token + ("+" if token.weight > 0 else "-")
            
            # Check if seen recently
            if lookup_key in self.last_seen[key]:
                if self.last_seen[key][lookup_key] + dedup_period >= event_time:
                    continue  # Skip this token
            
            # Update last seen time
            self.last_seen[key][lookup_key] = event_time
            
            # Add to segment
            self.segments[key].upsert(token.token, token.weight)
```

### 2.7 Top-K Selection

**Purpose**: Limit features to most frequent accessors

**Configuration**: `max_values_per_feature = 1000`

**Example**:
```
doc_popular has 5,000 unique accessors in history window
Counts: {alice: 50, bob: 45, ..., xavier: 1, yolanda: 1, zoe: 1}

Keep top 1000 by weight:
  Result: {alice: 50, bob: 45, ..., user_1000: 2}
  Drop: {user_1001: 2, ..., zoe: 1}
```

**Implementation**: Uses `OnlineHeaviestItems` class (maintains top-k heap)

### 2.8 Output Features

**For each action**, two features are generated:

**Feature 1: Tokens** (`/t` suffix)
```python
Feature(
  name="doc_access/prin/t",
  bag_of_weighted_words=WeightedTokens(
    tokens=[
      WeightedToken(token=b"alice", weight=1.0),
      WeightedToken(token=b"bob", weight=1.0),
      WeightedToken(token=b"carol", weight=1.0)
    ]
  )
)
```

**Feature 2: Weights** (`/w` suffix)
```python
Feature(
  name="doc_access/prin/w",
  bag_of_weighted_words=WeightedTokens(
    tokens=[
      WeightedToken(token=b"alice", weight=2.5),   # Access count (after dedup)
      WeightedToken(token=b"bob", weight=1.0),
      WeightedToken(token=b"carol", weight=1.5)
    ]
  )
)
```

**Why separate /t and /w?**:
- `/t`: Identity of accessors (looked up in embedding table)
- `/w`: Intensity of access (used for weighted averaging)

### 2.9 Filtering

**Training mode only**: Drop actions with too few historical accessors

```python
if len(feature.tokens) < min_values_per_feature:
    # Don't create FeaturizedAction for this action
    return None
```

**Example** (min_values_per_feature = 1):
```
Action on resource_new (0 historical accessors) → DROPPED in training
Action on resource_established (3 historical accessors) → KEPT
```

**Inference mode**: Keep everything (no minimum enforced)

---

## 3. Context Featurizer Pipeline

### 3.1 Purpose

**Question**: "Who are this principal's peers based on shared activities?"

**Output**: Features showing peer principals and relationship strengths

### 3.2 Input

**Context protos** for a specific context type:
```python
Context(
  type="code_review",
  principal="alice",
  valid_from=Timestamp(2024-06-01 00:00:00),
  peer_attributes=[
    PeerAttribute(name="change_number", value=b"12345", 
                  direction=D_FORWARD, weight=1.0,
                  time=Timestamp(2024-05-28 10:00:00))
  ]
)
```

### 3.3 Pipeline Structure

```python
def get_featurized_context(config, snapshot_times, tf_record_file):
    # 1. Read contexts
    contexts = read_context(config.type, tf_record_file, start_time, end_time)
    
    # 2. Build bipartite features (see Document 4)
    bipartite_features = build_bipartite_features(contexts, config, snapshot_times)
    
    return bipartite_features
```

### 3.4 Bipartite Feature Building

**Covered in Document 4**, but summary:
1. Collect contexts in lookback window for each snapshot
2. Build bipartite graphs (separate for each traversal mode)
3. Apply two-hop random walk
4. Generate features with peer tokens and weights

**Example output**:
```python
FeaturizedContext(
  valid_from=Timestamp(2024-06-01 00:00:00),
  features_per_source=[
    ContextSourceFeatures(
      source_type="code_review",
      features=[
        # Forward traversal
        Feature(
          name="code_review/change_number_f/t",
          bag_of_weighted_words=WeightedTokens(
            tokens=[
              WeightedToken(token=b"bob", weight=1.0),
              WeightedToken(token=b"carol", weight=1.0)
            ]
          )
        ),
        Feature(
          name="code_review/change_number_f/w",
          bag_of_weighted_words=WeightedTokens(
            tokens=[
              WeightedToken(token=b"bob", weight=0.65),
              WeightedToken(token=b"carol", weight=0.35)
            ]
          )
        ),
        # Backward traversal
        Feature(name="code_review/change_number_b/t", ...),
        Feature(name="code_review/change_number_b/w", ...)
      ]
    )
  ]
)
```

### 3.5 Feature Aggregation

**Multiple peer_feature_configs** → Multiple features per source

**Example**:
```textproto
context_sources {
  type: "code_review"
  peer_feature_configs { name: "change_number" ... }
  peer_feature_configs { name: "design_doc_id" ... }
}
```

**Output**:
```python
features=[
  Feature(name="code_review/change_number_f/t", ...),
  Feature(name="code_review/change_number_f/w", ...),
  Feature(name="code_review/change_number_b/t", ...),
  Feature(name="code_review/change_number_b/w", ...),
  Feature(name="code_review/design_doc_id_u/t", ...),  # Undirected
  Feature(name="code_review/design_doc_id_u/w", ...)
]
```

---

## 4. Snapshot Time Management

### 4.1 Generating Snapshot Times

**From directive**:
```textproto
dataset_parameters {
  snapshot_period { seconds: 7200 }  # 2 hours
  snapshot_time_offset { seconds: 0 }
}
```

**Code**:
```python
def generate_snapshot_times(start_time, end_time, period, offset):
    # Align to period boundaries
    aligned_start = align_to_period(start_time, period, offset)
    
    snapshots = []
    current = aligned_start
    while current < end_time:
        snapshots.append(current)
        current += period
    
    return snapshots
```

**Example**:
```
start_time: 2024-04-01 00:00:00
end_time: 2024-04-01 12:00:00
period: 2 hours
offset: 0

Generated snapshots:
  2024-04-01 00:00:00
  2024-04-01 02:00:00
  2024-04-01 04:00:00
  2024-04-01 06:00:00
  2024-04-01 08:00:00
  2024-04-01 10:00:00
```

### 4.2 Context Valid From

**Each FeaturizedContext has valid_from = snapshot time**:
```python
FeaturizedContext(
  valid_from=Timestamp(2024-04-01 02:00:00),
  features_per_source=[...]
)
```

**Meaning**: This context is valid for actions that occur ≥ 02:00:00

**Data used**: Contexts and actions from BEFORE 02:00:00 (causal consistency)

---

## 5. Merger Pipeline

### 5.1 Purpose

**Combine**:
- Featurized contexts (by principal and snapshot time)
- Featurized actions (by principal and timestamp)

**Output**: ContextualizedActions protos

### 5.2 Input

**From Context Featurizer**:
```python
contexts = [
  ("alice", FeaturizedContext(valid_from=Timestamp(2024-04-01 00:00:00), ...)),
  ("alice", FeaturizedContext(valid_from=Timestamp(2024-04-01 02:00:00), ...)),
  ("bob", FeaturizedContext(valid_from=Timestamp(2024-04-01 00:00:00), ...)),
  ...
]
```

**From History Featurizer**:
```python
actions = {
  "doc_access": [
    ("alice", FeaturizedAction(id=b"a1", occurred_at=Timestamp(2024-04-01 01:30:00), ...)),
    ("alice", FeaturizedAction(id=b"a2", occurred_at=Timestamp(2024-04-01 03:15:00), ...)),
    ("bob", FeaturizedAction(id=b"b1", occurred_at=Timestamp(2024-04-01 00:45:00), ...)),
    ...
  ]
}
```

### 5.3 Matching Algorithm

**For each action, find the correct context**:

```python
def find_snapshot_for_action(action_time, sorted_snapshots):
    # Find latest snapshot ≤ action_time
    idx = bisect.bisect_right(sorted_snapshots, action_time)
    
    if idx == 0:
        return None  # Action before first snapshot, drop it
    
    return sorted_snapshots[idx - 1]
```

**Example**:
```
Snapshots: [00:00, 02:00, 04:00, 06:00]

Action at 01:30 → snapshot 00:00 (latest ≤ 01:30)
Action at 02:00 → snapshot 00:00 (NOT 02:00, must be strictly <)
Action at 03:15 → snapshot 02:00
Action at 06:30 → snapshot 06:00
```

**Note**: Action exactly at snapshot time uses PREVIOUS snapshot (ensures context computed before action)

### 5.4 Grouping

**Group by (principal, snapshot_time)**:
```python
grouped_data = {
  ("alice", 2024-04-01 00:00:00): {
    "context": FeaturizedContext(...),
    "actions": [action1, action2, ...]
  },
  ("alice", 2024-04-01 02:00:00): {
    "context": FeaturizedContext(...),
    "actions": [action3, ...]
  },
  ...
}
```

### 5.5 Context-Only Entries

**If no actions for a context** (principal had peer relationships but performed no actions in this period):
```python
ContextualizedActions(
  principal="alice",
  context=FeaturizedContext(...),
  actions=[]  # Empty
)
```

**Purpose**: Provides negative examples **during training only**
- "Alice had context but didn't access anything"
- Model learns: low scores for context-resource pairs that don't occur
- These help the model understand what "normal inactivity" looks like

**Training vs Inference**:
- **Training**: Context-only entries are randomly downsampled but included
  - Controlled by `downsample_missing_actions()` function in `batch_lib.py`
  - Prevents over-representation of "no action" examples
- **Inference**: Context-only entries are typically not generated since we only score actual actions
  - The inference pipeline only processes actions that need scoring
  - No value in scoring "nothing happened"

### 5.6 Actions-Without-Context

**Critical behavior**: Actions are **ALWAYS DROPPED** if the principal has no context at the relevant snapshot time.

**When this happens**:
1. Action occurs before the first snapshot (temporal issue)
2. Principal has no context at any snapshot (no peer relationships discovered)

**This applies to BOTH training and inference**.

**Example**:
```
Alice has contexts at snapshots: 00:00, 02:00, 04:00
Bob has NO contexts (not in any bipartite graphs)
Eve has NO contexts (new account, no peer relationships)

Alice's action at 01:30 → ✓ matched to 00:00 snapshot
Bob's action at 01:30 → ✗ DROPPED (no context for Bob)
Eve's action at 01:30 → ✗ DROPPED (no context for Eve)
```

**Code location** (`pipelines/merger/pipeline.py`):
```python
def contextualize_actions(...):
    # Group actions with the appropriate context snapshot
    for source, principal_action_list in actions.items():
        for principal, action in principal_action_list:
            # Find the latest snapshot time ≤ action time
            idx = bisect.bisect_right(sorted_snapshots, action.occurred_at)
            
            if idx == 0:
                # Action before earliest snapshot → DROP
                continue
            
            snapshot_time = sorted_snapshots[idx - 1]
            key = (principal, snapshot_time)
            grouped_data[key]['actions'].append((source, action))
    
    # Later...
    for key, data in grouped_data.items():
        context_features = data['context_features']
        associated_actions = data['actions']
        
        # If there are no context features for this key, drop associated actions
        if not context_features:
            continue  # ← ACTIONS ARE DROPPED HERE
```

**Important implications**:

1. **New principals without peer relationships**: If Eve's account is created but has no participation in any activities that create peer_attributes (e.g., no code reviews, no meeting attendance), Facade **cannot score their actions** because there's no context to compare against.

2. **Cold start problem**: Principals need to establish peer relationships before Facade can monitor them. This requires:
   - Participating in activities that generate contexts (code reviews, meetings, etc.)
   - Waiting for at least one snapshot period to pass
   - Having those contexts processed through the bipartite graph pipeline

3. **Training vs inference**: The dropping behavior is **identical** in both modes. Context-only entries (contexts with no actions) are included in training data as negative examples, but actions without context are always dropped.

**Mitigation strategies**:

- **History-only actions**: If an action has sufficient historical data (e.g., resource accessed by many people before), the history features alone provide signal, but the action still needs a context to be scored
- **Extended lookback**: Use longer `context_lookback_duration` to capture more peer relationships from the past

### 5.7 Chunking Large Action Lists

**Configuration**:
```textproto
max_num_actions_per_contextualized_actions: 1000
```

**Problem**: A single principal might have 10,000 actions at one snapshot

**Solution**: Split into multiple ContextualizedActions protos

```python
alice_actions = [action_1, action_2, ..., action_2500]  # 2500 actions
max_per_ca = 1000

Creates 3 ContextualizedActions:
CA_1: context + actions[0:1000]
CA_2: context + actions[1000:2000]  ← Context duplicated
CA_3: context + actions[2000:2500]  ← Context duplicated
```

**Trade-off**: Duplicate context features vs. proto size limits

### 5.8 Sampling (Training Only)

**Configuration**:
```textproto
max_num_contextualized_actions_per_principal_snapshot: 100
```

**Purpose**: Prevent over-representation of very active principals

**Example**:
```
Alice at snapshot 00:00 has 150 ContextualizedActions (after chunking)
max_num_cas = 100

Training mode:
  Randomly sample 100, drop 50

Inference mode:
  Keep all 150 (no sampling)
```

### 5.9 Output Structure

**Final ContextualizedActions proto**:
```python
ContextualizedActions(
  principal="alice",
  
  context=FeaturizedContext(
    valid_from=Timestamp(2024-04-01 02:00:00),
    features_per_source=[
      ContextSourceFeatures(
        source_type="code_review",
        features=[
          Feature(name="code_review/change_number_f/t", ...),
          Feature(name="code_review/change_number_f/w", ...),
          Feature(name="code_review/change_number_b/t", ...),
          Feature(name="code_review/change_number_b/w", ...)
        ]
      )
    ]
  ),
  
  actions=[
    FeaturizedActionsBySource(
      source_type="doc_access",
      actions=[
        FeaturizedAction(
          id=b"action_1",
          resource_id="doc_ml_guide",
          occurred_at=Timestamp(2024-04-01 03:15:00),
          features=[
            Feature(name="doc_access/prin/t", ...),
            Feature(name="doc_access/prin/w", ...)
          ]
        ),
        FeaturizedAction(
          id=b"action_2",
          resource_id="doc_python_tutorial",
          occurred_at=Timestamp(2024-04-01 03:45:00),
          features=[...]
        )
      ]
    )
  ]
)
```

**Key properties**:
- **One principal** per proto
- **One context** (snapshot time) per proto
- **Multiple actions** (up to max_num_actions_per_ca)
- **Actions sorted by source_type** (alphabetically)

---

## 6. Vocabulary Building

### 6.1 Purpose

**Create vocabulary** of all unique tokens for embedding table initialization

### 6.2 Collection

**From context features**:
```python
tokens_from_contexts = set()
for ca in contextualized_actions:
    for feature in ca.context.features:
        for weighted_token in feature.bag_of_weighted_words.tokens:
            tokens_from_contexts.add(weighted_token.token)
```

**From action features**:
```python
tokens_from_actions = set()
for ca in contextualized_actions:
    for action_source in ca.actions:
        for action in action_source.actions:
            for feature in action.features:
                for weighted_token in feature.bag_of_weighted_words.tokens:
                    tokens_from_actions.add(weighted_token.token)
```

### 6.3 Vocabulary Proto

```python
Vocab(
  action_name_to_vocab={
    "doc_access": Vocabulary(
      embedding_name="action_username",
      tokens=[b"alice", b"bob", b"carol", ...]
    )
  },
  context_vocab=Vocabulary(
    embedding_name="context_username",
    tokens=[b"alice", b"bob", b"carol", b"diana", ...]
  )
)
```

**Note**: Action and context vocabularies can overlap but are separate
- Allows different embeddings for same token in different contexts

### 6.4 Output

**vocab.tfrecord**: TFRecord file containing Vocab proto

**Usage**: Loaded during model training to initialize embedding tables

---

## 7. TFRecord Generation

### 7.1 From ContextualizedActions to tf.SequenceExample

**Conversion function** (in `common/tf_example.py`):
```python
def to_tf_input(ca: ContextualizedActions) -> tf.train.SequenceExample:
    # Convert to TensorFlow's input format
    ...
```

**tf.SequenceExample structure**:
```
context (fixed-size data):
  - principal: string
  - context features: sparse tensors
  
feature_lists (variable-size sequences):
  - action_ids: list of action IDs
  - action_resources: list of resource IDs
  - action_features: list of feature tensors
```

### 7.2 Writing TFRecords

```python
with tf.io.TFRecordWriter(output_file) as writer:
    for ca in contextualized_actions:
        sequence_example = to_tf_input(ca)
        writer.write(sequence_example.SerializeToString())
```

**Output files**:
- `train.tfrecord`: Training data
- `validation.tfrecord`: Validation data
- `vocab.tfrecord`: Vocabulary

---

## 8. Complete Example: End-to-End

### 8.1 Setup

**Directive**:
```textproto
action_sources {
  type: "doc_access"
  history_duration { seconds: 604800 }  # 7 days
  action_deduplication_period { seconds: 3600 }  # 1 hour
  max_values_per_feature: 100
}
context_sources {
  type: "code_review"
  context_lookback { seconds: 604800 }  # 7 days
  peer_feature_configs {
    name: "change_number"
    max_peers: 50
    bipartite_graph {
      traversal_modes: TM_FORWARD
      half_life { seconds: 604800 }
      edge_weighting_method: EWM_DISCOUNTED_LATEST
    }
  }
}
dataset_parameters {
  snapshot_period { seconds: 86400 }  # 1 day
}
```

**Timeline**: Generate data for 2024-04-01 to 2024-04-07 (7 days)

### 8.2 Input Data

**Actions** (doc_access):
```
2024-03-28 10:00 - bob accessed doc_A
2024-03-29 11:00 - alice accessed doc_A
2024-03-30 12:00 - alice accessed doc_A
2024-04-01 14:00 - alice accessed doc_A  ← TO FEATURIZE
2024-04-02 15:00 - bob accessed doc_B    ← TO FEATURIZE
```

**Contexts** (code_review):
```
2024-03-28 - alice authored CL#100, bob reviewed
2024-03-30 - alice authored CL#200, carol reviewed
```

### 8.3 Snapshot Times

```
snapshots = [
  2024-04-01 00:00:00,
  2024-04-02 00:00:00,
  2024-04-03 00:00:00,
  ...
]
```

### 8.4 History Featurizer Output

**For action: alice accessed doc_A at 2024-04-01 14:00**

History window: 2024-03-25 14:00 to 2024-04-01 14:00

Historical accesses:
```
2024-03-28 10:00 - bob (7 days + 4 hours ago, within window)
2024-03-29 11:00 - alice (3 days + 3 hours ago)
2024-03-30 12:00 - alice (2 days + 2 hours ago)
```

After deduplication (1 hour period):
```
bob: 1 access
alice: 2 accesses
```

**Generated features**:
```python
Feature(name="doc_access/prin/t", tokens=[alice, bob])
Feature(name="doc_access/prin/w", tokens=[alice:2.0, bob:1.0])
```

### 8.5 Context Featurizer Output

**For alice at snapshot 2024-04-01 00:00:00**

Lookback window: 2024-03-25 00:00 to 2024-04-01 00:00

Contexts:
```
2024-03-28 - alice → CL#100 → bob (forward traversal)
2024-03-30 - alice → CL#200 → carol (forward traversal)
```

**Bipartite graph** (forward traversal):
```
alice → CL#100 (weight after time decay ≈ 0.5)
alice → CL#200 (weight after time decay ≈ 0.63)
CL#100 → bob
CL#200 → carol
```

**Two-hop paths**:
```
alice → CL#100 → bob (weight ≈ 0.25)
alice → CL#200 → carol (weight ≈ 0.32)
```

**Generated features**:
```python
Feature(name="code_review/change_number_f/t", tokens=[bob, carol])
Feature(name="code_review/change_number_f/w", tokens=[bob:0.25, carol:0.32])
```

### 8.6 Merger Output

**For alice at snapshot 2024-04-01 00:00:00**:

**Context**: From context featurizer (valid_from = 2024-04-01 00:00)

**Actions**: Find actions with occurred_at ≥ 00:00 and matched to this snapshot
- alice accessed doc_A at 14:00 → matches snapshot 00:00 ✓

**Result**:
```python
ContextualizedActions(
  principal="alice",
  context=FeaturizedContext(
    valid_from=Timestamp(2024-04-01 00:00:00),
    features_per_source=[
      ContextSourceFeatures(
        source_type="code_review",
        features=[
          Feature(name="code_review/change_number_f/t", 
                  tokens=[bob, carol]),
          Feature(name="code_review/change_number_f/w",
                  tokens=[bob:0.25, carol:0.32])
        ]
      )
    ]
  ),
  actions=[
    FeaturizedActionsBySource(
      source_type="doc_access",
      actions=[
        FeaturizedAction(
          id=b"action_alice_doc_A_20240401",
          resource_id="doc_A",
          occurred_at=Timestamp(2024-04-01 14:00:00),
          features=[
            Feature(name="doc_access/prin/t", tokens=[alice, bob]),
            Feature(name="doc_access/prin/w", tokens=[alice:2.0, bob:1.0])
          ]
        )
      ]
    )
  ]
)
```

**Interpretation**:
- Alice's context: She collaborates with Bob and Carol (reviewers)
- Alice's action: She accessed doc_A (which was previously accessed by herself and Bob)
- Model will learn: "People with reviewers Bob/Carol accessing docs like doc_A is normal"

---

## 9. Summary

### Three Pipelines

1. **History Featurizer**: 
   - Temporal window-based processing
   - Event timeline with accumulation/expiration
   - Deduplication and top-k selection

2. **Context Featurizer**:
   - Bipartite graph construction
   - Two-hop random walk
   - Time decay and edge weighting

3. **Merger**:
   - Snapshot-based alignment
   - Context-action matching
   - Chunking and sampling

### Key Features Generated

**Action features**:
- `{action_type}/prin/t`: Who accessed historically (tokens)
- `{action_type}/prin/w`: How often they accessed (weights)

**Context features**:
- `{context_type}/{attribute}_f/t`: Forward traversal peers (tokens)
- `{context_type}/{attribute}_f/w`: Forward traversal weights
- `{context_type}/{attribute}_b/t`: Backward traversal peers (tokens)
- `{context_type}/{attribute}_b/w`: Backward traversal weights

### Output

**ContextualizedActions protos** containing:
- Principal identity
- Context features (who are their peers)
- Action features (what was accessed, who accessed it before during that snapshot)
- Ready for model training

### Next Steps

**Document 6** will explain how the model processes these features through neural networks to produce embeddings and scores.

**Ready for Document 6: Model Architecture and Training**
