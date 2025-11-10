# Document 3: Configuration Deep Dive

## Introduction

Facade's behavior is controlled by two configuration files:
- **`directive.textproto`**: WHAT to featurize (data processing, graph construction)
- **`config.textproto`**: HOW to learn (model architecture, training hyperparameters)

This document provides a comprehensive explanation of every parameter, their effects, and how to tune them for your use case.

---

## Part 1: directive.textproto - Data Processing Configuration

### 1. Overview

The directive controls:
1. **Action sources**: Which action types to process and how to build history features
2. **Context sources**: Which context types to process and how to build bipartite graphs
3. **Dataset parameters**: Snapshot timing and data organization

```protobuf
message Directive {
  int32 model_version = 1;
  repeated ActionSourceConfig action_sources = 2;
  repeated ContextSourceConfig context_sources = 3;
  DatasetParameters dataset_parameters = 4;
}
```

---

### 2. model_version

```textproto
model_version: 1
```

**Purpose**: Version identifier for this directive configuration

**When to change**:
- Increment when making breaking changes to featurization
- Each version requires a separate model (can't mix training data from different versions)

**Example versioning strategy**:
- v1: Initial deployment with code_review context
- v2: Added calendar context source
- v3: Changed history_duration from 30 to 90 days

---

### 3. Action Sources Configuration

```textproto
action_sources {
  type: "doc_access"
  max_history_keys_per_day: -1
  history_duration {
    seconds: 7776000  # 90 days
  }
  action_deduplication_period {
    seconds: 7200  # 2 hours
  }
  max_values_per_feature: 1000
  min_values_per_feature: 1
}
```

### 3.1 type

**Purpose**: Identifies this action type (matches Action.type in data)

**Requirements**:
- Must match `[\w\d-_]+` regex
- Must be unique across all action_sources
- Must match Action.type in your TFRecord data

**Examples**:
```textproto
action_sources { type: "doc_access" }
action_sources { type: "db_query" }
action_sources { type: "file_download" }
```

### 3.2 history_duration

**Purpose**: How far back to look when building history features

**Interpretation**: For an action at time T, use history from [T - history_duration, T)

**Example** (history_duration = 90 days):
```
Action: Alice accesses doc_X at 2024-07-01
History window: 2024-04-01 to 2024-07-01
Question: "Who accessed doc_X in the past 90 days?"
```

**Tuning guidelines**:

| Use Case | Recommended Duration | Rationale |
|----------|---------------------|-----------|
| Rapidly changing teams | 30-60 days | Recent history most relevant |
| Stable organizations | 90-180 days | More data = better statistics |
| Seasonal patterns | 365+ days | Capture yearly cycles |
| Rare events | 180-365 days | Need longer window for statistics |

**Trade-offs**:
- ✅ Longer duration: More historical data, better statistics
- ❌ Longer duration: Slower computation, may include stale data
- ✅ Shorter duration: Fresher data, adapts to role changes
- ❌ Shorter duration: Less data, poor statistics for rare resources

### 3.3 action_deduplication_period

**Purpose**: Ignore repeated actions within this time window

**Why needed**: Action logs often contain duplicates or near-duplicates
- Multiple log entries for same event
- Repeated accesses (e.g., auto-refresh) that aren't meaningful

**Example** (deduplication_period = 2 hours):
```
Raw logs:
  10:00 - Alice accesses doc_X
  10:05 - Alice accesses doc_X  ← IGNORED (within 2 hours)
  10:30 - Alice accesses doc_X  ← IGNORED (within 2 hours)
  12:01 - Alice accesses doc_X  ✓ KEPT (> 2 hours since 10:00)

Result: Only 10:00 and 12:01 actions are counted
```

**Tuning guidelines**:

| Action Type | Recommended Period | Rationale |
|-------------|-------------------|-----------|
| Document views | 1-4 hours | Users may re-open documents frequently |
| Database queries | 5-30 minutes | Queries may be automated/scripted |
| File downloads | 24 hours | Rare to download same file multiple times |
| Login events | 1 hour | Session management may cause duplicates |

**Effect on features**: After deduplication, history features show:
- **Tokens**: WHICH principals accessed (deduplicated)
- **Weights**: HOW MANY times they accessed (after deduplication)

### 3.4 max_history_keys_per_day

**Purpose**: Filter out extremely popular resources

**Formula**: Drop resources with > `max_history_keys_per_day * num_days` accesses

**Why needed**: Super-popular resources (e.g., company homepage, shared dashboards) don't provide useful signal
- Accessed by everyone → doesn't distinguish normal from anomalous
- Computationally expensive to process

**Example** (max_history_keys_per_day = 100, history_duration = 90 days):
```
Resource A: 5,000 accesses in 90 days → 5000 / 90 ≈ 56 per day → ✓ KEPT
Resource B: 15,000 accesses in 90 days → 15000 / 90 ≈ 167 per day → ✗ DROPPED
```

**Special value**: `-1` disables filtering (keep all resources)

**Tuning guidelines**:
- **Large organization**: 50-200 per day
- **Small organization**: Can disable (-1) or use higher threshold
- **Monitor**: Check what % of actions are dropped; adjust if too many/few

### 3.5 max_values_per_feature

**Purpose**: Limit number of unique tokens per feature (top-k selection)

**Why needed**: 
- Memory efficiency (don't store thousands of historical accessors)
- Model efficiency (embeddings for thousands of tokens are expensive)
- Statistical robustness (very rare tokens have poor embeddings)

**Example** (max_values_per_feature = 1000):
```
History of doc_X shows 2,500 unique principals accessed it
→ Keep top 1,000 by access frequency
→ Drop the remaining 1,500 rare accessors
```

**Tuning guidelines**:

| Organization Size | Recommended Value | Rationale |
|-------------------|------------------|-----------|
| Small (< 500 people) | 100-500 | Most resources accessed by small fraction |
| Medium (500-5000) | 500-1000 | Balance coverage vs. efficiency |
| Large (> 5000) | 1000-2000 | Need more capacity for popular resources |

**Trade-offs**:
- ✅ Higher value: More complete history, better coverage
- ❌ Higher value: Slower computation, more memory, noisier features
- ✅ Lower value: Faster, cleaner signal (only frequent accessors)
- ❌ Lower value: May lose important rare collaborators

### 3.6 min_values_per_feature

**Purpose**: Drop actions with too few history values (training/validation only)

**Why needed**: Actions with very sparse history don't provide good training signal
- Not enough data to learn meaningful patterns
- High variance in scores

**Example** (min_values_per_feature = 1):
```
Training mode:
  Action on resource_A (3 historical accessors) → ✓ KEPT (≥ 1)
  Action on resource_B (0 historical accessors) → ✗ DROPPED (< 1)

Inference mode:
  Action on resource_B (0 historical accessors) → ✓ KEPT (min not enforced)
```

**Important**: This filter is **NOT applied during inference**
- Training: Can afford to be selective
- Inference: Must score everything, even zero-history resources

**Tuning guidelines**:
- **Typical value**: 1 (require at least some history)
- **Conservative**: 2-5 (only actions with meaningful history)
- **Aggressive**: 0 (allow even zero-history actions in training)

---

### 4. Context Sources Configuration

```textproto
context_sources {
  type: "code_review"
  context_lookback {
    seconds: 7776000  # 90 days
  }
  peer_feature_configs {
    name: "change_number"
    max_peers: 200
    bipartite_graph {
      traversal_modes: TM_FORWARD
      traversal_modes: TM_BACKWARD
      half_life {
        seconds: 7776000  # 90 days
      }
      edge_weighting_method: EWM_DISCOUNTED_LATEST
    }
    aggregation_method: AGG_ACCUMULATE
  }
}
```

### 4.1 type

**Purpose**: Identifies this context type (matches Context.type in data)

**Requirements**: Same as action type (unique, matches `[\w\d-_]+`, matches data)

**Examples**:
```textproto
context_sources { type: "code_review" }
context_sources { type: "calendar" }
context_sources { type: "team_membership" }
```

### 4.2 context_lookback

**Purpose**: How far back to look when collecting context data

**Interpretation**: At snapshot time T, use contexts from [T - context_lookback, T)

**Example** (context_lookback = 90 days):
```
Snapshot time: 2024-07-01 12:00
Lookback window: 2024-04-01 12:00 to 2024-07-01 12:00
Collect: All Context protos with valid_from in this window
```

**Relationship to aggregation_method**:

**AGG_ACCUMULATE** (keep all):
```
Lookback = 90 days
→ Collect all code reviews from past 90 days
→ Build graph from all of them
```

**AGG_LATEST** (keep most recent):
```
Lookback = 90 days (defines search window)
→ But only use the LATEST value per attribute
→ Useful for "current team" (not historical teams)
```

**Tuning guidelines**:
- Should generally match or exceed `history_duration` from action sources
- Reasoning: Context should cover the same time period as action history
- Exception: AGG_LATEST contexts can use shorter lookback (only need recent data)

### 4.3 Peer Feature Configs

Each `peer_feature_configs` entry defines ONE bipartite graph configuration.

#### 4.3.1 name

**Purpose**: Which PeerAttribute.name to process

**Must match**: A `PeerAttribute.name` value in your Context data

**Example**:
```textproto
# This config processes PeerAttribute entries with name="change_number"
peer_feature_configs {
  name: "change_number"
  ...
}
```

**Multiple configs**: You can have multiple peer_feature_configs for different attributes:
```textproto
context_sources {
  type: "code_review"
  peer_feature_configs { name: "change_number" ... }
}
context_sources {
  type: "calendar"
  peer_feature_configs { name: "meeting_id" ... }
  peer_feature_configs { name: "organizer_id" ... }
}
```

#### 4.3.2 max_peers

**Purpose**: Maximum number of peer principals to include per principal

**Selection**: After graph traversal, if more than max_peers, keep top-k by weight

**Example** (max_peers = 200):
```
Graph traversal finds 500 peers for Alice
Weights: [bob: 10.5, carol: 8.3, ..., zoe: 0.01]
→ Sort by weight, keep top 200
→ Drop bottom 300 (weakest connections)
```

**Tuning guidelines**:

| Organization Type | Recommended Value | Rationale |
|-------------------|------------------|-----------|
| Small teams | 50-100 | Most people have < 100 close collaborators |
| Large org, focused roles | 100-200 | Typical professional network size |
| Large org, cross-functional | 200-500 | May collaborate with many teams |

**Trade-offs**:
- ✅ Higher max_peers: More complete social network
- ❌ Higher max_peers: More noise (weak connections), slower computation
- ✅ Lower max_peers: Focus on strongest relationships
- ❌ Lower max_peers: May miss important but infrequent collaborators

#### 4.3.3 aggregation_method

**Controls**: How to handle multiple Context protos in the lookback window

**Options**:

**AGG_ACCUMULATE** (most common):
```textproto
aggregation_method: AGG_ACCUMULATE
```
**Behavior**: Keep ALL attribute values from all contexts in window

**Example** (code reviews):
```
Context lookback: Past 90 days
Alice authored CLs: #100, #200, #300, #400, #500

Graph edges:
  Alice → CL#100 → reviewers_of_100
  Alice → CL#200 → reviewers_of_200
  Alice → CL#300 → reviewers_of_300
  Alice → CL#400 → reviewers_of_400
  Alice → CL#500 → reviewers_of_500

All 5 CLs contribute to finding Alice's review network
```

**Use cases**: 
- Code reviews (all recent reviews matter)
- Meetings (all recent meetings matter)
- Any relationship where history is relevant

**AGG_LATEST** (for static attributes):
```textproto
aggregation_method: AGG_LATEST
```
**Behavior**: Keep ONLY the most recent attribute value per attribute name

**Example** (team membership):
```
Contexts:
  2024-04-01: Alice in team "engineering"
  2024-05-01: Alice moved to team "research"
  2024-06-01: Alice still in team "research"

Result: Only keep "research" (latest value)
Graph: Alice → team_research → members_of_research
```

**Use cases**:
- Current team membership (don't want historical teams)
- Current manager (previous managers not relevant)
- Current office location

**Critical**: AGG_LATEST raises error if multiple values at same timestamp
- E.g., can't have `valid_from=2024-07-01, value=team_A` AND `valid_from=2024-07-01, value=team_B`

#### 4.3.4 Bipartite Graph Configuration

```textproto
bipartite_graph {
  traversal_modes: TM_FORWARD
  traversal_modes: TM_BACKWARD
  half_life { seconds: 7776000 }
  edge_weighting_method: EWM_DISCOUNTED_LATEST
}
```

##### traversal_modes

**Controls**: Which graph directions to traverse

**Options**:

**TM_UNDIRECTED**: Ignore edge directions
```textproto
traversal_modes: TM_UNDIRECTED
```
**Graph**:
```
Alice ←→ meeting_789 ←→ Bob
(No direction, symmetric relationship)
```
**Result**: Feature name suffix `_u`
- `calendar/meeting_id_u/t` (tokens)
- `calendar/meeting_id_u/w` (weights)

**TM_FORWARD**: Follow D_FORWARD edges
```textproto
traversal_modes: TM_FORWARD
```
**Graph**:
```
Alice --[D_FORWARD]--> CL#123 --[D_FORWARD]--> Carol
                                    (Alice authored, Carol also authored)
```
**Result**: Feature name suffix `_f`
- `code_review/change_number_f/t`
- `code_review/change_number_f/w`

**TM_BACKWARD**: Follow D_BACKWARD edges
```textproto
traversal_modes: TM_BACKWARD
```
**Graph**:
```
Alice --[D_BACKWARD]--> CL#123 --[D_BACKWARD]--> Bob
                                     (Alice reviewed, Bob also reviewed)
```
**Result**: Feature name suffix `_b`
- `code_review/change_number_b/t`
- `code_review/change_number_b/w`

**Multiple modes** (directional relationships):
```textproto
traversal_modes: TM_FORWARD
traversal_modes: TM_BACKWARD
```
**Creates TWO separate features**:
- Forward traversal → "people Alice reviews with"
- Backward traversal → "people who review Alice"

**Use cases**:

| Relationship Type | traversal_modes | Interpretation |
|-------------------|----------------|----------------|
| Meetings | TM_UNDIRECTED | Symmetric: "people I meet with" |
| Code reviews | TM_FORWARD + TM_BACKWARD | Forward: "my reviewers", Backward: "people I review" |
| Email | TM_FORWARD + TM_BACKWARD | Forward: "people I email", Backward: "people who email me" |
| Team membership | TM_UNDIRECTED | "my teammates" |

##### half_life

**Purpose**: Apply exponential time decay to edge weights

**Formula**: `weight *= (0.5) ^ ((current_time - attribute_time) / half_life)`

**Interpretation**: After `half_life` duration, weight is reduced to 50%

**Example** (half_life = 90 days):
```
Current time: 2024-07-01
CL#100 submitted: 2024-07-01 (0 days ago) → weight *= 1.0
CL#200 submitted: 2024-04-01 (90 days ago) → weight *= 0.5
CL#300 submitted: 2024-01-01 (180 days ago) → weight *= 0.25
```

**Tuning guidelines**:

| Scenario | Recommended half_life | Rationale |
|----------|----------------------|-----------|
| Fast-changing relationships | 30-60 days | Recent collaborations most relevant |
| Stable teams | 90-180 days | Relationships persist longer |
| Long-term projects | 180-365 days | Collaborations span months/years |
| No decay desired | 0 or omit | All history equally weighted |

**Special values**:
- `0` or unset: No time decay (all edges equally weighted regardless of age)
- Very large (e.g., 10 years): Effectively no decay for typical lookback windows

**Effect on features**: Older relationships contribute less to the final peer weights

##### edge_weighting_method

**Purpose**: When a principal has MULTIPLE edges to the same attribute value, how to combine them?

**Context**: Your data might have:
```
Alice authored CL#123 on 2024-04-01 (weight=1.0)
Alice updated CL#123 on 2024-04-15 (weight=1.0)
Alice finalized CL#123 on 2024-05-01 (weight=1.0)
```

**Question**: What's the final edge weight for Alice → CL#123?

**Options**:

**EWM_LATEST**: Use only the most recent
```textproto
edge_weighting_method: EWM_LATEST
```
**Result**: weight = 1.0 (from 2024-05-01, ignore earlier entries)

**EWM_DISCOUNTED_LATEST**: Latest weight with time decay
```textproto
edge_weighting_method: EWM_DISCOUNTED_LATEST
```
**Result**: weight = 1.0 * time_decay_factor(2024-05-01 → current_time)

**EWM_SUM_DISCOUNTED**: Sum all weights with time decay
```textproto
edge_weighting_method: EWM_SUM_DISCOUNTED
```
**Result**: weight = time_decay(2024-04-01) * 1.0 + time_decay(2024-04-15) * 1.0 + time_decay(2024-05-01) * 1.0

**Example** (current_time = 2024-07-01, half_life = 90 days):
```
Edge from 2024-04-01 (90 days ago): 1.0 * 0.5 = 0.5
Edge from 2024-04-15 (76 days ago): 1.0 * 0.54 = 0.54
Edge from 2024-05-01 (60 days ago): 1.0 * 0.63 = 0.63

EWM_LATEST: 1.0
EWM_DISCOUNTED_LATEST: 0.63
EWM_SUM_DISCOUNTED: 0.5 + 0.54 + 0.63 = 1.67
```

**EWM_LOG_SUM_DISCOUNTED**: Log of summed weights
```textproto
edge_weighting_method: EWM_LOG_SUM_DISCOUNTED
```
**Result**: weight = log(1 + EWM_SUM_DISCOUNTED)

**Why useful**: Compresses large counts (e.g., 100 vs 1000 repetitions)
```
EWM_SUM_DISCOUNTED: 10 vs 1000 → huge difference
EWM_LOG_SUM_DISCOUNTED: log(11) ≈ 2.4 vs log(1001) ≈ 6.9 → more balanced
```

**Tuning guidelines**:

| Use Case | Recommended Method | Rationale |
|----------|-------------------|-----------|
| One-time events (meetings) | EWM_DISCOUNTED_LATEST | Each meeting counted once |
| Recurring collaboration | EWM_SUM_DISCOUNTED | Frequency matters |
| High-frequency events | EWM_LOG_SUM_DISCOUNTED | Compress outlier counts |
| Recent relationships matter | EWM_DISCOUNTED_LATEST | Focus on current activity |
| Cumulative history matters | EWM_SUM_DISCOUNTED | Total collaboration counts |

---

### 5. Dataset Parameters

```textproto
dataset_parameters {
  snapshot_period { seconds: 7200 }  # 2 hours
  snapshot_time_offset { seconds: 0 } # timezone adjustment. ex.: 32400 for JST
  max_num_actions_per_contextualized_actions: 1000
  max_num_contextualized_actions_per_principal_snapshot: 100
}
```

#### 5.1 snapshot_period

**Purpose**: Time interval between context snapshots

**Example** (snapshot_period = 2 hours):
```
Start: 2024-04-01 00:00
Snapshots: 00:00, 02:00, 04:00, 06:00, ..., 22:00
```

**Tuning guidelines**:

| Action Frequency | Recommended Period | Rationale |
|------------------|-------------------|-----------|
| High (thousands/hour) | 1-2 hours | Frequent snapshots for fresh context |
| Medium (hundreds/hour) | 2-4 hours | Balance freshness vs. computation |
| Low (tens/hour) | 4-8 hours | Fewer snapshots acceptable |
| Very low (daily) | 12-24 hours | Match natural cycles |

**Trade-offs**:
- ✅ Shorter period: Fresher context, better temporal resolution
- ❌ Shorter period: More snapshots = more computation
- ✅ Longer period: Fewer snapshots = faster pipeline
- ❌ Longer period: Stale context (actions use older context)

#### 5.2 snapshot_time_offset

**Purpose**: Offset from midnight UTC for first snapshot

**Example** (offset = 3 hours, period = 2 hours):
```
Snapshots: 03:00, 05:00, 07:00, 09:00, ...
(Instead of 00:00, 02:00, 04:00, ...)
```

**Typical value**: 0 (start at midnight)

**Use cases for non-zero offset**:
- Align with business hours (e.g., 08:00 start for 8am-8pm business)
- Avoid data pipeline conflicts (e.g., other ETL jobs run at midnight)
- Match organizational time zones (though Facade uses UTC internally)

#### 5.3 max_num_actions_per_contextualized_actions

**Purpose**: Prevent proto size explosion

**Why needed**: ContextualizedActions proto contains:
- Context features (shared across actions)
- Multiple actions

**Problem**: A single principal might have thousands of actions in one snapshot period

**Solution**: Split into multiple ContextualizedActions protos
```
Alice has 2,500 actions in snapshot period
max_num_actions_per_contextualized_actions = 1000

Creates 3 protos:
  CA_1: Alice's context + actions 1-1000
  CA_2: Alice's context + actions 1001-2000  (context duplicated)
  CA_3: Alice's context + actions 2001-2500  (context duplicated)
```

**Tuning guidelines**:
- **Typical value**: 1000
- **High action volume**: 500 (more splitting, smaller protos)
- **Low action volume**: 2000-5000 (less splitting overhead)

**Note**: Context features are duplicated in each split proto
- Trade-off: Duplication cost vs. proto size limits

#### 5.4 max_num_contextualized_actions_per_principal_snapshot

**Purpose**: Limit training data per principal per snapshot (training/validation only)

**Why needed**: Some principals are extremely active (e.g., automated accounts)
- Thousands of actions per snapshot
- Would dominate training data
- Reduce diversity

**Behavior**:
```
Training mode:
  Alice has 150 CAs at snapshot T
  max_num_cas = 100
  → Keep random 100, drop 50

Inference mode:
  Always keep ALL (this limit not applied)
```

**Tuning guidelines**:
- **Typical value**: 100
- **Many active users**: 50 (prevent dominance by power users)
- **Few active users**: 200-500 (need more data)
- **Special**: 0 or negative = no limit (keep all)

**Important**: Inference ALWAYS processes everything (no sampling)

---

## Part 2: config.textproto - Model Configuration

### 1. Overview

The config controls:
1. **Embedding dimensions**: Size of vector representations
2. **Token embeddings**: Vocabulary handling
3. **Architecture**: Neural network structure
4. **Training hyperparameters**: Optimization, loss, regularization

```protobuf
message ModelHyperparameters {
  int32 embedding_dims = 1;
  ScoringFunction scoring_function = 2;
  repeated Transformation action_embeddings_transformations = 3;
  repeated Transformation context_embeddings_transformations = 4;
  map<string, TokenEmbeddingConfig> token_embedding_name_to_config = 5;
  ConcatenateThenSNN context_architecture = 6;
  map<string, ConcatenateThenSNN> action_name_to_architecture = 7;
  TrainingHyperparameters training_hyperparameters = 8;
  string principal_feature_name = 9;
}
```

---

### 2. Embedding Dimensions

```textproto
embedding_dims: 32
```

**Purpose**: Dimensionality of final context and action embeddings

**Interpretation**:
- Context tower outputs 32-dimensional vector
- Action tower outputs 32-dimensional vector
- Scoring: dot product of these 32-d vectors

**Tuning guidelines**:

| Data Scale | Recommended Dims | Rationale |
|------------|-----------------|-----------|
| Small (< 1000 principals) | 32-128 | Sufficient capacity |
| Medium (1000-10000) | 128-512 | More complexity |
| Large (> 10000) | 512-1028 | High capacity needed |

**Trade-offs**:
- ✅ Higher dims: More expressive, can capture subtle patterns
- ❌ Higher dims: More parameters, slower training, overfitting risk
- ✅ Lower dims: Faster, more regularization
- ❌ Lower dims: May underfit complex patterns

**Typical value**: 256 (good balance for most applications)

---

### 3. Scoring Function

```textproto
scoring_function: SF_OMDOT
```

**Purpose**: How to compute similarity between context and action embeddings

**Options**:

**SF_OMDOT** (One Minus Dot product):
```
score = 1 - dot_product(normalize(context_emb), normalize(action_emb))
```
**Range**: [0, 2]
- 0 = identical (perfect match)
- 1 = orthogonal
- 2 = opposite

**Use case**: When you want LOWER scores for anomalies (distance-like)

**SF_DOT** (standard dot product):
```
score = dot_product(normalize(context_emb), normalize(action_emb))
```
**Range**: [-1, 1]
- 1 = identical
- 0 = orthogonal
- -1 = opposite

**Use case**: When you want HIGHER scores for normal (similarity-like)

**Typical choice**: SF_OMDOT (aligns with "anomaly score" interpretation)

---

### 4. Embedding Transformations

```textproto
action_embeddings_transformations: TR_SOFTPLUS
action_embeddings_transformations: TR_L2_NORMALIZED
context_embeddings_transformations: TR_SOFTPLUS
context_embeddings_transformations: TR_L2_NORMALIZED
```

**Purpose**: Apply transformations to embeddings before scoring

**Applied in order**: First TR_SOFTPLUS, then TR_L2_NORMALIZED

**TR_SOFTPLUS**:
```
softplus(x) = log(1 + exp(x))
```
**Effect**: Ensures all values are positive (similar to ReLU but smooth)

**TR_L2_NORMALIZED**:
```
normalized(v) = v / ||v||_2
```
**Effect**: Unit length vectors (length = 1)

**Why this combination?**:
1. TR_SOFTPLUS: Makes embeddings positive
2. TR_L2_NORMALIZED: Makes embeddings unit-length
3. Result: All embeddings lie on positive unit hypersphere
4. Dot product = cosine similarity (measures angle, not magnitude)

**Alternative**: None (no transformations)
- Allows embeddings to have arbitrary magnitude
- Less geometrically constrained

**Typical choice**: TR_SOFTPLUS + TR_L2_NORMALIZED (standard practice)

---

### 5. Token Embedding Configurations

```textproto
token_embedding_name_to_config {
  key: "action_username"
  value {
    dimensions: 16
    num_oov_indices: 1
  }
}
token_embedding_name_to_config {
  key: "context_username"
  value {
    dimensions: 16
    num_oov_indices: 1
  }
}
```

**Purpose**: Configure embedding tables for token lookup

**Key concepts**:

#### 5.1 Token Embedding Names

**action_username** and **context_username** are identifiers for embedding tables:
- Separate tables for action vs. context features
- Allows different embeddings for same token in different contexts
- Example: "alice" as action accessor vs. "alice" as context peer

**Why separate?**:
```
Action: "alice accessed doc_X"
  → Represents Alice as an ACCESSOR
Context: "bob's reviewers include alice"
  → Represents Alice as a REVIEWER

Different roles → different embedding tables
```

#### 5.2 dimensions

```textproto
dimensions: 16
```

**Purpose**: Size of token embedding vectors (before being processed by towers)

**Not the same as** `embedding_dims` (final output dimensions)

**Flow**:
```
Token "alice" 
  → Look up in embedding table (16 dims)
  → Process through neural network layers
  → Output final embedding (32 dims)
```

**Tuning guidelines**:
- **Small vocabulary (< 1000 tokens)**: 8-16 dims
- **Medium vocabulary (1000-10000)**: 16-32 dims
- **Large vocabulary (> 10000)**: 32-64 dims

**Typical value**: 16 (half of final embedding_dims)

#### 5.3 num_oov_indices

```textproto
num_oov_indices: 1
```

**Purpose**: Number of special tokens for out-of-vocabulary (OOV) items

**Why needed**: At inference, may encounter new tokens not in training vocabulary
- New employee joined
- New resource created
- Typo or data quality issue

**How it works**:
```
Vocabulary built from training data:
  [alice, bob, carol, ..., zoe]
  + 1 OOV token: [OOV]

Inference encounters "xavier" (not in vocab):
  → Map to [OOV] token
  → Use OOV embedding
```

**Typical value**: 1 (single OOV token for all unknowns)

**Alternative values**:
- `0`: Error on unknown tokens (strict mode)
- `> 1`: Multiple OOV tokens (hash unknown tokens to distribute across OOVs)

---

### 6. Architecture Configuration

#### 6.1 Context Architecture

```textproto
context_architecture {
  concatenate_then_snn {
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
    snn {
      layer_sizes: 24
    }
  }
}
```

**Purpose**: Defines how to process context features into final embedding

**Architecture**: ConcatenateThenSNN (Concatenate then Siamese Neural Network)

**Flow**:
```
1. Process each segment_reduction separately
2. Concatenate results
3. Pass through SNN (feed-forward network)
4. Output final context embedding
```

##### segment_reductions

**Purpose**: Process one feature (bag of weighted tokens) into a vector

**Example**:
```
Feature: code_review/change_number_f/t
Tokens: [bob, carol, diana]
Weights (from /w feature): [0.6, 0.3, 0.1]

Steps:
1. Look up embeddings: E_bob, E_carol, E_diana (each 16-dim)
2. Apply weight scaling/normalization
3. Compute weighted average: 0.6*E_bob + 0.3*E_carol + 0.1*E_diana
4. Output: segment vector (16-dim)
```

**token_embedding_name**: Which embedding table to use (`"context_username"`)

**token_feature_name**: Which feature contains the tokens (`"code_review/change_number_f/t"`)

**intensity_feature_name**: Which feature contains the weights (`"code_review/change_number_f/w"`)

##### segment_weight_scaling

**Options**:

**WS_IDENTITY**: No scaling (use weights as-is)
```textproto
segment_weight_scaling: WS_IDENTITY
```
**Effect**: weight = weight (no change)

**WS_SQRT**: Square root scaling
```textproto
segment_weight_scaling: WS_SQRT
```
**Effect**: weight = sqrt(weight)
**Use case**: Compress large weight differences

**WS_LOG**: Logarithmic scaling
```textproto
segment_weight_scaling: WS_LOG
```
**Effect**: weight = log(1 + weight)
**Use case**: Strongly compress outlier weights

**Typical choice**: WS_IDENTITY (no scaling)

##### segment_weight_normalization

**Options**:

**WN_L2**: L2 normalization (make weights sum to ||w||_2 = 1)
```textproto
segment_weight_normalization: WN_L2
```
**Effect**: Normalized weights used for averaging
```
weights = [0.6, 0.3, 0.1]
||w||_2 = sqrt(0.6^2 + 0.3^2 + 0.1^2) ≈ 0.67
normalized = [0.6/0.67, 0.3/0.67, 0.1/0.67] ≈ [0.89, 0.45, 0.15]
```

**WN_SUM**: Sum normalization (make weights sum to 1)
```textproto
segment_weight_normalization: WN_SUM
```
**Effect**: Weights become probabilities
```
weights = [0.6, 0.3, 0.1]
sum = 1.0
normalized = [0.6/1.0, 0.3/1.0, 0.1/1.0] = [0.6, 0.3, 0.1]
```

**WN_NONE**: No normalization
```textproto
segment_weight_normalization: WN_NONE
```

**Typical choice**: WN_L2 (standard practice)

##### snn (Siamese Neural Network)

```textproto
snn {
  layer_sizes: 24
}
```

**Purpose**: Feed-forward network to process concatenated segment vectors

**layer_sizes**: Hidden layer dimensions

**Example** (layer_sizes: 24):
```
Input: Concatenated segments (2 segments * 16 dims = 32 dims)
  ↓
Hidden layer 1: 32 → 24 (fully connected + ReLU)
  ↓
Output layer: 24 → 32 (final embedding_dims)
```

**Multiple layers**:
```textproto
snn {
  layer_sizes: 48
  layer_sizes: 24
}
```
**Flow**:
```
Input: 32 dims
  ↓
Hidden 1: 32 → 48
  ↓
Hidden 2: 48 → 24
  ↓
Output: 24 → 32
```

**Tuning guidelines**:
- **Simple data**: No hidden layers (direct linear projection)
- **Medium complexity**: 1 layer, size ≈ embedding_dims
- **Complex patterns**: 2 layers, increasing then decreasing

**Typical value**: Single layer, size close to embedding_dims

#### 6.2 Action Architecture

```textproto
action_name_to_architecture {
  key: "doc_access"
  value {
    concatenate_then_snn {
      segment_reductions {
        segment_weight_scaling: WS_IDENTITY
        segment_weight_normalization: WN_L2
        token_embedding_name: "action_username"
        token_feature_name: "doc_access/prin/t"
        intensity_feature_name: "doc_access/prin/w"
      }
      snn {
        layer_sizes: 24
      }
    }
  }
}
```

**Key**: Must match action source type from directive

**Structure**: Same as context_architecture (segment_reductions + snn)

**Difference**: Processes action features instead of context features
- token_feature_name: `"doc_access/prin/t"` (action history)
- token_embedding_name: `"action_username"` (separate embedding table)

---

### 7. Training Hyperparameters

```textproto
training_hyperparameters {
  batch_size: 100
  training_examples: 80000
  dropout_tokens: 0.05
  dropout_neurons: 0.05
  optimizer { ... }
  learning_rate_schedule { ... }
  loss_function { ... }
  synthetic_positives_strategy { ... }
  action_name_to_loss_weight { ... }
  evaluation { ... }
}
```

#### 7.1 batch_size

```textproto
batch_size: 100
```

**Purpose**: Number of examples per training batch

**Tuning guidelines**:
- **Small dataset (< 10k examples)**: 32-64
- **Medium dataset (10k-1M)**: 64-128
- **Large dataset (> 1M)**: 128-512

**Trade-offs**:
- ✅ Larger batch: More stable gradients, faster training (GPU utilization)
- ❌ Larger batch: More memory, may hurt generalization
- ✅ Smaller batch: Less memory, more regularization effect
- ❌ Smaller batch: Noisier gradients, slower convergence

**Typical value**: 1000

#### 7.2 training_examples

```textproto
training_examples: 80000
```

**Purpose**: Total number of training examples to process before stopping

**Not epochs**: This is total examples, not passes through dataset

**Example**:
```
Dataset size: 10,000 examples
training_examples: 80,000
→ 8 epochs (80,000 / 10,000)

Dataset size: 100,000 examples
training_examples: 80,000
→ 0.8 epochs (80,000 / 100,000, don't even see full dataset once)
```

**Tuning guidelines**:
- Start with 2-5x dataset size
- Monitor validation metrics, increase if still improving
- Typical range: 50,000 - 500,000

#### 7.3 dropout_tokens

```textproto
dropout_tokens: 0.05
```

**Purpose**: Randomly drop tokens from features during training

**Effect**: 5% of tokens in each feature are randomly zeroed out

**Why useful**: 
- Regularization (prevents overfitting to specific tokens)
- Forces model to learn robust patterns
- Simulates missing data

**Typical range**: 0.0 - 0.2 (0.05 is conservative)

#### 7.4 dropout_neurons

```textproto
dropout_neurons: 0.05
```

**Purpose**: Standard neural network dropout in SNN layers

**Effect**: 5% of neurons randomly deactivated during training

**Typical range**: 0.0 - 0.5 (0.05 is conservative)

#### 7.5 Optimizer

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

**AdamW**: Adam with decoupled weight decay (modern best practice)

**Parameters**:
- **weight_decay**: L2 regularization strength (0.0001 - 0.001 typical)
- **beta_1**: Momentum for gradient (0.9 standard)
- **beta_2**: Momentum for squared gradient (0.999 standard)
- **epsilon**: Numerical stability (1e-7 standard)
- **global_clipnorm**: Gradient clipping (prevents explosions)

**Typical values**: Use defaults above unless you have specific needs

#### 7.6 Learning Rate Schedule

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

**OneCycle**: Learning rate starts low, increases to peak, then decreases

**Parameters**:
- **peak_learning_rate**: Maximum LR (0.0001 - 0.01 typical)
- **rampup**: Fraction of training spent ramping up (0.1 - 0.3 typical)
- **rampup_factor**: How much lower than peak to start
- **rampdown_factor**: How much lower than peak to end

**Typical tuning**: Mainly adjust `peak_learning_rate`
- Too high: Training unstable
- Too low: Training slow, may underfit

#### 7.7 Loss Function

```textproto
loss_function {
  pairwise_huber {
    soft_margin: 0.05
    hard_margin: 0.02
    norm_push: 1.0
  }
}
```

**Pairwise Huber**: Contrastive loss with Huber-like margins

**Parameters**:
- **soft_margin**: Begin penalizing when violation exceeds this
- **hard_margin**: Switch from quadratic to linear penalty
- **norm_push**: Strength of penalty

**Interpretation**: 
- Positive pairs should have score < soft_margin
- Negative pairs should have score > soft_margin
- Violations are penalized

**Typical values**: Use defaults unless experimenting

---

## 8. Configuration Strategy

### 8.1 Starting Point

**For new deployments, start with the sample config and adjust**:

1. **directive.textproto**:
   - Change `type` fields to match your data
   - Set `history_duration` and `context_lookback` to 60-90 days
   - Set `snapshot_period` to 2-4 hours
   - Keep other parameters at sample values

2. **config.textproto**:
   - Keep `embedding_dims: 32`, increase for significant dataset
   - Update `token_feature_name` and `intensity_feature_name` to match your directive
   - Keep architecture and training parameters at sample values

### 8.2 Tuning Priority

**High-impact parameters** (tune these first):
1. `history_duration` and `context_lookback` (directive)
2. `snapshot_period` (directive)
3. `max_peers` (directive)
4. `peak_learning_rate` (config)
5. `training_examples` (config)

**Medium-impact** (tune if needed):
1. `max_history_keys_per_day` (directive)
2. `edge_weighting_method` (directive)
3. `batch_size` (config)
4. `embedding_dims` (config)

**Low-impact** (rarely change):
1. Most other directive parameters
2. Optimizer details (config)
3. Dropout rates (config)

---

## Summary

### Key Configuration Files

**directive.textproto**:
- Action sources: History windows, deduplication, filtering
- Context sources: Bipartite graphs, traversal modes, time decay
- Dataset parameters: Snapshots, chunking

**config.textproto**:
- Embeddings: Dimensions, transformations
- Architecture: Segment reductions, SNN layers
- Training: Batch size, learning rate, loss function

### Most Important Parameters

1. **history_duration**: How far back to look (both actions and contexts)
2. **max_peers**: Social network size
3. **traversal_modes**: Graph direction handling
4. **edge_weighting_method**: How to aggregate repeated edges
5. **embedding_dims**: Model capacity

### Next Steps

**Document 4** will dive deep into the bipartite graph algorithm - the core innovation that makes these configurations work.

---

**Ready for Document 4: The Bipartite Graph Pipeline**
