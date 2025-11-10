# Document 2: Data Model and Proto Schemas

## Introduction

This document explains Facade's data model using **Protocol Buffers (protos)**. Understanding these data structures is essential for:
- Preparing your own data for Facade
- Understanding how information flows through the pipeline
- Configuring the system correctly

We'll examine each proto with practical examples from the sample data.

---

## 1. The Action Proto: Recording Resource Access Events

### 1.1 Proto Definition

```protobuf
message Action {
  string type = 1;
  bytes id = 2;
  string resource_id = 3;
  string principal = 4;
  Timestamp occurred_at = 5;
  bytes history_key = 6;
}
```

### 1.2 Field Explanations

| Field | Type | Purpose | Example |
|-------|------|---------|---------|
| `type` | string | Category of action, matches directive config | `"doc_access"`, `"db_query"`, `"file_download"` |
| `id` | bytes | Globally unique identifier for this action | `b"action_12345_alice_doc789_20240401"` |
| `resource_id` | string | Identifies what was accessed | `"doc_789"`, `"database_customer_records"` |
| `principal` | string | Who performed the action | `"alice@company.com"`, `"bob"` |
| `occurred_at` | Timestamp | When the action happened (UTC) | `2024-04-01 14:30:00 UTC` |
| `history_key` | bytes | Key for grouping historical actions | Usually same as `resource_id` for simple cases |

### 1.3 Practical Example: Document Access

**Scenario**: Alice accesses a document at 2pm on April 1st, 2024

```python
Action(
  type="doc_access",
  id=b"evt_550e8400_alice_doc_ml_guide",
  resource_id="doc_ml_guide",
  principal="alice@company.com",
  occurred_at=Timestamp(seconds=1711980000),  # 2024-04-01 14:00:00 UTC
  history_key=b"doc_ml_guide"
)
```

**Key insights**:
- `type` must match an entry in `directive.action_sources`
- `id` must be unique across all time (allows deduplication, forensics)
- `resource_id` is what the model scores (is accessing this resource normal?)
- `principal` links to Context data (who is this person? what's their social network?)
- `history_key` groups actions for history featurization (more on this in Document 5)

### 1.4 Why history_key?

Usually `history_key = resource_id`, but they can differ:

**Example: Database table access**
```python
# Two actions on the same table
Action(
  type="db_query",
  id=b"query_1",
  resource_id="customers.orders.table",  # Specific table
  principal="alice",
  history_key=b"customers.orders"  # Database-level grouping
)

Action(
  type="db_query",
  id=b"query_2",
  resource_id="customers.users.table",
  principal="bob",
  history_key=b"customers.users"  # Different database
)
```

**Usage**: History features can be built at database level while scoring at table level.

---

## 2. The Context Proto: Behavioral and Social Information

### 2.1 Proto Definition

```protobuf
message Context {
  string type = 1;
  string principal = 2;
  Timestamp valid_from = 3;
  repeated PeerAttribute peer_attributes = 4;
}
```

### 2.2 Field Explanations

| Field | Type | Purpose | Example |
|-------|------|---------|---------|
| `type` | string | Category of context, matches directive | `"code_review"`, `"calendar"`, `"team_membership"` |
| `principal` | string | Who this context describes | `"alice@company.com"` |
| `valid_from` | Timestamp | When this context became valid | `2024-04-01 00:00:00 UTC` |
| `peer_attributes` | repeated | List of relationships/activities | Code reviews, meetings, etc. |

### 2.3 Understanding PeerAttribute

This is where the **bipartite graph** is encoded:

```protobuf
message PeerAttribute {
  string name = 1;
  bytes value = 2;
  Direction direction = 3;
  float weight = 4;
  Timestamp time = 5;
}
```

| Field | Purpose | Example |
|-------|---------|---------|
| `name` | Type of relationship | `"change_number"`, `"meeting_id"`, `"project_id"` |
| `value` | The "middle node" in bipartite graph | `b"CL_12345"`, `b"meeting_789"` |
| `direction` | Relationship type (D_UNSET, D_FORWARD, D_BACKWARD) | D_FORWARD = author, D_BACKWARD = reviewer |
| `weight` | Importance of this edge (0.0 to inf) | `1.0` (default), `2.5` (more important) |
| `time` | When this relationship occurred | Timestamp of code review |

### 2.4 Practical Example: Code Review Context

**Scenario**: Alice authored CL #12345 on April 1st, Bob reviewed it

**Alice's context**:
```python
Context(
  type="code_review",
  principal="alice",
  valid_from=Timestamp(seconds=1711929600),  # 2024-04-01 00:00:00 UTC
  peer_attributes=[
    PeerAttribute(
      name="change_number",
      value=b"12345",
      direction=PeerAttribute.D_FORWARD,  # Alice is author
      weight=1.0,
      time=Timestamp(seconds=1711980000)  # When review happened
    )
  ]
)
```

**Bob's context**:
```python
Context(
  type="code_review",
  principal="bob",
  valid_from=Timestamp(seconds=1711929600),
  peer_attributes=[
    PeerAttribute(
      name="change_number",
      value=b"12345",
      direction=PeerAttribute.D_BACKWARD,  # Bob is reviewer
      weight=1.0,
      time=Timestamp(seconds=1711980000)
    )
  ]
)
```

**Bipartite graph representation**:
```
Alice (Principal) --[D_FORWARD]--> CL#12345 (Middle) <--[D_BACKWARD]-- Bob (Principal)
```

**Two-hop traversal**:
- Alice → (D_FORWARD) → CL#12345 → (D_BACKWARD) → Bob
- Interpretation: "Bob reviews Alice's CLs" (Alice's peer in the reviewer direction)

- Bob → (D_BACKWARD) → CL#12345 → (D_FORWARD) → Alice
- Interpretation: "Alice is reviewed by Bob" (Bob's peer in the author direction)

### 2.5 Undirected Relationships

For symmetric relationships (meetings, project membership), use `D_UNSET`:

**Meeting example**:
```python
# Alice and Bob both attended meeting_789
# Both have identical PeerAttributes (except principal)

Context(
  type="calendar",
  principal="alice",
  peer_attributes=[
    PeerAttribute(
      name="meeting_id",
      value=b"meeting_789",
      direction=PeerAttribute.D_UNSET,  # Undirected
      weight=1.0
    )
  ]
)
```

**Graph**:
```
Alice --[D_UNSET]-- meeting_789 --[D_UNSET]-- Bob
```

**Two-hop traversal** (no direction distinction):
- Alice → meeting_789 → Bob
- Interpretation: "Bob and Alice meet together"

### 2.6 Critical Insight: You Don't Reconstruct Relationships Upfront!

**IMPORTANT**: You only record individual participation in activities. The bipartite graph algorithm discovers peer relationships automatically.

**What you collect** (easy):
```python
# From meeting attendance logs
alice_context = Context(
  type="calendar",
  principal="alice",
  peer_attributes=[
    PeerAttribute(name="meeting_id", value=b"meeting_789")  # Alice was there
  ]
)

bob_context = Context(
  type="calendar", 
  principal="bob",
  peer_attributes=[
    PeerAttribute(name="meeting_id", value=b"meeting_789")  # Bob was there
  ]
)

carol_context = Context(
  type="calendar",
  principal="carol", 
  peer_attributes=[
    PeerAttribute(name="meeting_id", value=b"meeting_789")  # Carol was there
  ]
)
```

**What you DON'T need to do** (hard and unnecessary):
```python
# ❌ DON'T create explicit pair-wise relationships
alice_context = Context(
  peer_attributes=[
    PeerAttribute(value=b"met_with_bob"),     # ❌ Not needed!
    PeerAttribute(value=b"met_with_carol")    # ❌ Not needed!
  ]
)
```

**How the system works**:

1. **Data collection time**: Just log "alice attended meeting_789", "bob attended meeting_789", etc.

2. **Feature generation time** (during `dataset_maker_main.py`):
   - Bipartite graph algorithm (from `fold.py`) runs
   - Builds graph: alice ↔ meeting_789 ↔ bob ↔ meeting_789 ↔ carol
   - Two-hop walk discovers: alice → meeting_789 → bob (Alice's peer: Bob)
   - Two-hop walk discovers: alice → meeting_789 → carol (Alice's peer: Carol)
   - Generates features with discovered peers automatically

3. **Output features** (automatic):
   ```python
   Feature(
     name="calendar/meeting_id/t",
     bag_of_weighted_words=WeightedTokens(
       tokens=[
         WeightedToken(token=b"bob", weight=1.0),    # Discovered automatically!
         WeightedToken(token=b"carol", weight=1.0)   # Discovered automatically!
       ]
     )
   )
   ```

**Why this design is powerful**:
- ✅ **Scalable**: Meeting with 10 people = 10 simple records, not 45 pair-wise relationships
- ✅ **Simple data collection**: Just log "who did what", not "who interacted with whom"
- ✅ **Automatic discovery**: Algorithm finds all peer connections
- ✅ **Consistent**: Same pattern works for code reviews, projects, emails, etc.

**Real-world example**:
```python
# Meeting with 5 people
# You create: 5 Context protos (one per person)
# System discovers: 10 pair-wise relationships automatically (5 choose 2)

for person in ["alice", "bob", "carol", "diana", "eve"]:
    Context(
        principal=person,
        peer_attributes=[
            PeerAttribute(name="meeting_id", value=b"weekly_standup_2024_04_01")
        ]
    )

# Graph algorithm automatically finds:
# alice-bob, alice-carol, alice-diana, alice-eve,
# bob-carol, bob-diana, bob-eve,
# carol-diana, carol-eve,
# diana-eve
# (10 relationships from 5 records!)
```

---

## 3. Contextualized Actions: Merged Data Structure

### 3.1 Proto Definition

```protobuf
message ContextualizedActions {
  string principal = 3;
  FeaturizedContext context = 1;
  repeated FeaturizedActionsBySource actions = 2;
}
```

This is the **intermediate data structure** created during dataset generation, representing:
- A principal's **context** (social network, peer features) at a snapshot time
- All **actions** that principal performed under that context

### 3.2 FeaturizedContext

```protobuf
message FeaturizedContext {
  repeated ContextSourceFeatures features_per_source = 1;
  Timestamp valid_from = 2;
}

message ContextSourceFeatures {
  string source_type = 1;
  repeated Feature features = 2;
}
```

**Example** (after bipartite graph processing):
```python
FeaturizedContext(
  valid_from=Timestamp(seconds=1711929600),  # 2024-04-01 00:00:00
  features_per_source=[
    ContextSourceFeatures(
      source_type="code_review",
      features=[
        Feature(
          name="code_review/change_number_f/t",  # Forward traversal tokens
          bag_of_weighted_words=WeightedTokens(
            tokens=[
              WeightedToken(token=b"bob", weight=0.6),    # Bob reviews Alice often
              WeightedToken(token=b"carol", weight=0.4)   # Carol also reviews
            ]
          )
        ),
        Feature(
          name="code_review/change_number_f/w",  # Forward traversal weights
          bag_of_weighted_words=WeightedTokens(
            tokens=[
              WeightedToken(token=b"bob", weight=0.8),    # Intensity
              WeightedToken(token=b"carol", weight=0.5)
            ]
          )
        )
      ]
    )
  ]
)
```

**Key insight**: Raw `PeerAttribute` values (CL numbers) have been transformed into:
- **Tokens** (t): Principal identifiers (bob, carol) - WHO are the peers
- **Weights** (w): Collaboration intensity - HOW MUCH they collaborate

### 3.3 FeaturizedAction

```protobuf
message FeaturizedAction {
  bytes id = 1;
  string resource_id = 4;
  Timestamp occurred_at = 2;
  repeated Feature features = 3;
}
```

**Example** (Alice accessing a document):
```python
FeaturizedAction(
  id=b"evt_alice_doc_ml_guide",
  resource_id="doc_ml_guide",
  occurred_at=Timestamp(seconds=1711980000),
  features=[
    Feature(
      name="doc_access/prin/t",  # Principal history tokens
      bag_of_weighted_words=WeightedTokens(
        tokens=[
          WeightedToken(token=b"alice", weight=1.0)  # Alice accessed this
        ]
      )
    ),
    Feature(
      name="doc_access/prin/w",  # Principal history weights
      bag_of_weighted_words=WeightedTokens(
        tokens=[
          WeightedToken(token=b"alice", weight=5.0)  # She accessed it 5 times
        ]
      )
    )
  ]
)
```

**Interpretation**: The `features` field contains **history features**:
- "Who has accessed this resource before?" → Alice (herself)
- "How often?" → 5 times

(More complex examples in Document 5: Feature Engineering)

### 3.4 Complete Example: ContextualizedActions

**Scenario**: Alice's snapshot at 2024-04-01 00:00:00 with two actions

```python
ContextualizedActions(
  principal="alice",
  
  # Context: Who is Alice and her social network
  context=FeaturizedContext(
    valid_from=Timestamp(seconds=1711929600),
    features_per_source=[
      ContextSourceFeatures(
        source_type="code_review",
        features=[
          # Alice's code review peers
          Feature(
            name="code_review/change_number_f/t",
            bag_of_weighted_words=WeightedTokens(
              tokens=[
                WeightedToken(token=b"bob", weight=0.6),
                WeightedToken(token=b"carol", weight=0.4)
              ]
            )
          )
        ]
      )
    ]
  ),
  
  # Actions: What Alice accessed under this context
  actions=[
    FeaturizedActionsBySource(
      source_type="doc_access",
      actions=[
        FeaturizedAction(
          id=b"action_1",
          resource_id="doc_ml_guide",
          occurred_at=Timestamp(seconds=1711980000),
          features=[...]  # History features
        ),
        FeaturizedAction(
          id=b"action_2",
          resource_id="doc_python_tutorial",
          occurred_at=Timestamp(seconds=1711990000),
          features=[...]
        )
      ]
    )
  ]
)
```

**Training interpretation**:
- Model sees: Alice (with peers Bob, Carol) accessed doc_ml_guide
- Model learns: "People with peers like Bob/Carol access ML documents"
- Future inference: If Dave has peers like Bob/Carol, accessing ML documents is probably normal for Dave too

---

## 4. Feature Proto: The Fundamental Unit

### 4.1 Definition

```protobuf
message Feature {
  string name = 1;
  oneof type {
    WeightedTokens bag_of_weighted_words = 2;
  }
}

message WeightedTokens {
  repeated WeightedToken tokens = 1;
}

message WeightedToken {
  bytes token = 1;
  float weight = 2;
}
```

### 4.2 Feature Naming Convention

Features use a structured naming scheme:

```
<source_type>/<attribute_name>_<traversal>/<value_type>

Examples:
- code_review/change_number_f/t   (tokens from forward traversal)
- code_review/change_number_f/w   (weights from forward traversal)
- code_review/change_number_b/t   (tokens from backward traversal)
- code_review/change_number_b/w   (weights from backward traversal)
- doc_access/prin/t               (principal tokens from history)
- doc_access/prin/w               (principal weights from history)
```

**Components**:
- **source_type**: Which context/action source (`code_review`, `doc_access`)
- **attribute_name**: Which peer attribute (`change_number`, `meeting_id`, `prin` for principal)
- **traversal**: Graph direction (`f` = forward, `b` = backward, none for undirected)
- **value_type**: `t` = tokens (WHO), `w` = weights (HOW MUCH)

### 4.3 Why Separate Tokens and Weights?

The model needs both:

**Tokens** (`/t`): Identity of peers
```python
WeightedTokens(
  tokens=[
    WeightedToken(token=b"bob", weight=1.0),
    WeightedToken(token=b"carol", weight=1.0)
  ]
)
```
→ Embedded via lookup table → produces semantic representation of "Bob" and "Carol"

**Weights** (`/w`): Intensity of relationships
```python
WeightedTokens(
  tokens=[
    WeightedToken(token=b"bob", weight=10.5),    # Strong collaboration
    WeightedToken(token=b"carol", weight=2.3)    # Weaker collaboration
  ]
)
```
→ Used to compute weighted average of embeddings

**Model processing**:
1. Look up embeddings: `E_bob`, `E_carol`
2. Compute weighted average: `(10.5 * E_bob + 2.3 * E_carol) / (10.5 + 2.3)`
3. Result: Alice's context embedding reflects that Bob is a closer collaborator than Carol

---

## 5. The Complete Data Flow

### 5.1 From Raw Logs to Training Data

```
Step 1: Collect Raw Logs
├─ Action logs:
│  - "alice accessed doc_ml_guide at 2024-04-01 14:00"
│  - "bob accessed doc_python_tutorial at 2024-04-01 15:00"
│
└─ Context logs (individual participation, NOT relationships):
   - "alice authored CL#12345 at 2024-04-01 10:00"
   - "bob reviewed CL#12345 at 2024-04-01 10:00"
   - "alice attended meeting_789 at 2024-03-31 09:00"
   - "bob attended meeting_789 at 2024-03-31 09:00"
   - "carol attended meeting_789 at 2024-03-31 09:00"
   
   Note: You don't need to explicitly record "alice met with bob"
         The bipartite graph algorithm discovers this automatically!

      ▼

Step 2: Convert to Action/Context Protos
├─ Action protos (TFRecord file)
│
└─ Context protos (TFRecord file)

      ▼

Step 3: Process via Facade Pipeline
├─ History Featurizer: Build action history features
├─ Context Featurizer: Build bipartite graph features
└─ Merger: Combine contexts and actions by snapshot time

      ▼

Step 4: ContextualizedActions Protos
└─ Intermediate representation with features

      ▼

Step 5: Convert to tf.SequenceExample
└─ TensorFlow's input format for training

      ▼

Step 6: Train Model
└─ Learns embeddings via metric learning
```

### 5.2 Example Walkthrough

**Raw logs**:
```
# Context log
2024-04-01 10:00 | alice authored CL#12345, bob reviewed it

# Action log
2024-04-01 14:00 | alice accessed doc_ml_guide
```

**Step 2: Convert to protos**

context.tfrecord:
```python
Context(
  type="code_review",
  principal="alice",
  valid_from=Timestamp(2024-04-01 00:00:00),
  peer_attributes=[
    PeerAttribute(name="change_number", value=b"12345", direction=D_FORWARD)
  ]
)

Context(
  type="code_review",
  principal="bob",
  valid_from=Timestamp(2024-04-01 00:00:00),
  peer_attributes=[
    PeerAttribute(name="change_number", value=b"12345", direction=D_BACKWARD)
  ]
)
```

action.tfrecord:
```python
Action(
  type="doc_access",
  id=b"action_1",
  resource_id="doc_ml_guide",
  principal="alice",
  occurred_at=Timestamp(2024-04-01 14:00:00),
  history_key=b"doc_ml_guide"
)
```

**Step 3: Featurization**

Context featurizer processes Alice's context:
- Finds PeerAttribute with value=b"12345", direction=D_FORWARD
- Performs two-hop walk: alice → CL#12345 → bob
- Generates feature:
  ```python
  Feature(
    name="code_review/change_number_f/t",
    bag_of_weighted_words=WeightedTokens(
      tokens=[WeightedToken(token=b"bob", weight=1.0)]
    )
  )
  ```

History featurizer processes alice's action:
- Looks at all past actions on `doc_ml_guide`
- Finds: alice accessed it before
- Generates feature:
  ```python
  Feature(
    name="doc_access/prin/t",
    bag_of_weighted_words=WeightedTokens(
      tokens=[WeightedToken(token=b"alice", weight=1.0)]
    )
  )
  ```

**Step 4: Merger combines them**

```python
ContextualizedActions(
  principal="alice",
  context=FeaturizedContext(
    valid_from=Timestamp(2024-04-01 00:00:00),
    features_per_source=[
      ContextSourceFeatures(
        source_type="code_review",
        features=[Feature(name="code_review/change_number_f/t", ...)]
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
          features=[Feature(name="doc_access/prin/t", ...)]
        )
      ]
    )
  ]
)
```

**Step 5: Convert to tf.SequenceExample** (TensorFlow format)

**Step 6: Train model**

---

## 6. Understanding Snapshot Times

### 6.1 What Are Snapshots?

**Snapshot times** discretize the temporal dimension:

```
Timeline:
|-------|-------|-------|-------|-------|-------|
12:00   14:00   16:00   18:00   20:00   22:00
  ^       ^       ^       ^       ^       ^
  |       |       |       |       |       |
Snapshot 1      Snapshot 2      Snapshot 3 ...
```

**Configuration** (from directive):
```protobuf
dataset_parameters {
  snapshot_period { seconds: 7200 }  # 2 hours
  snapshot_time_offset { seconds: 0 }  # Start at midnight
}
```

**Generated snapshots** (for 2024-04-01):
- 00:00, 02:00, 04:00, 06:00, ..., 22:00

### 6.2 Why Snapshots?

**Efficiency**: Context features are expensive to compute
- Bipartite graph traversal is costly
- Without snapshots: recompute for every action
- With snapshots: compute once per snapshot, reuse for all actions in that period

**Example**:
- Snapshot time: 2024-04-01 12:00
- Alice's context computed at 12:00 (includes all code reviews up to 12:00)
- Actions between 12:00 and 14:00 use this context:
  - 12:30: Alice accesses doc_A → uses 12:00 context
  - 13:15: Alice accesses doc_B → uses 12:00 context
  - 13:45: Alice accesses doc_C → uses 12:00 context

### 6.3 Context Valid_From

The `valid_from` timestamp indicates when a context becomes active:

```python
FeaturizedContext(
  valid_from=Timestamp(2024-04-01 12:00:00),
  features_per_source=[...]
)
```

**Merger logic**:
- Action occurred_at: 2024-04-01 13:30
- Find latest snapshot ≤ 13:30 → 12:00
- Use context with valid_from = 12:00

**Critical rule**: Context features only use data from BEFORE valid_from
- Prevents data leakage
- Ensures causality (context predates action)

---

## 7. Data Preparation Checklist

### 7.1 For Actions

When preparing Action protos:

✅ **type**: Must match a `directive.action_sources[].type`
✅ **id**: Globally unique, stable identifier
✅ **resource_id**: What you want to score (granularity matters!)
✅ **principal**: Must match principals in Context data
✅ **occurred_at**: Accurate timestamps in UTC
✅ **history_key**: Usually = resource_id (unless grouping needed)

**Common mistakes**:
- ❌ resource_id too granular (e.g., "doc123_version_5_page_3")
  - Better: "doc123" (score at document level)
- ❌ principal format inconsistent (e.g., "alice" vs "alice@company.com")
- ❌ occurred_at in local timezone instead of UTC

### 7.2 For Contexts

When preparing Context protos:

✅ **type**: Must match a `directive.context_sources[].type`
✅ **principal**: Consistent with Action data
✅ **valid_from**: Snapshot time or earlier
✅ **peer_attributes**: Carefully choose:
  - **name**: Type of relationship (code review, meeting, etc.)
  - **value**: The "middle node" (CL number, meeting ID, etc.)
  - **direction**: D_FORWARD/D_BACKWARD if asymmetric, D_UNSET if symmetric
  - **weight**: Default to 1.0, increase for more important relationships
  - **time**: When the relationship occurred (for time decay)

**Common mistakes**:
- ❌ Using principal names as `value` (should be the shared activity/attribute)
  - Wrong: `value=b"bob"` (principal name)
  - Right: `value=b"CL_12345"` (shared code review)
- ❌ Inconsistent direction usage (mixing D_FORWARD and D_UNSET for same attribute name)
- ❌ Forgetting to set `time` (defaults to valid_from, loses temporal granularity)

---

## 8. Summary

### Key Proto Structures

1. **Action**: Records "Principal X accessed Resource Y at time T"
2. **Context**: Records "Principal X has relationship Z at time T"
3. **PeerAttribute**: Encodes relationships as bipartite graph edges
4. **ContextualizedActions**: Combines context + actions with features
5. **Feature**: Bag of weighted tokens (WHO and HOW MUCH)

### Critical Concepts

- **Bipartite graphs** via PeerAttribute (Principal ↔ Activity ↔ Principal)
- **Snapshot times** for efficient context computation
- **Features** separate tokens (identity) from weights (intensity)
- **Featurization** transforms raw protos into ML-ready representations

### Data Flow

```
Raw Logs → Action/Context Protos → Featurization → ContextualizedActions → tf.SequenceExample → Training
```

### Next Steps

**Document 3** will show how to configure:
- `directive.textproto`: Which features to extract, how to process graphs
- `config.textproto`: Model architecture, embeddings, training hyperparameters

---

## Practice Questions

Test your understanding:

1. **What's the difference between `resource_id` and `history_key` in an Action?**
   - resource_id: What is scored (granularity for anomaly detection)
   - history_key: How historical actions are grouped (can be coarser)

2. **How does PeerAttribute encode a bipartite graph?**
   - `value` is the middle node
   - Multiple principals with same `value` are connected via two-hop traversal

3. **Why do we need both D_FORWARD and D_BACKWARD directions?**
   - Captures asymmetric relationships (author vs reviewer, organizer vs attendee)
   - Enables separate feature extraction for different relationship types

4. **What does `valid_from` represent in FeaturizedContext?**
   - The snapshot time when this context became active
   - Features only use data from BEFORE this time

5. **Why separate `/t` (tokens) and `/w` (weights) features?**
   - Tokens: looked up in embedding table (semantics)
   - Weights: used for weighted averaging (importance)

**Ready for Document 3: Configuration Deep Dive**
