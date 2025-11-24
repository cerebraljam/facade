# Document 9: Extending Facade - Resource-Centric View

## Introduction

Facade's current design answers: **"Given Principal P's social context, is Action A normal?"**

This document explores your proposed extension: **"Given Resource R, is it normal for Principal P to access it, based on the history of other Principals who accessed R?"**

We'll cover:
1. The conceptual shift from principal-centric to resource-centric
2. Designing a second bipartite graph (Principal ↔ Resource ↔ Principal)
3. Combining both dimensions for enhanced detection
4. Implementation roadmap
5. Evaluation strategies

---

## 1. The Two Dimensions of Anomaly Detection

### 1.1 Current Approach: Principal-Centric

**Question**: "Is this action normal for **this principal**?"

**Context**: Principal's social network (collaborators)

**Example**:
```
Alice accesses document X
Alice's context: Works with Bob, Carol (code review network)
Bob accessed document X → Alice's access seems normal
Score: LOW (0.15)
```

**Strength**: Detects behavior anomalous **for that specific user**

**Limitation**: Misses resource-level anomalies

**Failure case**:
```
Alice accesses TOP_SECRET_DATABASE
Alice's peers (Bob, Carol) also accessed it recently
Score: LOW (0.12) ← MISSED ANOMALY!

Reality: All three are colluding insiders
```

### 1.2 Proposed Extension: Resource-Centric

**Question**: "Is this access normal for **this resource**?"

**Context**: Resource's access history (who typically accesses it)

**Example**:
```
Alice accesses TOP_SECRET_DATABASE
Typical accessors: CEO, CFO, Security Team
Alice: Junior engineer
Score: HIGH (0.89) ← DETECTED!
```

**Strength**: Detects resource-level anomalies independent of peer collusion

**Complementarity**: Catches what principal-centric misses

### 1.3 Combined Approach: Dual-Perspective Scoring

**Score** = f(Principal-Centric Score, Resource-Centric Score)

**Example decision matrix**:

| Principal Score | Resource Score | Combined | Interpretation |
|----------------|----------------|----------|----------------|
| Low (0.1) | Low (0.1) | Low (0.1) | Normal: User's typical behavior, resource's typical accessor |
| Low (0.1) | High (0.9) | High (0.7) | **ALERT**: User's peers access it, but user is atypical for resource |
| High (0.9) | Low (0.1) | High (0.7) | **ALERT**: Unusual for user, but typical accessors do access it |
| High (0.9) | High (0.9) | High (0.95) | **CRITICAL**: Anomalous on both dimensions |

**Benefits**:
- Detects collusion (both scores low, but actually all colluding)
- Detects privilege escalation (resource score high)
- Reduces false positives (if one score high but other very low)

---

## 2. Resource-Centric Bipartite Graph Design

### 2.1 The New Graph Structure

**Current graph** (Principal ↔ Attribute ↔ Principal):
```
Alice ←[code_review]→ Bob
Alice ←[code_review]→ Carol
Bob ←[code_review]→ Diana
```

**New graph** (Principal ↔ Resource ↔ Principal):
```
Alice ←[accessed]→ doc_123 ←[accessed]→ Bob
Alice ←[accessed]→ doc_456 ←[accessed]→ Carol
CEO ←[accessed]→ salary_db ←[accessed]→ CFO
```

**Key difference**: 
- **Attribute graph**: Collaboration relationships
- **Resource graph**: Shared access patterns

### 2.2 Graph Construction

**Input**: Action logs

**Process**:
```python
# For each resource, track who accessed it
resource_to_principals = defaultdict(list)

for action in actions:
    resource = action.resource_id
    principal = action.principal
    timestamp = action.time
    resource_to_principals[resource].append((principal, timestamp))
```

**Example**:
```python
resource_to_principals = {
  'doc_123': [
    ('alice@company.com', datetime(2024, 7, 1)),
    ('bob@company.com', datetime(2024, 7, 2)),
    ('carol@company.com', datetime(2024, 7, 3))
  ],
  'salary_db': [
    ('ceo@company.com', datetime(2024, 6, 15)),
    ('cfo@company.com', datetime(2024, 6, 20))
  ]
}
```

**Create PeerAttributes**:
```python
def create_resource_peer_attributes(principal, actions, resource_to_principals):
    """For a given principal, find who else accessed the same resources."""
    peer_attrs = []
    
    # Get resources this principal accessed
    principal_resources = [a.resource_id for a in actions if a.principal == principal]
    
    for resource in principal_resources:
        # Find other principals who accessed this resource
        other_accessors = resource_to_principals[resource]
        peers = {}
        
        for other_principal, timestamp in other_accessors:
            if other_principal != principal:
                # Weight by recency
                time_diff = (current_time - timestamp).days
                intensity = 1.0 / (1.0 + time_diff)
                peers[other_principal] = peers.get(other_principal, 0) + intensity
        
        if peers:
            peer_attr = PeerAttribute(
                name='resource_access',
                peer_ids=[p.encode('utf-8') for p in peers.keys()],
                intensities=list(peers.values()),
                timestamp=current_time
            )
            peer_attrs.append(peer_attr)
    
    return peer_attrs
```

**Example output** (for Alice):
```python
# Alice accessed doc_123 and doc_456
# doc_123 was also accessed by Bob, Carol
# doc_456 was also accessed by Carol

peer_attributes = [
    PeerAttribute(
        name='resource_access',
        peer_ids=[b'bob@company.com', b'carol@company.com'],
        intensities=[0.9, 0.85]  # Based on recency
    )
]
```

### 2.3 Two-Hop Random Walk on Resource Graph

**Apply existing algorithm** (from Document 4):

**1-hop neighbors** (direct co-accessors):
```
Alice accessed doc_123
Bob also accessed doc_123
Carol also accessed doc_123

Alice's 1-hop: {Bob: 0.9, Carol: 0.85}
```

**2-hop neighbors** (co-accessors of co-accessors):
```
Bob accessed doc_789
Diana also accessed doc_789

Alice → Bob (via doc_123) → Diana (via doc_789)

Alice's 2-hop: {Diana: 0.9 * 0.7 = 0.63}
```

**Result**: Alice's resource-centric network = {Bob, Carol, Diana}

**Interpretation**: "People who access similar resources to Alice"

---

## 3. Context Proto Extension

### 3.1 New Context Source Type

**Add to directive.textproto**:
```textproto
# Existing context source (collaboration)
context_sources {
  source_type: "code_review"
  token_features {
    token_feature_name: "code_review/change_number_f/t"
    weight_feature_name: "code_review/change_number_f/w"
    graph_settings {
      peer_attribute_name: "change_number"
      graph_walk_mode: GWM_FORWARD
      node_distance: 2
      pruning { top_k: 20 }
    }
  }
}

# NEW: Resource-centric context
context_sources {
  source_type: "resource_access"
  token_features {
    token_feature_name: "resource_access/co_accessor_f/t"
    weight_feature_name: "resource_access/co_accessor_f/w"
    graph_settings {
      peer_attribute_name: "resource_access"
      graph_walk_mode: GWM_FORWARD
      node_distance: 2
      pruning { top_k: 20 }
    }
  }
  token_features {
    token_feature_name: "resource_access/co_accessor_b/t"
    weight_feature_name: "resource_access/co_accessor_b/w"
    graph_settings {
      peer_attribute_name: "resource_access"
      graph_walk_mode: GWM_BACKWARD
      node_distance: 2
      pruning { top_k: 20 }
    }
  }
}
```

**What this does**:
- **Forward**: "Who else accessed resources that Alice accessed?"
- **Backward**: "Who accessed resources that Alice's co-accessors accessed?"

### 3.2 Context Generation

**Process**:
```python
def create_resource_context(principal, snapshot_time, actions, resource_to_principals):
    """Create context from resource access patterns."""
    context = Context()
    context.principal = principal
    context.snapshot_time = snapshot_time
    context.source_type = "resource_access"
    
    # Generate peer attributes
    peer_attrs = create_resource_peer_attributes(principal, actions, resource_to_principals)
    context.peer_attributes.extend(peer_attrs)
    
    return context
```

**Example**:
```textproto
principal: "alice@company.com"
snapshot_time { seconds: 1720000000 }
source_type: "resource_access"
peer_attributes {
  name: "resource_access"
  peer_ids: "bob@company.com"
  peer_ids: "carol@company.com"
  intensities: 0.9
  intensities: 0.85
  timestamp { seconds: 1720000000 }
}
```

---

## 4. Dual-Tower Architecture

### 4.1 Extended Context Tower

**Current architecture** (principal-centric only):
```python
context_features = [
  "code_review/change_number_f/t",  # Collaboration network
  "code_review/change_number_b/t"
]
```

**Extended architecture** (dual perspective):
```python
context_features = [
  # Principal-centric (collaboration)
  "code_review/change_number_f/t",
  "code_review/change_number_b/t",
  
  # Resource-centric (co-accessors)
  "resource_access/co_accessor_f/t",
  "resource_access/co_accessor_b/t"
]
```

**Updated config.textproto**:
```textproto
context_architecture {
  segment_feature_names: "code_review/change_number_f/t"
  segment_feature_names: "code_review/change_number_b/t"
  segment_feature_names: "resource_access/co_accessor_f/t"  # NEW
  segment_feature_names: "resource_access/co_accessor_b/t"  # NEW
}

# New segment embedders
segment_reductions {
  token_embedding_name: "context_username"
  token_feature_name: "resource_access/co_accessor_f/t"
  intensity_feature_name: "resource_access/co_accessor_f/w"
  segment_weight_scaling: WS_IDENTITY
  segment_weight_normalization: WN_L2
}
segment_reductions {
  token_embedding_name: "context_username"
  token_feature_name: "resource_access/co_accessor_b/t"
  intensity_feature_name: "resource_access/co_accessor_b/w"
  segment_weight_scaling: WS_IDENTITY
  segment_weight_normalization: WN_L2
}
```

**Result**: Context embedding captures both collaboration AND resource access patterns

### 4.2 Alternative: Separate Towers with Ensemble

**Option 2**: Two independent models, combine scores

**Model 1** (principal-centric):
```python
context_features_1 = ["code_review/change_number_f/t", "code_review/change_number_b/t"]
score_1 = model_1(context_features_1, action_features)
```

**Model 2** (resource-centric):
```python
context_features_2 = ["resource_access/co_accessor_f/t", "resource_access/co_accessor_b/t"]
score_2 = model_2(context_features_2, action_features)
```

**Combined score**:
```python
final_score = max(score_1, score_2)  # Alert if either detects anomaly
# OR
final_score = (score_1 + score_2) / 2  # Average
# OR
final_score = 0.7 * score_1 + 0.3 * score_2  # Weighted
```

**Trade-offs**:

| Approach | Pros | Cons |
|----------|------|------|
| **Single model** (4 context features) | Learns interactions between perspectives | More complex, harder to debug |
| **Ensemble** (2 separate models) | Modular, easier to tune | Misses inter-perspective patterns |

**Recommendation**: Start with single model (simpler), move to ensemble if needed

---

## 5. Resource-Level Features

### 5.1 Resource Metadata Features

**Beyond co-accessors, add resource properties**:

**Example**:
```python
# Classify resources by sensitivity
resource_sensitivity = {
  'doc_123': 'PUBLIC',
  'design_doc_456': 'INTERNAL',
  'salary_db': 'CONFIDENTIAL',
  'exec_emails': 'RESTRICTED'
}

# Add to action features
action.features.add(
  name="doc_access/resource_sensitivity/t",
  bag_of_weighted_words=WeightedTokens(
    tokens=[WeightedToken(token=b"CONFIDENTIAL", weight=1.0)]
  )
)
```

**Configuration**:
```textproto
action_sources {
  source_type: "doc_access"
  token_features {
    token_feature_name: "doc_access/resource_sensitivity/t"
    weight_feature_name: "doc_access/resource_sensitivity/w"
    history_settings {
      count_transform: CT_IDENTITY
    }
  }
}
```

**Effect**: Model learns typical sensitivity levels for each user

### 5.2 Resource Access Frequency

**Feature**: How often is this resource accessed?

```python
# Compute access frequency
resource_access_counts = defaultdict(int)
for action in actions:
    resource_access_counts[action.resource_id] += 1

# Categorize
def access_frequency_bucket(count):
    if count < 10: return "rare"
    elif count < 100: return "moderate"
    else: return "frequent"

# Add to action
action.features.add(
  name="doc_access/resource_frequency/t",
  bag_of_weighted_words=WeightedTokens(
    tokens=[WeightedToken(token=access_frequency_bucket(count), weight=1.0)]
  )
)
```

**Effect**: Rare resources get higher anomaly scores when accessed

### 5.3 Resource Age

**Feature**: How old is this resource?

```python
resource_creation_time = {
  'doc_123': datetime(2024, 1, 1),
  'new_doc_999': datetime(2024, 7, 8)  # Just created
}

age_days = (current_time - resource_creation_time[resource_id]).days
age_bucket = "new" if age_days < 7 else "established"

action.features.add(
  name="doc_access/resource_age/t",
  bag_of_weighted_words=WeightedTokens(
    tokens=[WeightedToken(token=age_bucket, weight=1.0)]
  )
)
```

**Effect**: Accessing brand-new resources might be more suspicious

---

## 6. Implementation Roadmap

### 6.1 Phase 1: Resource Graph Generation (Weeks 1-2)

**Tasks**:
1. Extend action logs with resource co-accessor tracking
2. Implement `create_resource_peer_attributes()` function
3. Generate resource-centric contexts
4. Verify contexts contain expected co-accessors

**Deliverable**: `resource_context.tfrecord`

**Validation**:
```python
# Check that Alice's resource context includes Bob (who accessed same docs)
assert 'bob@company.com' in alice_resource_context.peer_attributes[0].peer_ids
```

### 6.2 Phase 2: Configuration Updates (Week 3)

**Tasks**:
1. Add resource context source to directive.textproto
2. Extend context_architecture in config.textproto
3. Add resource-centric segment embedders
4. Update feature mappings

**Deliverable**: Updated `directive.textproto` and `config.textproto`

**Validation**: Run dataset_maker_main.py, verify 4 context features generated

### 6.3 Phase 3: Model Training (Weeks 4-5)

**Tasks**:
1. Generate training dataset with extended contexts
2. Train model with 4 context features
3. Monitor convergence
4. Evaluate metrics

**Deliverable**: Trained model with dual-perspective embeddings

**Validation**: TPR@1%FPR should be ≥ baseline model

### 6.4 Phase 4: Evaluation (Week 6)

**Tasks**:
1. Create test set with known anomalies
2. Compare baseline vs extended model
3. Analyze failure modes
4. Tune thresholds

**Test cases**:
```python
# Case 1: Collusion attack (should be caught by resource-centric)
alice_accesses_secret_db  # Alice's peers also accessed it recently
baseline_score = 0.15  # LOW (peers accessed it)
extended_score = 0.82  # HIGH (Alice not typical accessor of secret_db)

# Case 2: Normal access (should be low on both)
bob_accesses_project_doc
baseline_score = 0.12  # LOW (peers accessed it)
extended_score = 0.08  # LOW (Bob is typical accessor)
```

**Success criteria**: Extended model detects at least 20% more anomalies

### 6.5 Phase 5: Deployment (Weeks 7-8)

**Tasks**:
1. Deploy to staging environment
2. Process 1 week of live data
3. Analyze alert volume and precision
4. Gradual rollout to production

---

## 7. Advanced: Multi-Dimensional Scoring

### 7.1 Combining Three Perspectives

**Perspective 1**: Principal history
- "Has Alice accessed this type of resource before?"

**Perspective 2**: Principal social network (current)
- "Do Alice's collaborators access this?"

**Perspective 3**: Resource social network (new)
- "Is Alice a typical accessor of this resource?"

**Architecture**:
```python
# Three context towers
context_emb_history = history_tower(history_features)
context_emb_collab = collab_tower(collab_features)
context_emb_resource = resource_tower(resource_features)

# Combine (e.g., concatenate)
combined_context = tf.concat([
  context_emb_history,
  context_emb_collab,
  context_emb_resource
], axis=-1)

# Feed through final SNN
final_context_emb = final_snn(combined_context)

# Score action
score = dot_product(final_context_emb, action_emb)
```

### 7.2 Attention Mechanism

**Idea**: Learn which perspective is most important for each decision

```python
# Attention weights
attention_scores = attention_layer([
  context_emb_history,
  context_emb_collab,
  context_emb_resource
])
# attention_scores = [0.2, 0.3, 0.5]  (resource most important)

# Weighted combination
combined_context = (
  0.2 * context_emb_history +
  0.3 * context_emb_collab +
  0.5 * context_emb_resource
)
```

**Benefits**:
- Model adapts perspective importance per case
- Interpretability: Which perspective drove the alert?

---

## 8. Evaluation Strategies

### 8.1 Synthetic Attack Scenarios

**Create labeled test cases**:

**Scenario 1: Privilege escalation**
```python
# Normal: Alice accesses PUBLIC docs
# Attack: Alice accesses RESTRICTED exec_emails (never accessed before)

actions = [
  create_action("alice", "public_doc_1", normal=True),
  create_action("alice", "public_doc_2", normal=True),
  create_action("alice", "exec_emails", normal=False, label="privilege_escalation")
]
```

**Expected**:
- Baseline score: 0.85 (HIGH - unusual for Alice)
- Extended score: 0.92 (HIGHER - Alice not typical accessor of exec_emails)

**Scenario 2: Collusion attack**
```python
# Alice, Bob, Carol (all engineers) collude to access salary_db
# They coordinate, so all access within short time window

actions = [
  create_action("alice", "salary_db", normal=False, label="collusion"),
  create_action("bob", "salary_db", normal=False, label="collusion"),
  create_action("carol", "salary_db", normal=False, label="collusion")
]
```

**Expected**:
- Baseline score: 0.20 (LOW - peers accessed it, seems normal)
- Extended score: 0.88 (HIGH - none are typical accessors of salary_db)

**Scenario 3: Data exfiltration**
```python
# Alice accesses 50 different docs in 1 hour (unusual volume)

actions = [
  create_action("alice", f"doc_{i}", normal=False, label="exfiltration")
  for i in range(50)
]
```

**Expected**: Both models should detect (high volume)

### 8.2 A/B Testing

**Deployment strategy**:
1. Run both models in parallel (baseline + extended)
2. Route 50% of alerts to each
3. Measure analyst feedback

**Metrics**:
```python
# Baseline model
alerts_baseline = 100
true_positives_baseline = 8
false_positives_baseline = 92
precision_baseline = 8 / 100 = 0.08

# Extended model
alerts_extended = 120
true_positives_extended = 15
false_positives_extended = 105
precision_extended = 15 / 120 = 0.125

# Extended model detects 87.5% more threats (15 vs 8)
# But also 14% more false positives (105 vs 92)
# Trade-off acceptable? Depends on SOC capacity
```

### 8.3 Ablation Studies

**Test each component's contribution**:

**Config 1**: Baseline (history + collaboration)
**Config 2**: + Resource co-accessors
**Config 3**: + Resource metadata (sensitivity, frequency)
**Config 4**: + Resource age

**Results** (example):
```
Config 1 (baseline): TPR@1%FPR = 0.72
Config 2 (+co-accessors): TPR@1%FPR = 0.81  (+12.5%)
Config 3 (+metadata): TPR@1%FPR = 0.84  (+3.7%)
Config 4 (+age): TPR@1%FPR = 0.85  (+1.2%)
```

**Conclusion**: Resource co-accessors provide biggest lift

---

## 9. Real-World Considerations

### 9.1 Resource Graph Scalability

**Challenge**: Millions of resources, graph computation expensive

**Solutions**:

**1. Sample resources**:
```python
# Only compute co-accessors for "interesting" resources
def is_interesting(resource):
    return (
        resource.sensitivity in ['CONFIDENTIAL', 'RESTRICTED'] or
        resource.access_count < 100  # Rare resources
    )
```

**2. Incremental updates**:
```python
# Don't rebuild entire graph daily
# Update only for new resources and recent accessors
def incremental_update(graph, new_actions):
    for action in new_actions:
        graph.add_edge(action.principal, action.resource)
```

**3. Approximate co-accessors**:
```python
# Use MinHash/LSH for fast approximate similarity
from datasketch import MinHash, MinHashLSH

lsh = MinHashLSH(threshold=0.5, num_perm=128)
for principal in principals:
    m = MinHash(num_perm=128)
    for resource in principal_resources[principal]:
        m.update(resource.encode('utf8'))
    lsh.insert(principal, m)

# Find approximate co-accessors
co_accessors = lsh.query(alice_minhash)
```

### 9.2 Cold Start Problem

**Issue**: New resource, no access history

**Example**:
```
new_doc_999 just created
Alice accesses it
No one else has accessed it yet
Resource co-accessor features: EMPTY
```

**Solutions**:

**1. Resource content features**:
```python
# Use document title/tags to find similar resources
new_doc_title = "Q3 Financial Report"
similar_docs = find_similar_by_title(new_doc_title)
# Use typical accessors of similar_docs as proxy
```

**2. Gradual weighting**:
```python
# Weight resource-centric score by resource age
resource_age_days = (now - resource.created_at).days
resource_weight = min(1.0, resource_age_days / 30)  # Full weight after 30 days

final_score = (
  0.5 * principal_centric_score +
  0.5 * resource_weight * resource_centric_score
)
```

### 9.3 Privacy Considerations

**Issue**: Resource graph reveals sensitive information

**Example**:
```
Co-accessor graph shows:
  Alice ↔ salary_db ↔ CEO
  → Reveals Alice has access to salary data (might be confidential)
```

**Mitigations**:

**1. Differential privacy**:
```python
# Add noise to co-accessor intensities
noisy_intensity = true_intensity + np.random.laplace(0, sensitivity / epsilon)
```

**2. Aggregation**:
```python
# Don't expose individual co-accessors, only statistics
resource_features = {
  'typical_role': 'executive',  # Most common role
  'access_diversity': 0.3  # Entropy of accessor roles
}
# Don't reveal specific names
```

---

## 10. Complete Example

### 10.1 Scenario Setup

**Organization**: 10 employees, 3 departments (Eng, HR, Exec)

**Resources**:
- `eng_doc_1`, `eng_doc_2` (accessed by engineers)
- `hr_policy`, `salary_db` (accessed by HR)
- `exec_strategy` (accessed by executives)

**Normal patterns**:
- Alice (Eng) accesses eng_docs
- Bob (HR) accesses hr_policy, salary_db
- CEO accesses exec_strategy

**Attack**:
- Alice (Eng) accesses salary_db (privilege escalation)

### 10.2 Baseline Model Behavior

**Context for Alice**:
```
Collaboration network: Carol (Eng), Diana (Eng)
Carol's accesses: eng_doc_1, eng_doc_2
Diana's accesses: eng_doc_1
```

**Scoring**:
```
Alice accesses salary_db
Context embedding: [engineers who access eng_docs]
Action embedding: [salary_db]
Score: 0.75 (HIGH - Alice doesn't typically access HR docs)
```

**Result**: ALERT (good!)

### 10.3 Extended Model Behavior

**Context for Alice** (extended):
```
Collaboration network: Carol (Eng), Diana (Eng)
Resource co-accessor network: (empty - Alice hasn't accessed HR resources)

Resource co-accessors for salary_db: Bob (HR), Eve (HR)
```

**Scoring**:
```
Alice accesses salary_db
Context embedding: [
  Segment 1: Carol, Diana (collaboration)
  Segment 2: (empty) (resource co-accessors)
]
Action embedding: [salary_db]

Comparison with salary_db's typical accessors: Bob, Eve (HR)
Alice's profile: Engineer
Score: 0.91 (VERY HIGH - Alice is not typical accessor)
```

**Result**: CRITICAL ALERT (even better!)

### 10.4 Collusion Attack (Extended Model Advantage)

**Attack**:
- Alice (Eng), Carol (Eng), Diana (Eng) all access salary_db (colluding)

**Baseline model**:
```
Alice accesses salary_db
Alice's peers (Carol, Diana) also accessed it recently
Score: 0.25 (LOW - seems normal based on peers)
```

**Result**: MISSED ANOMALY

**Extended model**:
```
Alice accesses salary_db
Typical accessors of salary_db: Bob (HR), Eve (HR)
Alice, Carol, Diana are engineers (not typical)
Score: 0.89 (HIGH - resource-level anomaly)
```

**Result**: ALERT (catches collusion!)

---

## 11. Summary

### Key Innovations

**1. Dual perspective**:
- Principal-centric: "Normal for this user?"
- Resource-centric: "Normal for this resource?"

**2. Second bipartite graph**:
- Principal ↔ Resource ↔ Principal
- Captures who typically accesses what

**3. Enhanced detection**:
- Catches collusion attacks
- Detects privilege escalation
- Reduces reliance on collaboration networks alone

### Implementation Checklist

- [ ] Generate resource co-accessor PeerAttributes
- [ ] Extend directive.textproto with resource context source
- [ ] Update config.textproto with resource-centric features
- [ ] Train extended model
- [ ] Evaluate on collusion test cases
- [ ] Deploy and monitor

### Trade-offs

**Pros**:
- ✅ Better detection of resource-level anomalies
- ✅ Robust against collusion
- ✅ Complementary to existing approach

**Cons**:
- ❌ Increased computational cost (second graph)
- ❌ Cold start for new resources
- ❌ Privacy concerns (resource access patterns)

### When to Use

**Use resource-centric view when**:
- High-value resources need protection
- Collusion is a significant threat
- Resources have clear "typical accessor" profiles

**Stick with principal-centric when**:
- Resources are homogeneous
- User behavior is more variable
- Computational resources limited

---

## 12. Future Directions

### 12.1 Multi-Modal Graphs

**Combine multiple graph types**:
- Collaboration graph (who works with whom)
- Resource access graph (who accesses what)
- Communication graph (who emails whom)
- Organizational graph (reporting structure)

**Ensemble all perspectives** for comprehensive anomaly detection

### 12.2 Temporal Dynamics

**Track how resource access patterns evolve**:
```python
# salary_db typical accessors over time
week_1_accessors = {Bob, Eve}  # HR team
week_2_accessors = {Bob, Eve, CFO}  # CFO joins
week_3_accessors = {Bob, Eve, CFO, Alice}  # Alice added (suspicious?)

# Detect sudden changes in typical accessor profile
```

### 12.3 Graph Neural Networks

**Replace random walks with GNNs**:
```python
# Learn embeddings directly from graph structure
from torch_geometric.nn import GCNConv

class ResourceGNN(nn.Module):
    def forward(self, resource_graph):
        # Message passing on Principal ↔ Resource graph
        embeddings = gcn_layer(resource_graph)
        return embeddings
```

**Benefits**: More expressive than random walks

---

## Conclusion

You've now designed a **dual-perspective insider threat detection system** that answers:

1. **"Is this action normal for this principal?"** (existing Facade)
2. **"Is this principal a typical accessor of this resource?"** (your extension)

By combining both perspectives, the system achieves:
- **Robustness** against collusion
- **Generalization** to new principals and resources
- **Interpretability** through multiple lenses

**You're ready to implement this extension and push the boundaries of insider threat detection!**

---

**End of Documentation Series**

You now have complete knowledge to:
✅ Understand Facade's architecture  
✅ Reproduce the system from scratch  
✅ Extend it with novel capabilities  
✅ Deploy it in production  

**Happy building! 🚀**
