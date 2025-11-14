# Document 4: The Bipartite Graph Pipeline

## Introduction

This document explains **THE CORE INNOVATION** of Facade: the bipartite graph algorithm that transforms social relationships into features.

**Why this matters**: This is what enables Facade to generalize. Without it, Facade would just be "has this person accessed this before?" With it, Facade becomes "do people like this person access things like this?"

---

## 1. What is a Bipartite Graph?

### 1.1 Definition

A **bipartite graph** has two types of nodes:
- **Principal nodes**: People (Alice, Bob, Carol, ...)
- **Middle nodes**: Shared activities/attributes (code reviews, meetings, projects, ...)

**Key property**: Edges only connect principals to middle nodes, never principal-to-principal or middle-to-middle.

### 1.2 Example: Code Review Bipartite Graph

```
Principals              Middle Nodes           Principals
(Authors)               (Code Reviews)         (Reviewers)

  Alice  ───────────→  CL #12345  ───────────→  Bob
                                                  |
  Alice  ───────────→  CL #67890  ───────────────┘
                              |
  Carol  ────────────────────┘
```

**Interpretation**:
- Alice authored CL #12345, Bob reviewed it
- Alice authored CL #67890, Bob reviewed it
- Carol authored CL #67890, Bob reviewed it

### 1.3 Why "Middle Nodes"?

The "middle" is between two principals in a 2-hop path:

```
Alice → CL #12345 → Bob
  └── hop 1 ──┘ └── hop 2 ──┘
  
  Principal → Middle → Principal
```

**This 2-hop traversal is the key to finding social connections.**

---

## 2. From Raw Context Data to Bipartite Graph

### 2.1 Input: Context Protos

```python
Context(
  type="code_review",
  principal="alice",
  peer_attributes=[
    PeerAttribute(
      name="change_number",
      value=b"12345",
      direction=D_FORWARD,  # Alice is author
      weight=1.0
    )
  ]
)

Context(
  type="code_review",
  principal="bob",
  peer_attributes=[
    PeerAttribute(
      name="change_number",
      value=b"12345",
      direction=D_BACKWARD,  # Bob is reviewer
      weight=1.0
    )
  ]
)
```

### 2.2 Building the Graph

**Step 1**: Create edges from PeerAttributes

```python
# Alice's edge
Edge(
  i="alice",          # Principal node
  j="CL#12345",       # Middle node
  is_middle=False,    # i (alice) is NOT middle
  weight=1.0
)

# Bob's edge  
Edge(
  i="CL#12345",       # Middle node
  j="bob",            # Principal node
  is_middle=True,     # i (CL#12345) IS middle
  weight=1.0
)
```

**Key insight**: Direction is encoded via `is_middle`:
- D_FORWARD: principal → middle (is_middle=False)
- D_BACKWARD: middle → principal (is_middle=True)

**Step 2**: Collect all edges

```
Full graph from multiple contexts:

alice → CL#12345 (is_middle=False, weight=1.0)
CL#12345 → bob (is_middle=True, weight=1.0)
alice → CL#67890 (is_middle=False, weight=1.0)
CL#67890 → bob (is_middle=True, weight=1.0)
carol → CL#67890 (is_middle=False, weight=1.0)
```

---

## 3. The Two-Hop Random Walk Algorithm

### 3.1 Algorithm Overview

**Goal**: Transform bipartite graph (Principal ↔ Middle ↔ Principal) into a principal-only graph (Principal ↔ Principal)

**Method**: For each principal, find other principals reachable via 2-hop paths

**Key function**: `two_hops_random_walk_neighbors()` in `fold.py`

### 3.2 Step-by-Step Walkthrough

Let's trace the algorithm with our example:

```
Input edges:
1. alice → CL#12345 (is_middle=False, weight=1.0)
2. CL#12345 → bob (is_middle=True, weight=1.0)
3. alice → CL#67890 (is_middle=False, weight=1.0)
4. CL#67890 → bob (is_middle=True, weight=1.0)
5. carol → CL#67890 (is_middle=False, weight=1.0)
```

#### Step 1: Filter Near-Zero Edges

```python
filtered_edges = [e for e in edges if e.weight > 1e-9]
```

**Result**: All 5 edges kept (all weights = 1.0)

#### Step 2: Normalize Outgoing Weights

**For each origin node, normalize edges to sum to 1.0** (creates probability distribution)

**Origin: alice**
```
Edges: alice → CL#12345 (1.0), alice → CL#67890 (1.0)
Total: 2.0
Normalized: alice → CL#12345 (0.5), alice → CL#67890 (0.5)
```

**Origin: CL#12345**
```
Edges: CL#12345 → bob (1.0)
Total: 1.0
Normalized: CL#12345 → bob (1.0)
```

**Origin: CL#67890**
```
Edges: CL#67890 → bob (1.0)
Total: 1.0
Normalized: CL#67890 → bob (1.0)
```

**Origin: carol**
```
Edges: carol → CL#67890 (1.0)
Total: 1.0
Normalized: carol → CL#67890 (1.0)
```

**Why normalize?** Ensures fair comparison across nodes with different numbers of edges. Represents "random walk transition probability."

#### Step 3: Group by Middle Node

**Reorganize edges by middle node, separating incoming vs outgoing:**

**Middle node: CL#12345**
```
Incoming (from principals): 
  - alice with weight 0.5 (normalized from Step 2)

Outgoing (to principals):
  - bob with weight 1.0
```

**Middle node: CL#67890**
```
Incoming (from principals):
  - alice with weight 0.5
  - carol with weight 1.0

Outgoing (to principals):
  - bob with weight 1.0
```

#### Step 4: Compute 2-Hop Path Weights (Cross-Product)

**For each middle node, multiply incoming × outgoing weights:**

**Via CL#12345**:
```
Path: alice → CL#12345 → bob
Weight: 0.5 × 1.0 = 0.5
```

**Via CL#67890**:
```
Path: alice → CL#67890 → bob
Weight: 0.5 × 1.0 = 0.5

Path: carol → CL#67890 → bob
Weight: 1.0 × 1.0 = 1.0
```

**Accumulate paths with same (start, end):**
```
path_weights = {
  (alice, bob): 0.5 + 0.5 = 1.0,   # Two paths from alice to bob
  (carol, bob): 1.0                # One path from carol to bob
}
```

**Interpretation**:
- Alice reaches Bob via 2 different code reviews (CL#12345 and CL#67890)
- Carol reaches Bob via 1 code review (CL#67890)
- Combined weight higher for Alice→Bob because of multiple connections

#### Step 5: Group by Destination

**Reorganize by which principal is being reached:**

```
Neighbors of bob:
  - alice with weight 1.0
  - carol with weight 1.0
```

#### Step 6: Select Top-K Neighbors

```python
max_neighbors = 200
```

**For bob** (only 2 neighbors, both kept):
```
final_neighbors["bob"] = [
  (1.0, "alice"),
  (1.0, "carol")
]
```

**Result**: Bob's peer features include Alice and Carol (people who he reviews code for)

---

## 4. Directed vs Undirected Traversals

### 4.1 Traversal Modes

Recall from Document 3:
- `TM_FORWARD`: Follow D_FORWARD edges
- `TM_BACKWARD`: Follow D_BACKWARD edges  
- `TM_UNDIRECTED`: Ignore directions

### 4.2 Forward Traversal Example

**Configuration**:
```textproto
traversal_modes: TM_FORWARD
```

**Graph construction**:
```
Only edges where direction = D_FORWARD:
  alice → CL#12345 (author)
  alice → CL#67890 (author)
  carol → CL#67890 (author)
```

**Wait, where are the reviewers?**

In **forward traversal**, we're finding "people who have the same reviewers as me":
- Alice authored CLs reviewed by Bob
- Carol authored CLs reviewed by Bob
- Therefore: Alice and Carol are peers (both have Bob as reviewer)

**Full graph for forward traversal**:
```
alice → CL#12345 → [who else authored CLs reviewed by bob?] → carol
```

**Resulting feature for Alice**:
```
code_review/change_number_f/t: [carol]
code_review/change_number_f/w: [weight]
```

**Interpretation**: "Carol is Alice's peer in the forward direction" = "Carol is also reviewed by people who review Alice"

### 4.3 Backward Traversal Example

**Configuration**:
```textproto
traversal_modes: TM_BACKWARD
```

**Graph construction**:
```
Only edges where direction = D_BACKWARD:
  CL#12345 → bob (reviewer)
  CL#67890 → bob (reviewer)
```

**Backward traversal finds**: "People who review the same CLs as me"

**For Bob**:
```
bob reviewed CL#12345, CL#67890
Who else reviewed those CLs? (In this example, only Bob)
```

**If we had**:
```
CL#12345 → bob
CL#12345 → diana
```

**Then**: Diana is Bob's peer in backward direction = "Diana reviews the same CLs as Bob"

### 4.4 Asymmetric Relationships

**Key insight**: Forward and backward traversals yield DIFFERENT peer sets!

**Example**:
```
Alice authors CLs reviewed by Bob and Carol
Bob reviews CLs authored by Alice and Eve

Forward traversal for Alice:
  → Finds Eve (both Alice and Eve are reviewed by Bob)
  
Backward traversal for Alice:
  → Finds people who author CLs that Alice's reviewers also review
  → In this case: finds herself and Eve
```

**Why useful**: Captures different aspects of relationships:
- **Forward (author)**: "People with similar reviewers" = doing similar work
- **Backward (reviewer)**: "People who review similar things" = expertise overlap

### 4.5 Undirected Traversal

**Configuration**:
```textproto
traversal_modes: TM_UNDIRECTED
```

**Effect**: Treats all edges as bidirectional (ignores D_FORWARD/D_BACKWARD)

**Use case**: Symmetric relationships (meetings, project membership)

**Example** (meetings):
```
alice attended meeting_789
bob attended meeting_789
carol attended meeting_789

Undirected traversal:
  alice → meeting_789 → bob
  alice → meeting_789 → carol
  
Result: Bob and Carol are Alice's peers (meeting attendees)
```

---

## 5. Time Decay and Edge Weighting

### 5.1 Initial Edge Weights

Edges start with weights from PeerAttribute:

```python
PeerAttribute(
  name="change_number",
  value=b"12345",
  weight=2.5,  # Can be > 1.0
  time=Timestamp(...)
)
```

**Creates edge with initial weight = 2.5**

### 5.2 Time Decay (Half-Life)

**Configuration**:
```textproto
half_life { seconds: 7776000 }  # 90 days
```

**Formula**:
```python
time_decay = 0.5 ^ ((current_time - attribute_time) / half_life)
final_weight = initial_weight * time_decay
```

**Example**:
```
Current time: 2024-07-01
Attribute time: 2024-04-01 (90 days ago)
Half-life: 90 days

time_decay = 0.5 ^ (90 / 90) = 0.5
final_weight = 1.0 * 0.5 = 0.5
```

**Effect**: Recent collaborations weighted more heavily than old ones

### 5.3 Edge Weighting Methods

Recall from Document 3, when a principal has multiple edges to the same middle node:

**EWM_LATEST**: Use most recent edge only
```
alice → CL#123 at 2024-04-01 (weight=1.0)
alice → CL#123 at 2024-05-01 (weight=1.0)  ← Use this one
Final: weight=1.0
```

**EWM_DISCOUNTED_LATEST**: Most recent with time decay
```
Final: weight = 1.0 * time_decay(2024-05-01 → now)
```

**EWM_SUM_DISCOUNTED**: Sum all edges with time decay
```
Final: weight = time_decay(2024-04-01) * 1.0 + time_decay(2024-05-01) * 1.0
```

**Why matters**: Affects how repeated collaborations are valued:
- LATEST: Only last interaction counts
- SUM: Frequency matters (more interactions = higher weight)

---

## 6. Complete Example: End-to-End

### 6.1 Scenario

**Raw context data**:
```
2024-04-01: Alice authored CL#100, reviewed by Bob
2024-04-15: Alice authored CL#200, reviewed by Bob and Carol
2024-05-01: Diana authored CL#300, reviewed by Bob
2024-06-01: Alice authored CL#400, reviewed by Carol
```

**Configuration**:
```textproto
context_sources {
  type: "code_review"
  context_lookback { seconds: 7776000 }  # 90 days
  peer_feature_configs {
    name: "change_number"
    max_peers: 200
    bipartite_graph {
      traversal_modes: TM_FORWARD
      traversal_modes: TM_BACKWARD
      half_life { seconds: 7776000 }  # 90 days
      edge_weighting_method: EWM_DISCOUNTED_LATEST
    }
    aggregation_method: AGG_ACCUMULATE
  }
}
```

### 6.2 Step 1: Create Context Protos

```python
# Alice's contexts
Context(principal="alice", peer_attributes=[
  PeerAttribute(name="change_number", value=b"100", 
                direction=D_FORWARD, time=Timestamp(2024-04-01))
])
Context(principal="alice", peer_attributes=[
  PeerAttribute(name="change_number", value=b"200",
                direction=D_FORWARD, time=Timestamp(2024-04-15))
])
Context(principal="alice", peer_attributes=[
  PeerAttribute(name="change_number", value=b"400",
                direction=D_FORWARD, time=Timestamp(2024-06-01))
])

# Bob's contexts (reviewer)
Context(principal="bob", peer_attributes=[
  PeerAttribute(name="change_number", value=b"100",
                direction=D_BACKWARD, time=Timestamp(2024-04-01))
])
Context(principal="bob", peer_attributes=[
  PeerAttribute(name="change_number", value=b"200",
                direction=D_BACKWARD, time=Timestamp(2024-04-15))
])
Context(principal="bob", peer_attributes=[
  PeerAttribute(name="change_number", value=b"300",
                direction=D_BACKWARD, time=Timestamp(2024-05-01))
])

# Carol's contexts (reviewer)
Context(principal="carol", peer_attributes=[
  PeerAttribute(name="change_number", value=b"200",
                direction=D_BACKWARD, time=Timestamp(2024-04-15))
])
Context(principal="carol", peer_attributes=[
  PeerAttribute(name="change_number", value=b"400",
                direction=D_BACKWARD, time=Timestamp(2024-06-01))
])

# Diana's contexts (author)
Context(principal="diana", peer_attributes=[
  PeerAttribute(name="change_number", value=b"300",
                direction=D_FORWARD, time=Timestamp(2024-05-01))
])
```

### 6.3 Step 2: Build Bipartite Graphs (Separate for Forward/Backward)

**Forward Traversal Graph** (Authors → CLs):
```
alice → CL#100
alice → CL#200
alice → CL#400
diana → CL#300
```

**Backward Traversal Graph** (CLs → Reviewers):
```
CL#100 → bob
CL#200 → bob
CL#200 → carol
CL#300 → bob
CL#400 → carol
```

### 6.4 Step 3: Apply Time Decay

**Assume current time**: 2024-07-01, half-life = 90 days

```
CL#100 (91 days ago): decay = 0.5^(91/90) ≈ 0.496
CL#200 (76 days ago): decay = 0.5^(76/90) ≈ 0.557
CL#300 (61 days ago): decay = 0.5^(61/90) ≈ 0.621
CL#400 (30 days ago): decay = 0.5^(30/90) ≈ 0.794
```

**Edge weights after decay**:
```
alice → CL#100 (0.496)
alice → CL#200 (0.557)
alice → CL#400 (0.794)
diana → CL#300 (0.621)
CL#100 → bob (0.496)
CL#200 → bob (0.557)
CL#200 → carol (0.557)
CL#300 → bob (0.621)
CL#400 → carol (0.794)
```

### 6.5 Step 4: Forward Traversal - Finding Alice's Reviewers

**What forward traversal means**: When `traversal_mode: TM_FORWARD` is configured, we're following the "forward" direction of edges based on PeerAttribute directions.

**For code reviews**:
- `D_FORWARD` edge: Principal → Attribute (author authored CL)
- `D_BACKWARD` edge: Attribute → Principal (CL reviewed by reviewer)

**Edge creation logic** (from `make_edge` in pipeline_utils.py):
```python
# For TM_FORWARD with D_FORWARD attribute:
if direction == D_FORWARD:
    edge = Edge(i=principal, j=attribute, is_middle=False)  # Alice → CL#100
    
# For TM_FORWARD with D_BACKWARD attribute:
if direction == D_BACKWARD:
    edge = Edge(i=attribute, j=principal, is_middle=True)   # CL#100 → Bob
```

**Result**: Forward traversal creates 2-hop paths from authors to their reviewers.

**Step 4a: Normalize outgoing edges for each node**

After time decay, we have these edges:

**From principals** (authors):
```
alice → CL#100 (weight=0.496)
alice → CL#200 (weight=0.557)
alice → CL#400 (weight=0.794)
diana → CL#300 (weight=0.621)
```

**From middle nodes** (CLs to reviewers):
```
CL#100 → bob (weight=0.496)
CL#200 → bob (weight=0.557)
CL#200 → carol (weight=0.557)
CL#300 → bob (weight=0.621)
CL#400 → carol (weight=0.794)
```

**Normalization** (sum of outgoing edges = 1.0):

**Alice** (3 outgoing edges, total = 0.496 + 0.557 + 0.794 = 1.847):
```
alice → CL#100: 0.496 / 1.847 = 0.269
alice → CL#200: 0.557 / 1.847 = 0.301  
alice → CL#400: 0.794 / 1.847 = 0.430
```

**Diana** (1 outgoing edge, total = 0.621):
```
diana → CL#300: 0.621 / 0.621 = 1.0
```

**CL#100** (1 outgoing edge):
```
CL#100 → bob: 0.496 / 0.496 = 1.0
```

**CL#200** (2 outgoing edges, total = 0.557 + 0.557 = 1.114):
```
CL#200 → bob: 0.557 / 1.114 = 0.5
CL#200 → carol: 0.557 / 1.114 = 0.5
```

**CL#300** (1 outgoing edge):
```
CL#300 → bob: 0.621 / 0.621 = 1.0
```

**CL#400** (1 outgoing edge):
```
CL#400 → carol: 0.794 / 0.794 = 1.0
```

**Step 4b: Group by middle node and separate incoming/outgoing**

The algorithm reorganizes edges by middle node:

**Middle node: CL#100**
```
Incoming (from authors): 
  - alice (normalized weight: 0.269)
  
Outgoing (to reviewers):
  - bob (normalized weight: 1.0)
```

**Middle node: CL#200**
```
Incoming (from authors):
  - alice (normalized weight: 0.301)
  
Outgoing (to reviewers):
  - bob (normalized weight: 0.5)
  - carol (normalized weight: 0.5)
```

**Middle node: CL#300**
```
Incoming (from authors):
  - diana (normalized weight: 1.0)
  
Outgoing (to reviewers):
  - bob (normalized weight: 1.0)
```

**Middle node: CL#400**
```
Incoming (from authors):
  - alice (normalized weight: 0.430)
  
Outgoing (to reviewers):
  - carol (normalized weight: 1.0)
```

**Step 4c: Compute 2-hop path weights (cross-product)**

For each middle node, multiply every incoming weight by every outgoing weight:

**Via CL#100**:
```
alice → CL#100 → bob
Path weight: 0.269 × 1.0 = 0.269
```

**Via CL#200**:
```
alice → CL#200 → bob
Path weight: 0.301 × 0.5 = 0.151

alice → CL#200 → carol  
Path weight: 0.301 × 0.5 = 0.151
```

**Via CL#300**:
```
diana → CL#300 → bob
Path weight: 1.0 × 1.0 = 1.0
```

**Via CL#400**:
```
alice → CL#400 → carol
Path weight: 0.430 × 1.0 = 0.430
```

**Step 4d: Accumulate paths with same (start, end)**

Combine weights for multiple paths between the same pair:

```python
path_weights = {
  (alice, bob): 0.269 + 0.151 = 0.420,    # Two paths: via CL#100, CL#200
  (alice, carol): 0.151 + 0.430 = 0.581,  # Two paths: via CL#200, CL#400
  (diana, bob): 1.0                        # One path: via CL#300
}
```

**Interpretation**:
- Alice reaches Bob with combined weight 0.420 (Bob reviewed CL#100 and CL#200)
- Alice reaches Carol with combined weight 0.581 (Carol reviewed CL#200 and CL#400)
- Diana reaches Bob with weight 1.0 (Bob reviewed CL#300)

**Step 4e: Group by destination (final neighbor lists)**

Reorganize by which principal (reviewer) is being reached:

**Bob's incoming paths** (who does Bob review?):
```
neighbors_of_bob = [
  (1.0, diana),    # Strongest connection
  (0.420, alice)
]
```

**Carol's incoming paths** (who does Carol review?):
```
neighbors_of_carol = [
  (0.581, alice)
]
```

**Step 4f: Select top-K neighbors**

Configuration: `max_neighbors = 200`

Since both Bob and Carol have fewer than 200 neighbors, all are kept:

```python
final_neighbors = {
  bob: [(1.0, diana), (0.420, alice)],
  carol: [(0.581, alice)]
}
```

**What this means**:
- **For Bob** (as a reviewer): His forward-traversal peers are Diana (strongest) and Alice (weaker)
  - These are the people whose code Bob reviews
- **For Carol** (as a reviewer): Her forward-traversal peer is Alice  
  - This is the person whose code Carol reviews

### 6.6 Step 5: Backward Traversal - Finding Authors' Peers

**What backward traversal means**: When `traversal_mode: TM_BACKWARD` is configured, we follow edges in the "backward" direction.

**Edge creation logic** (from `make_edge` in pipeline_utils.py):
```python
# For TM_BACKWARD with D_FORWARD attribute:
if direction == D_FORWARD:
    edge = Edge(i=attribute, j=principal, is_middle=True)   # CL#100 → Alice

# For TM_BACKWARD with D_BACKWARD attribute:  
if direction == D_BACKWARD:
    edge = Edge(i=principal, j=attribute, is_middle=False)  # Bob → CL#100
```

**Result**: Backward traversal creates 2-hop paths from reviewers to authors OR from authors through reviewers to other authors.

**Step 5a: Build backward edges**

**From reviewers to CLs** (is_middle=False):
```
bob → CL#100 (weight=0.496 after decay)
bob → CL#200 (weight=0.557)
bob → CL#300 (weight=0.621)
carol → CL#200 (weight=0.557)
carol → CL#400 (weight=0.794)
```

**From CLs to authors** (is_middle=True):
```
CL#100 → alice (weight=0.496)
CL#200 → alice (weight=0.557)
CL#300 → diana (weight=0.621)
CL#400 → alice (weight=0.794)
```

**Step 5b: Normalize**

**Bob** (3 outgoing edges, total = 0.496 + 0.557 + 0.621 = 1.674):
```
bob → CL#100: 0.496 / 1.674 = 0.296
bob → CL#200: 0.557 / 1.674 = 0.333
bob → CL#300: 0.621 / 1.674 = 0.371
```

**Carol** (2 outgoing edges, total = 0.557 + 0.794 = 1.351):
```
carol → CL#200: 0.557 / 1.351 = 0.412
carol → CL#400: 0.794 / 1.351 = 0.588
```

**CL#100** (1 outgoing to alice):
```
CL#100 → alice: 1.0
```

**CL#200** (1 outgoing to alice):
```
CL#200 → alice: 1.0
```

**CL#300** (1 outgoing to diana):
```
CL#300 → diana: 1.0
```

**CL#400** (1 outgoing to alice):
```
CL#400 → alice: 1.0
```

**Step 5c: Group by middle node and compute cross-product**

**Via CL#100**:
```
bob → CL#100 → alice
Path weight: 0.296 × 1.0 = 0.296
```

**Via CL#200**:
```
bob → CL#200 → alice
Path weight: 0.333 × 1.0 = 0.333

carol → CL#200 → alice
Path weight: 0.412 × 1.0 = 0.412
```

**Via CL#300**:
```
bob → CL#300 → diana
Path weight: 0.371 × 1.0 = 0.371
```

**Via CL#400**:
```
carol → CL#400 → alice
Path weight: 0.588 × 1.0 = 0.588
```

**Step 5d: Accumulate**:
```python
path_weights = {
  (bob, alice): 0.296 + 0.333 = 0.629,   # Bob → alice via CL#100, CL#200
  (bob, diana): 0.371,                    # Bob → diana via CL#300
  (carol, alice): 0.412 + 0.588 = 1.0    # Carol → alice via CL#200, CL#400
}
```

**Step 5e: Group by destination**:

**Alice's incoming paths** (whose code does Alice author?):
```
neighbors_of_alice = [
  (1.0, carol),     # Strongest connection
  (0.629, bob)
]
```

**Diana's incoming paths**:
```
neighbors_of_diana = [
  (0.371, bob)
]
```

### 6.7 Step 6: Generate Final Features

The algorithm runs **both forward and backward traversals separately**, generating two sets of features per principal.

**For Alice** (as an author):

**Forward traversal result** (`code_review/change_number_f`):
- Alice's peers: Bob (0.420), Carol (0.581)
- Interpretation: "Who reviews Alice's code?"

**Backward traversal result** (`code_review/change_number_b`):
- Alice's peers: Carol (1.0), Bob (0.629)  
- Interpretation: "Who else has reviewers similar to Alice's reviewers?"
  - (This captures: Carol and Bob are connected to Alice through the bipartite structure)

**Generated features**:
```python
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
      WeightedToken(token=b"bob", weight=0.420),
      WeightedToken(token=b"carol", weight=0.581)
    ]
  )
),
Feature(
  name="code_review/change_number_b/t",
  bag_of_weighted_words=WeightedTokens(
    tokens=[
      WeightedToken(token=b"carol", weight=1.0),
      WeightedToken(token=b"bob", weight=1.0)
    ]
  )
),
Feature(
  name="code_review/change_number_b/w",
  bag_of_weighted_words=WeightedTokens(
    tokens=[
      WeightedToken(token=b"carol", weight=1.0),
      WeightedToken(token=b"bob", weight=0.629)
    ]
  )
)
```

**For Bob** (as a reviewer):

**Forward traversal result**:
- Bob's peers: Diana (1.0), Alice (0.420)
- Interpretation: "Whose code does Bob review?"

**Backward traversal result**:
- Bob's peers: Alice (0.629), Diana (0.371)
- Interpretation: Similar pattern from different graph perspective

**Key insight**: The `/t` features contain WHO (peer identities as tokens), while `/w` features contain HOW MUCH (collaboration strength as weights). Both are used together during model training.

### 6.6 Step 5: Generate Final Features

**For Alice at snapshot 2024-07-01**:
```python
FeaturizedContext(
  valid_from=Timestamp(2024-07-01 00:00:00),
  features_per_source=[
    ContextSourceFeatures(
      source_type="code_review",
      features=[
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
              WeightedToken(token=b"bob", weight=0.42),  # Combined path weight
              WeightedToken(token=b"carol", weight=0.58)
            ]
          )
        )
      ]
    )
  ]
)
```

**Key**: The `/t` feature contains WHO (bob, carol), the `/w` feature contains HOW MUCH (weights representing collaboration intensity)

---

## 7. Why This Matters: The Generalization Power

### 7.1 Without Bipartite Graphs

**Simple approach**: "Has Alice accessed this resource before?"

**Problem**:
```
Training: Alice accessed doc_ML_101, doc_ML_102
Inference: Alice accesses doc_ML_103 (NEW RESOURCE)
→ No history data
→ Can't make good prediction
```

### 7.2 With Bipartite Graphs

**Facade approach**: "What do Alice's collaborators access?"

**Process**:
```
1. Build Alice's social network via code reviews
   → Peers: Bob, Carol (via bipartite graph)

2. Training observes:
   → Bob accessed doc_ML_101, doc_ML_102, doc_ML_103
   → Carol accessed doc_ML_102, doc_ML_103
   → Model learns: "People with peers like Bob/Carol access ML docs"

3. Inference: Alice accesses doc_ML_103
   → Alice has peers Bob/Carol
   → Bob/Carol access ML docs
   → doc_ML_103 is likely normal for Alice
   → HIGH similarity score (normal)
```

**Result**: Generalization to unseen resources based on social network!

### 7.3 The Magic Formula

```
Social Network (from bipartite graph)
  × Resource Access Patterns (from history)
  = Prediction for New Resources
```

**This is why Facade works**: It doesn't just learn individual patterns, it learns social-network-based patterns.

---

## 8. Implementation Details from fold.py

### 8.1 Weight Perturbation

```python
noise = random.uniform(0, weight_perturb_scale)
weight_with_noise = weight + noise
```

**Why**: Break ties deterministically for reproducibility
- Very small noise (1e-8) doesn't affect results
- Ensures consistent ordering when weights are equal

### 8.2 Top-K Selection at Middle Nodes

```python
max_edges_from_middle = 1000
top_incoming = incoming_edges[:max_edges_from_middle]
top_outgoing = outgoing_edges[:max_edges_from_middle]
```

**Why**: Very popular middle nodes (e.g., large code reviews with 100+ participants) would create huge cross-products
- Limit to top 1000 incoming and 1000 outgoing edges
- Prevents O(N²) explosion
- Focuses on strongest connections

### 8.3 Final Top-K Selection

```python
max_neighbors = 200
final_neighbors[dest_node] = neighbors[:max_neighbors]
```

**Why**: Each principal limited to 200 peers
- Memory efficiency
- Model efficiency (fewer embeddings to process)
- Signal-to-noise (weak connections filtered out)

---

## 9. Designing Your Own Bipartite Graphs

### 9.1 Choosing Middle Nodes

**Good middle nodes**:
- ✅ Shared activities: code reviews, meetings, projects
- ✅ Shared attributes: team membership, office location
- ✅ Meaningful relationships: indicates collaboration or similarity

**Bad middle nodes**:
- ❌ Too common: "all employees" (everyone connected to everyone)
- ❌ Too rare: middle nodes with only 1 principal (no peers)
- ❌ Not meaningful: random groupings

### 9.2 Directional vs Undirectional

**Use directional** (TM_FORWARD + TM_BACKWARD) when:
- Asymmetric relationships (author/reviewer, sender/recipient)
- Want to capture different roles
- Example: Email (sent vs received), Code reviews (authored vs reviewed)

**Use undirectional** (TM_UNDIRECTED) when:
- Symmetric relationships (meeting attendance, team membership)
- Role doesn't matter
- Example: Projects, calendar events, shared resources

### 9.3 Setting max_peers

**Guidelines**:
- Start with 100-200 (typical professional network size)
- Increase if your organization has cross-functional roles
- Decrease if you want tighter peer groups
- Monitor: What % of people have > max_peers connections?

---

## 10. Summary

### Core Concepts

1. **Bipartite graph**: Principals ↔ Middle Nodes ↔ Principals
2. **Two-hop walk**: Find peers via shared middle nodes
3. **Random walk normalization**: Fair probability distribution
4. **Time decay**: Recent connections weighted more
5. **Top-K selection**: Focus on strongest relationships

### The Algorithm

```
1. Build edges from PeerAttributes
2. Filter near-zero weights
3. Normalize outgoing edges (random walk probabilities)
4. Group by middle nodes (incoming vs outgoing)
5. Compute cross-product (2-hop path weights)
6. Sum paths between same (start, end)
7. Select top-K neighbors per principal
```

### Why It Works

**Generalization**: Social network features enable predictions for unseen resources

**Robustness**: Multiple paths to peers (redundancy)

**Interpretability**: Features are actual collaborators (not abstract latent factors)

---

## Next Steps

**Document 5** will show how these bipartite graph features are combined with history features and merged with actions to create training data.

**Ready for Document 5: Feature Engineering Pipeline**
