# Document 7: Inference and Scoring

## Introduction

This document explains how to use a trained Facade model to score new actions in production. We'll cover:
1. Model loading and serving signatures
2. Inference pipeline execution
3. Score interpretation and thresholding
4. Production deployment considerations

---

## 1. Overview of Inference Process

### 1.1 High-Level Flow

```
Raw Data (Actions + Contexts)
  ↓
Feature Engineering Pipeline
  - History featurization
  - Bipartite graph computation
  - Feature merging
  ↓
ContextualizedActions → tf.SequenceExample
  ↓
Trained Model (SavedModel)
  - Context tower
  - Action tower
  - Scoring function
  ↓
Scores (per action)
  ↓
Threshold Application
  ↓
Alerts (anomalous actions)
```

### 1.2 Inference vs Training Differences

| Aspect | Training | Inference |
|--------|----------|-----------|
| Input | Historical data (with labels) | Real-time or batch data |
| Pipeline Type | `PipelineType.TRAIN` | `PipelineType.INFERENCE` |
| Dropout | Enabled | Disabled |
| Contrastive Pairs | Generated | Not needed |
| Output | Model weights | Scores per action |
| Frequency | Once | Continuous |

---

## 2. Running Inference

### 2.1 Command-Line Invocation

**Script**: `batch/inference_main.py`

```bash
python -m batch.inference_main \
  --directive=sample/directive.textproto \
  --start_time="2024-07-08 00:00:00" \
  --end_time="2024-07-15 00:00:00" \
  --action_path=sample/action.tfrecord \
  --context_path=sample/context.tfrecord \
  --output_file=sample/scores.tfrecord \
  --model_config=sample/config.textproto \
  --model_dir=sample/model
```

**Parameters**:
- `--directive`: Same directive.textproto used during training
- `--start_time` / `--end_time`: Time window to score
- `--action_path`: TFRecord file with Action protos
- `--context_path`: TFRecord file with Context protos
- `--output_file`: Where to write Score protos
- `--model_config`: Same config.textproto used during training
- `--model_dir`: Directory containing trained model

### 2.2 Step-by-Step Process

**Step 1: Load configurations**
```python
directive = directive_utils.read_directive('sample/directive.textproto')
model_config = configuration.read_config('sample/config.textproto')
```

**Step 2: Run feature engineering pipeline**
```python
contextualized_actions = batch_lib.compute_contextualized_actions(
  directive=directive,
  pipeline_type=PipelineType.INFERENCE,
  start_time=datetime(2024, 7, 8),
  end_time=datetime(2024, 7, 15),
  action_path='sample/action.tfrecord',
  context_path='sample/context.tfrecord'
)
```

**This executes**:
- History featurization (event timelines)
- Bipartite graph computation (peer features)
- Merger (snapshot alignment)

**Output**: List of ContextualizedActions

**Step 3: Convert to tf.SequenceExample**
```python
sequence_examples = []
for ca in contextualized_actions:
  seq_ex = tf_example.to_tf_input(ca)
  sequence_examples.append(seq_ex.SerializeToString())
```

**Step 4: Load trained model**
```python
with keras.utils.custom_object_scope({'OneCycle': one_cycle}):
  model = tf.keras.models.load_model('sample/model/export/final')
```

**Custom object scope**: Required because OneCycle is a custom LR schedule

**Step 5: Get serving function**
```python
serving_fn = model.signatures['serving_default']
```

**Step 6: Run inference**
```python
results = serving_fn(inputs=sequence_examples)
```

**Step 7: Extract scores**
```python
scores = []
for action_type in model_config.action_name_to_architecture:
  action_type_scores = results[action_type + '_scores']
  for score in action_type_scores:
    scores.append(Score(
      principal=...,
      action_type=action_type,
      resource_id=...,
      score=score
    ))
```

**Step 8: Write scores to file**
```python
with tf.io.TFRecordWriter('sample/scores.tfrecord') as writer:
  for s in scores:
    writer.write(s.SerializeToString())
```

---

## 3. Model Loading and Serving

### 3.1 SavedModel Format

**Directory structure**:
```
sample/model/export/final/
├── saved_model.pb         # Model graph
├── variables/
│   ├── variables.data-00000-of-00001  # Weights
│   └── variables.index
└── assets/                # Additional resources
```

**Loading**:
```python
model = tf.keras.models.load_model('sample/model/export/final')
```

**Loads**:
- Model architecture (towers, layers)
- Trained weights (embeddings, SNN parameters)
- Serving signature (input/output specs)

### 3.2 Serving Signature

**What is it?**: Interface specification for model inference

**Example signature**:
```python
serving_default(inputs: List[str]) -> Dict[str, Tensor]
```

**Input**: List of serialized tf.SequenceExample
**Output**: Dictionary with keys like `"doc_access_scores"`, `"code_review_scores"`

**Accessing**:
```python
serving_fn = model.signatures['serving_default']
```

**Calling**:
```python
results = serving_fn(inputs=[seq_ex_1, seq_ex_2, ...])
```

**Results structure**:
```python
{
  'doc_access_scores': tf.Tensor([0.12, 0.85, 0.03, ...], shape=(N,)),
  'code_review_scores': tf.Tensor([0.21, 0.67, ...], shape=(M,))
}
```

**N, M**: Number of actions of each type across all contexts

---

## 4. Score Interpretation

### 4.1 The Score Proto

**Definition** (from `protos/score.proto`):
```protobuf
message Score {
  string principal = 1;
  string action_type = 2;
  string resource_id = 3;
  bytes action_id = 4;
  float score = 5;  // In (0, 1), higher = more anomalous
}
```

**Example**:
```textproto
principal: "alice@company.com"
action_type: "doc_access"
resource_id: "confidential_doc_789"
action_id: "\x01\x02\x03..."
score: 0.87
```

### 4.2 Score Range and Meaning

**Score ∈ (0, 1)**: Normalized anomaly score

**Interpretation**:
- **0.0 - 0.3**: Normal behavior (expected access)
- **0.3 - 0.7**: Moderate anomaly (investigate if unusual for user)
- **0.7 - 1.0**: High anomaly (strong alert)

**How scores are derived**:

**During training**, model learns:
```
Positive pairs (actual occurrences): Low score
Negative pairs (synthetic): High score
```

**During inference**:
```
Context embedding: Represents "normal" behavior for this principal
Action embedding: Represents the specific access

Similarity = dot_product(context_emb, action_emb)

If scoring_function = SF_OMDOT:
  score = 1 - similarity
```

**Result**:
- **High similarity** (normal access) → **Low score**
- **Low similarity** (anomalous) → **High score**

### 4.3 Concrete Examples

**Example 1: Normal access**
```
Principal: alice@company.com
Action: Access doc_123
Context features:
  - Alice frequently accesses docs in same project
  - Alice's peers (Bob, Carol) also access doc_123
  - Time: During work hours

Context embedding: [0.15, 0.22, 0.31, ...]
Action embedding:  [0.14, 0.23, 0.29, ...]

Similarity: 0.92  (very similar)
Score: 1 - 0.92 = 0.08  (LOW - normal)
```

**Example 2: Anomalous access**
```
Principal: alice@company.com
Action: Access confidential_salary_data
Context features:
  - Alice works in engineering (no salary access in history)
  - Alice's peers: No one has accessed this resource
  - Time: 2 AM (unusual)

Context embedding: [0.15, 0.22, 0.31, ...]
Action embedding:  [0.82, -0.11, 0.05, ...]

Similarity: 0.13  (very different)
Score: 1 - 0.13 = 0.87  (HIGH - anomalous)
```

**Example 3: New resource, but normal pattern**
```
Principal: bob@company.com
Action: Access new_project_doc (never accessed before)
Context features:
  - Bob's peers (Alice, Carol) have accessed new_project_doc
  - Bob recently joined project team
  - Similar docs accessed by Bob

Context embedding: [0.21, 0.18, 0.44, ...]
Action embedding:  [0.19, 0.17, 0.42, ...]

Similarity: 0.81  (similar, due to peer features)
Score: 1 - 0.81 = 0.19  (LOW - normal despite newness)
```

**Key insight**: Score reflects context, not just history

---

## 5. Threshold Selection

### 5.1 FPR-Based Thresholding

**From training/validation**:

**Compute ROC curve**:
```python
# During validation
labels = [0, 0, 1, 0, 1, ...]  # 0 = normal, 1 = anomaly
scores = [0.05, 0.12, 0.78, 0.09, 0.85, ...]

fpr, tpr, thresholds = roc_curve(labels, scores)
```

**Choose threshold for desired FPR**:
```python
# Want FPR = 1% (1 false alarm per 100 normal actions)
target_fpr = 0.01
idx = np.argmin(np.abs(fpr - target_fpr))
threshold = thresholds[idx]  # e.g., 0.65
```

**In production**:
```python
if score > 0.65:
  alert(action)  # Anomaly detected
```

### 5.2 Percentile-Based Thresholding

**Alternative approach**: Set threshold based on score distribution

**From historical scores**:
```python
historical_scores = [0.05, 0.12, 0.08, 0.15, 0.92, ...]
threshold = np.percentile(historical_scores, 99)  # 99th percentile
# threshold ≈ 0.75
```

**Interpretation**: Alert on top 1% most anomalous actions

### 5.3 Per-Principal Thresholding

**Advanced approach**: Different thresholds per user

**Rationale**:
- Some users have naturally more variable behavior
- Executives may access wider range of resources

**Implementation**:
```python
thresholds = {
  'alice@company.com': 0.70,
  'bob@company.com': 0.50,  # Lower threshold (more sensitive)
  'ceo@company.com': 0.85   # Higher threshold (less sensitive)
}

if score > thresholds[principal]:
  alert(action)
```

---

## 6. Reading Scores

### 6.1 Using read_scores_main.py

**Script**: `batch/read_scores_main.py`

```bash
python -m batch.read_scores_main \
  --score_file=sample/scores.tfrecord \
  --top_n=20
```

**Output**:
```
principal: "alice@company.com"
action_type: "doc_access"
resource_id: "confidential_doc_789"
score: 0.87

principal: "bob@company.com"
action_type: "code_review"
resource_id: "change_456"
score: 0.81

...
```

**Sorted by score** (highest first)

### 6.2 Custom Analysis

**Load scores programmatically**:
```python
import tensorflow as tf
from protos import score_pb2

reader = tf.data.TFRecordDataset(['sample/scores.tfrecord'])
scores = []
for raw_record in reader:
  score = score_pb2.Score()
  score.ParseFromString(raw_record.numpy())
  scores.append(score)
```

**Filter by principal**:
```python
alice_scores = [s for s in scores if s.principal == 'alice@company.com']
```

**Aggregate statistics**:
```python
import numpy as np

all_scores = [s.score for s in scores]
print(f"Mean: {np.mean(all_scores)}")
print(f"Median: {np.median(all_scores)}")
print(f"95th percentile: {np.percentile(all_scores, 95)}")
```

**Identify outliers**:
```python
threshold = 0.70
anomalies = [s for s in scores if s.score > threshold]
print(f"Found {len(anomalies)} anomalies")
```

---

## 7. Production Deployment

### 7.1 Facade's Batch Architecture

**Important**: Facade is designed for **batch processing**, not real-time streaming.

**Why batch-only?**
1. **Bipartite graph computation**: Expensive to compute, requires historical context data
2. **History featurization**: Needs complete action history for time window
3. **Feature alignment**: Snapshot-based merging of contexts and actions
4. **File-based I/O**: Reads TFRecords, writes TFRecords

**Typical use case**: Daily or hourly scoring of recent activity

**Real-time adaptation**: Would require significant architectural changes (covered in advanced topics)

### 7.2 Batch Deployment Pattern

**Architecture**:
```
Daily Cron Job
  ↓
Read yesterday's actions/contexts
  ↓
Run inference_main.py
  ↓
Write scores to database
  ↓
Analysts query high-scoring actions
```

**Example cron**:
```bash
0 2 * * * python -m batch.inference_main \
  --start_time="$(date -d 'yesterday' '+%Y-%m-%d 00:00:00')" \
  --end_time="$(date -d 'today' '+%Y-%m-%d 00:00:00')" \
  --output_file=/data/scores/$(date '+%Y%m%d').tfrecord
```

### 7.3 Understanding Data Requirements for Incremental Scoring

**Critical insight**: To score yesterday's actions, you need **historical data**, not just yesterday's events.

**Why?** The history featurization process builds features like:
- "Alice accessed this document 5 times in the last 30 days"
- "Bob typically accesses engineering resources"

**From the code** (`action/action_source.py`):
```python
history_duration = convert_proto_duration_to_timedelta(config.history_duration)
earliest_event = start_time - history_duration  # Goes back in time!
raw_actions = read_actions(tfrecord_path, earliest_event, end_time)
```

**What this means**:
- To score July 1 actions
- With `history_duration` = 90 days (from directive.textproto)
- You need actions from **April 1 to July 1** (90 days + 1)
- But only July 1 actions are scored (the rest provide historical context)

**Data requirements for daily scoring**:
- **Actions**: 90 days of history + 1 target day
- **Contexts**: 90 days (for bipartite graph computation)

### 7.4 Incremental Processing with Rolling Windows

**Recommended approach**: Maintain daily files and concatenate a rolling window

**Step 1: Daily data generation**
```bash
#!/bin/bash
# Generate yesterday's raw action and context data from logs

YESTERDAY=$(date -d 'yesterday' '+%Y%m%d')

# Extract actions from logs (your log processing pipeline)
python scripts/extract_actions_from_logs.py \
  --date="yesterday" \
  --output=data/actions/day_$YESTERDAY.tfrecord

# Extract contexts from logs (collaboration events, meetings, etc.)
python scripts/extract_contexts_from_logs.py \
  --date="yesterday" \
  --output=data/contexts/day_$YESTERDAY.tfrecord
```

**Step 2: Rolling window assembly**
```bash
#!/bin/bash
# Concatenate last 90 days of data

# For actions (90 days for history)
ls -t data/actions/day_*.tfrecord | head -90 | \
  xargs cat > data/actions/rolling_90days.tfrecord

# For contexts (90 days for bipartite graphs)
ls -t data/contexts/day_*.tfrecord | head -90 | \
  xargs cat > data/contexts/rolling_90days.tfrecord

# Clean up old files (keep 100 days for safety)
find data/actions/ -name "day_*.tfrecord" -mtime +100 -delete
find data/contexts/ -name "day_*.tfrecord" -mtime +100 -delete
```

**Step 3: Run inference**
```bash
#!/bin/bash
# Score yesterday's actions using 90-day history

START=$(date -d 'yesterday' '+%Y-%m-%d 00:00:00')
END=$(date '+%Y-%m-%d 00:00:00')
OUTPUT=data/scores/$(date -d 'yesterday' '+%Y%m%d').tfrecord

python -m batch.inference_main \
  --directive=config/directive.textproto \
  --start_time="$START" \
  --end_time="$END" \
  --action_path=data/actions/rolling_90days.tfrecord \
  --context_path=data/contexts/rolling_90days.tfrecord \
  --output_file="$OUTPUT" \
  --model_config=config/config.textproto \
  --model_dir=models/production/
```

**Complete daily workflow**:
```bash
#!/bin/bash
# complete_daily_scoring.sh

set -e  # Exit on error

echo "=== Daily Facade Scoring Pipeline ==="
DATE=$(date -d 'yesterday' '+%Y%m%d')

# Step 1: Generate yesterday's data
echo "Generating data for $DATE..."
./scripts/01_extract_daily_data.sh

# Step 2: Assemble rolling windows
echo "Assembling rolling windows..."
./scripts/02_assemble_rolling_windows.sh

# Step 3: Run inference
echo "Running inference..."
./scripts/03_run_inference.sh

# Step 4: Load scores to database/alerting system
echo "Loading scores..."
python scripts/load_scores_to_db.py \
  --score_file=data/scores/$DATE.tfrecord \
  --date=$DATE

echo "=== Pipeline Complete ==="
```

**Why this works**:
- ✅ Only generates new data daily (efficient)
- ✅ Maintains rolling 90-day window automatically
- ✅ History features always have fresh data
- ✅ Bipartite graphs computed from recent collaborations
- ✅ Easy to debug (each day is a separate file)

### 7.5 Alternative: Full Regeneration (Simpler but Less Efficient)

**If rolling windows are too complex**, regenerate everything daily:

```bash
#!/bin/bash
# Simpler approach: regenerate 90-day window daily

START=$(date -d '90 days ago' '+%Y-%m-%d 00:00:00')
SCORE_START=$(date -d 'yesterday' '+%Y-%m-%d 00:00:00')
END=$(date '+%Y-%m-%d 00:00:00')

# Regenerate 90 days of actions
python scripts/extract_actions_from_logs.py \
  --start="$START" \
  --end="$END" \
  --output=data/actions_90days.tfrecord

# Regenerate 90 days of contexts
python scripts/extract_contexts_from_logs.py \
  --start="$START" \
  --end="$END" \
  --output=data/contexts_90days.tfrecord

# Run inference
python -m batch.inference_main \
  --start_time="$SCORE_START" \
  --end_time="$END" \
  --action_path=data/actions_90days.tfrecord \
  --context_path=data/contexts_90days.tfrecord \
  --output_file=data/scores/$(date -d yesterday +%Y%m%d).tfrecord
```

**Trade-offs**:
- ✅ Simpler (no file management)
- ✅ Always fresh data
- ❌ Regenerates 89 days of duplicate data daily (wasteful)
- ❌ Slower (more log processing)

**When to use**: Small datasets, or when log processing is very fast

---

## 8. Monitoring and Evaluation

### 8.1 Score Distribution Drift

**Monitor over time**:
```python
# Week 1
scores_week1 = [0.05, 0.12, 0.08, ...]
mean_week1 = np.mean(scores_week1)  # 0.15

# Week 5
scores_week5 = [0.35, 0.42, 0.31, ...]
mean_week5 = np.mean(scores_week5)  # 0.38

if mean_week5 > mean_week1 * 1.5:
  alert("Score distribution shifted - model may need retraining")
```

**Causes**:
- Organizational changes (new systems, processes)
- Model staleness (behavior patterns evolve)
- Data quality issues

### 8.2 Alert Investigation Rate

**Track true positives vs false positives**:
```python
alerts_generated = 100
true_positives = 12  # Confirmed insider threats
false_positives = 88  # Normal behavior flagged

precision = 12 / 100 = 0.12  # 12% precision
```

**Adjust threshold**:
```python
# If precision too low, increase threshold
new_threshold = 0.80  # Was 0.70
```

### 8.3 Model Retraining

**When to retrain**:
1. Score drift detected
2. Precision degradation
3. New action types introduced
4. Organizational changes

**Process**:
1. Collect new action/context data
2. Re-run dataset_maker_main.py
3. Re-run train_main.py
4. Validate on holdout set
5. Deploy new model

**Frequency**: Monthly or quarterly

---

## 9. Advanced Scoring Techniques

### 9.1 Ensembling

**Train multiple models with different configurations**:
```python
model_1 = load_model('model_v1/')  # Config 1
model_2 = load_model('model_v2/')  # Config 2

scores_1 = model_1(inputs)
scores_2 = model_2(inputs)

# Average scores
final_scores = (scores_1 + scores_2) / 2
```

**Benefits**: More robust predictions

### 9.2 Explainability

**Identify which features drive high scores**:

**Approach 1: Feature ablation**
```python
# Baseline score
score_full = model(full_features)  # 0.85

# Remove forward peer features
score_no_forward = model(features_without_forward_peers)  # 0.62

# Forward peers contribute: 0.85 - 0.62 = 0.23
```

**Approach 2: Attention mechanisms** (requires model modification)
- Add attention layers to tower architecture
- Visualize attention weights per feature

### 9.3 Feedback Loops

**Incorporate analyst feedback**:
```python
# Analyst confirms this was a true anomaly
score_proto.label = LABEL_ANOMALY
score_proto.analyst_comment = "Unauthorized access attempt"

# Add to retraining dataset as positive example
```

**Active learning**: Prioritize labeling high-uncertainty cases

---

## 10. Common Issues and Debugging

### 10.1 All Scores Near 0.5

**Symptom**: Scores clustered around 0.5 (no discrimination)

**Causes**:
- Model didn't converge during training
- Features not informative
- Scoring function mismatch

**Debug**:
```python
# Check model metrics
metrics = model.evaluate(validation_dataset)
print(metrics['tpr_at_1pct_fpr'])  # Should be > 0.5
```

### 10.2 Scores Too High/Low

**Symptom**: All scores > 0.9 or all scores < 0.1

**Causes**:
- Wrong scoring function (SF_DOT vs SF_OMDOT)
- Normalization issue

**Fix**:
```python
# Check config
if config.scoring_function == SF_DOT:
  # Should be SF_OMDOT for anomaly scores
  update config and retrain
```

### 10.3 Inference Crashes

**Error**: `InvalidArgumentError: Incompatible shapes`

**Cause**: Model expects different feature structure than provided

**Debug**:
```python
# Print expected vs actual features
print(model.input_spec)
print(sequence_example.feature_lists.feature_list.keys())
```

**Fix**: Ensure directive and config match training configuration

---

## 11. Summary

### Inference Pipeline

1. **Input**: Action/Context TFRecords, trained model
2. **Feature Engineering**: Same pipeline as training (PipelineType.INFERENCE)
3. **Model Loading**: Load SavedModel with serving signature
4. **Scoring**: Run inference, extract scores per action
5. **Output**: Score protos with anomaly scores

### Score Interpretation

- **Range**: (0, 1) where higher = more anomalous
- **Threshold**: Set based on FPR tolerance (e.g., 1%)
- **Context-aware**: Scores reflect both history and social network

### Deployment Options

- **Batch**: Daily/hourly scoring jobs
- **Real-time**: Model serving + streaming pipeline
- **Hybrid**: Pre-compute features, real-time scoring

### Monitoring

- **Score distribution**: Detect drift
- **Alert metrics**: Precision, recall
- **Retraining**: When behavior patterns change

### Next Steps

**Document 8** will provide a complete step-by-step reconstruction guide, walking through an end-to-end example from raw data to trained model to inference.

**Ready for Document 8: Step-by-Step Reconstruction Guide**
