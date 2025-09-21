# Copyright 2025 github.com/cerebraljam
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This code is derived from the Facade project by Google Inc.
# Original project: https://github.com/google/facade

"""
Custom Facade Implementation Template

This module provides custom implementations for your specific use case,
building on top of the Facade framework developed by Google.
"""

import os
import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import tensorflow as tf
from google.protobuf import text_format

from model import model as model_lib
from model.common import configuration
from batch import batch_lib
from common import directive_utils
from protos import config_pb2


@dataclass
class CustomConfig:
    """Configuration for your organization's Facade deployment."""
    
    # Model configuration
    embedding_dims: int = 128
    batch_size: int = 256
    learning_rate: float = 0.04
    
    # Action types specific to your environment
    action_types: List[str] = None
    
    # Data paths
    data_dir: str = "./data"
    model_dir: str = "./models"
    config_dir: str = "./config"
    
    # Training schedule
    training_examples: int = 3000000
    validation_split: float = 0.2
    
    # Incremental training
    incremental_training: bool = True
    incremental_learning_rate: float = 0.01
    
    def __post_init__(self):
        if self.action_types is None:
            self.action_types = ["documents", "database", "api_calls", "file_access"]


class CustomModelBuilder:
    """Builds and configures Facade models for your organization's environment."""
    
    def __init__(self, config: CustomConfig):
        self.config = config
        self.model_config = self._create_model_config()
    
    def _create_model_config(self) -> config_pb2.ModelHyperparameters:
        """Creates the protobuf configuration for the model."""
        model_config = config_pb2.ModelHyperparameters()
        
        # Basic model parameters
        model_config.embedding_dims = self.config.embedding_dims
        model_config.scoring_function = config_pb2.ScoringFunction.SF_DOT
        model_config.principal_feature_name = "user_id"
        
        # Transformations
        model_config.action_embeddings_transformations.append(
            config_pb2.Transformation.TR_SOFTPLUS
        )
        model_config.context_embeddings_transformations.append(
            config_pb2.Transformation.TR_SOFTPLUS
        )
        
        # Token embeddings
        self._configure_token_embeddings(model_config)
        
        # Context architecture
        self._configure_context_architecture(model_config)
        
        # Action architectures
        self._configure_action_architectures(model_config)
        
        # Training hyperparameters
        self._configure_training_hyperparameters(model_config)
        
        return model_config
    
    def _configure_token_embeddings(self, model_config):
        """Configure token embedding dimensions for different feature types."""
        embeddings = {
            "users": 64,           # User identifiers
            "resources": 128,      # Resource identifiers  
            "departments": 32,     # Organizational units
            "roles": 16,           # User roles
            "projects": 32,        # Project identifiers
            "locations": 16,       # Geographic/network locations
        }
        
        for name, dimensions in embeddings.items():
            embedding_config = model_config.token_embedding_name_to_config[name]
            embedding_config.dimensions = dimensions
            # Vocabulary size will be determined during data preprocessing
            embedding_config.vocab_size = 10000  # Placeholder
    
    def _configure_context_architecture(self, model_config):
        """Configure the context tower architecture."""
        arch = model_config.context_architecture.concatenate_then_snn
        
        # SNN layers
        arch.snn.layer_sizes.extend([256, 128])
        
        # User's organizational context
        reduction = arch.segment_reductions.add()
        reduction.token_feature_name = "user_department"
        reduction.token_embedding_name = "departments"
        reduction.segment_weight_scaling = config_pb2.SegmentReduction.WS_IDENTITY
        reduction.segment_weight_normalization = config_pb2.SegmentReduction.WN_L2
        
        # User's role and permissions
        reduction = arch.segment_reductions.add()
        reduction.token_feature_name = "user_roles"
        reduction.token_embedding_name = "roles"
        reduction.segment_weight_scaling = config_pb2.SegmentReduction.WS_LOG
        reduction.segment_weight_normalization = config_pb2.SegmentReduction.WN_L2
        
        # Collaboration network (recent collaborators)
        reduction = arch.segment_reductions.add()
        reduction.token_feature_name = "recent_collaborators"
        reduction.intensity_feature_name = "collaboration_weights"
        reduction.token_embedding_name = "users"
        reduction.segment_weight_scaling = config_pb2.SegmentReduction.WS_IDENTITY
        reduction.segment_weight_normalization = config_pb2.SegmentReduction.WN_L2
        
        # Time-based features (hour of day, day of week)
        dense = arch.fixed_size_dense_features.add()
        dense.feature_name = "temporal_features"
        dense.size = 24 + 7  # Hour embeddings + day embeddings
        
        # Location/network context
        dense = arch.fixed_size_dense_features.add()
        dense.feature_name = "network_context"
        dense.size = 10  # Network segment, VPN status, etc.
    
    def _configure_action_architectures(self, model_config):
        """Configure action tower architectures for each action type."""
        
        for action_type in self.config.action_types:
            arch = model_config.action_name_to_architecture[action_type].concatenate_then_snn
            arch.snn.layer_sizes.extend([128, 64])
            
            if action_type == "documents":
                self._configure_document_action(arch)
            elif action_type == "database":
                self._configure_database_action(arch)
            elif action_type == "api_calls":
                self._configure_api_action(arch)
            elif action_type == "file_access":
                self._configure_file_action(arch)
    
    def _configure_document_action(self, arch):
        """Configure document access action architecture."""
        # Document historical access patterns
        reduction = arch.segment_reductions.add()
        reduction.token_feature_name = "document_previous_accessors"
        reduction.intensity_feature_name = "accessor_weights"
        reduction.token_embedding_name = "users"
        reduction.segment_weight_scaling = config_pb2.SegmentReduction.WS_LOG
        reduction.segment_weight_normalization = config_pb2.SegmentReduction.WN_L2
        
        # Document metadata
        reduction = arch.segment_reductions.add()
        reduction.token_feature_name = "document_tags"
        reduction.token_embedding_name = "projects"
        reduction.segment_weight_scaling = config_pb2.SegmentReduction.WS_IDENTITY
        reduction.segment_weight_normalization = config_pb2.SegmentReduction.WN_L2
        
        # Access type (view, edit, share, download)
        dense = arch.fixed_size_dense_features.add()
        dense.feature_name = "access_type_onehot"
        dense.size = 4
    
    def _configure_database_action(self, arch):
        """Configure database access action architecture."""
        # Query patterns
        reduction = arch.segment_reductions.add()
        reduction.token_feature_name = "table_names"
        reduction.token_embedding_name = "resources"
        reduction.segment_weight_scaling = config_pb2.SegmentReduction.WS_LOG
        reduction.segment_weight_normalization = config_pb2.SegmentReduction.WN_L2
        
        # Query characteristics
        dense = arch.fixed_size_dense_features.add()
        dense.feature_name = "query_features"
        dense.size = 8  # Query complexity, result size, etc.
    
    def _configure_api_action(self, arch):
        """Configure API call action architecture."""
        # API endpoint patterns
        reduction = arch.segment_reductions.add()
        reduction.token_feature_name = "api_endpoints"
        reduction.token_embedding_name = "resources"
        reduction.segment_weight_scaling = config_pb2.SegmentReduction.WS_IDENTITY
        reduction.segment_weight_normalization = config_pb2.SegmentReduction.WN_L2
        
        # Request characteristics
        dense = arch.fixed_size_dense_features.add()
        dense.feature_name = "request_features"
        dense.size = 6  # Method, payload size, response code, etc.
    
    def _configure_file_action(self, arch):
        """Configure file system access action architecture."""
        # File path components
        reduction = arch.segment_reductions.add()
        reduction.token_feature_name = "file_path_tokens"
        reduction.token_embedding_name = "resources"
        reduction.segment_weight_scaling = config_pb2.SegmentReduction.WS_LOG
        reduction.segment_weight_normalization = config_pb2.SegmentReduction.WN_L2
        
        # File access patterns
        dense = arch.fixed_size_dense_features.add()
        dense.feature_name = "file_access_features"
        dense.size = 5  # File size, extension type, permissions, etc.
    
    def _configure_training_hyperparameters(self, model_config):
        """Configure training hyperparameters."""
        training = model_config.training_hyperparameters
        
        # Basic training parameters
        training.batch_size = self.config.batch_size
        training.training_examples = self.config.training_examples
        training.dropout_tokens = 0.33
        training.dropout_neurons = 0.33
        
        # Learning rate schedule
        if self.config.incremental_training:
            lr = self.config.incremental_learning_rate
            rampup_factor = 50.0
            rampdown_factor = 20000.0
            rampup = 0.2
        else:
            lr = self.config.learning_rate
            rampup_factor = 25.0
            rampdown_factor = 10000.0
            rampup = 0.3
        
        one_cycle = training.learning_rate_schedule.one_cycle
        one_cycle.peak_learning_rate = lr
        one_cycle.learning_rate_rampup_factor = rampup_factor
        one_cycle.learning_rate_rampdown_factor = rampdown_factor
        one_cycle.rampup = rampup
        one_cycle.interpolation = config_pb2.LearningRateSchedule.OneCycle.Interpolation.I_COSINE
        
        # Optimizer
        sgd = training.optimizer.sgd
        sgd.global_clipnorm = 100.0
        
        # Loss function
        loss = training.loss_function.generalized_logistic
        loss.soft_margin = 1.0
        loss.hard_margin = 1.0
        loss.negative_push = 1.0
        
        # Synthetic positives strategy
        strategy = training.synthetic_positives_strategy.random_sample_within_minibatch
        strategy.contrastive_scores_per_query = 10
        strategy.positive_instances_weight_factor = 1.0
        
        # Action weights
        total_actions = len(self.config.action_types)
        equal_weight = 1.0 / total_actions
        for action_type in self.config.action_types:
            training.action_name_to_loss_weight[action_type] = equal_weight
        
        # Evaluation configuration
        eval_config = training.evaluation
        eval_config.examples_per_training_epoch = int(self.config.training_examples * 0.1)
        eval_config.batch_size = 50
        eval_config.examples_per_evaluation_epoch = int(self.config.training_examples * 0.05)
        eval_config.metrics_fpr_thresholds.extend([0.001, 0.01, 0.1, 1.0])
        
        eval_strategy = eval_config.synthetic_positives
        eval_strategy.contrastive_scores_per_query = 1
        eval_strategy.positive_instances_weight_factor = 1.0
    
    def save_config(self, filepath: str):
        """Save the model configuration to a textproto file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(text_format.MessageToString(self.model_config))
    
    def build_model(self, vocabulary_filepattern: str) -> model_lib.FacadeModel:
        """Build the Facade model with your organization's configuration."""
        return model_lib.FacadeModel(self.model_config, vocabulary_filepattern)


class CustomTrainingPipeline:
    """Manages the training pipeline for your organization's Facade deployment."""
    
    def __init__(self, config: CustomConfig):
        self.config = config
        self.model_builder = CustomModelBuilder(config)
    
    def setup_directories(self):
        """Create necessary directories for the training pipeline."""
        directories = [
            self.config.data_dir,
            self.config.model_dir,
            self.config.config_dir,
            f"{self.config.data_dir}/train",
            f"{self.config.data_dir}/validation",
            f"{self.config.data_dir}/vocab",
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def create_training_script(self) -> str:
        """Generate training script for your use case."""
        script = f"""#!/bin/bash
# Custom Facade Training Script
# Generated on {datetime.datetime.now().isoformat()}

set -e

# Configuration
DATA_DIR="{self.config.data_dir}"
MODEL_DIR="{self.config.model_dir}"
CONFIG_DIR="{self.config.config_dir}"

# Training parameters
BATCH_SIZE={self.config.batch_size}
LEARNING_RATE={self.config.learning_rate}

echo "Starting custom Facade training pipeline..."

# Step 1: Generate training data
echo "Generating training data..."
python -m batch.dataset_maker_main \\
  --directive=$CONFIG_DIR/directive.textproto \\
  --start_time="$(date -d '1 year ago' '+%Y-%m-%d %H:%M:%S')" \\
  --end_time="$(date -d '3 months ago' '+%Y-%m-%d %H:%M:%S')" \\
  --action_path=$DATA_DIR/actions.tfrecord \\
  --context_path=$DATA_DIR/contexts.tfrecord \\
  --train_output=$DATA_DIR/train/train.tfrecord

# Step 2: Generate validation data
echo "Generating validation data..."
python -m batch.dataset_maker_main \\
  --directive=$CONFIG_DIR/directive.textproto \\
  --start_time="$(date -d '3 months ago' '+%Y-%m-%d %H:%M:%S')" \\
  --end_time="$(date -d '1 week ago' '+%Y-%m-%d %H:%M:%S')" \\
  --action_path=$DATA_DIR/actions.tfrecord \\
  --context_path=$DATA_DIR/contexts.tfrecord \\
  --validation_output=$DATA_DIR/validation/validation.tfrecord

# Step 3: Train the model
echo "Training the model..."
python -m model.train_main \\
  --train_file=$DATA_DIR/train/train.tfrecord \\
  --vocabulary_file=$DATA_DIR/vocab/vocabs.tfrecord \\
  --model_config=$CONFIG_DIR/config.textproto \\
  --model_dir=$MODEL_DIR

# Step 4: Evaluate the model
echo "Evaluating the model..."
python -m model.train_main \\
  --eval_file=$DATA_DIR/validation/validation.tfrecord \\
  --vocabulary_file=$DATA_DIR/vocab/vocabs.tfrecord \\
  --model_config=$CONFIG_DIR/config.textproto \\
  --model_dir=$MODEL_DIR \\
  --is_evaluation_task

echo "Training pipeline completed successfully!"
"""
        return script
    
    def create_incremental_training_script(self) -> str:
        """Generate incremental training script for model updates."""
        script = f"""#!/bin/bash
# Custom Facade Incremental Training Script
# For quarterly model updates

set -e

# Configuration
DATA_DIR="{self.config.data_dir}"
MODEL_DIR="{self.config.model_dir}"
CONFIG_DIR="{self.config.config_dir}"

# Get the latest checkpoint
LATEST_CHECKPOINT=$(ls -t $MODEL_DIR/ckpt-* | head -1)

echo "Starting incremental training from checkpoint: $LATEST_CHECKPOINT"

# Generate new training data (last 3 months)
echo "Generating incremental training data..."
python -m batch.dataset_maker_main \\
  --directive=$CONFIG_DIR/directive.textproto \\
  --start_time="$(date -d '3 months ago' '+%Y-%m-%d %H:%M:%S')" \\
  --end_time="$(date '+%Y-%m-%d %H:%M:%S')" \\
  --action_path=$DATA_DIR/actions_incremental.tfrecord \\
  --context_path=$DATA_DIR/contexts_incremental.tfrecord \\
  --train_output=$DATA_DIR/train/train_incremental.tfrecord

# Continue training from the latest checkpoint
echo "Continuing training..."
python -m model.train_main \\
  --train_file=$DATA_DIR/train/train_incremental.tfrecord \\
  --vocabulary_file=$DATA_DIR/vocab/vocabs.tfrecord \\
  --model_config=$CONFIG_DIR/config_incremental.textproto \\
  --model_dir=$MODEL_DIR \\
  --checkpoint_path=$LATEST_CHECKPOINT

echo "Incremental training completed!"
"""
        return script
    
    def create_inference_script(self) -> str:
        """Generate inference script for production scoring."""
        script = f"""#!/bin/bash
# Custom Facade Inference Script
# For production threat scoring

set -e

# Configuration
DATA_DIR="{self.config.data_dir}"
MODEL_DIR="{self.config.model_dir}"
CONFIG_DIR="{self.config.config_dir}"

# Default time range (last 24 hours)
START_TIME="${{1:-$(date -d '1 day ago' '+%Y-%m-%d %H:%M:%S')}}"
END_TIME="${{2:-$(date '+%Y-%m-%d %H:%M:%S')}}"
OUTPUT_FILE="${{3:-$DATA_DIR/scores_$(date '+%Y%m%d_%H%M%S').tfrecord}}"

echo "Running inference for time range: $START_TIME to $END_TIME"

# Run inference
python -m batch.inference_main \\
  --directive=$CONFIG_DIR/directive.textproto \\
  --start_time="$START_TIME" \\
  --end_time="$END_TIME" \\
  --action_path=$DATA_DIR/actions_live.tfrecord \\
  --context_path=$DATA_DIR/contexts_live.tfrecord \\
  --output_file=$OUTPUT_FILE \\
  --model_config=$CONFIG_DIR/config.textproto \\
  --model_dir=$MODEL_DIR

# Read and display top scores
echo "Top threat scores:"
python -m batch.read_scores_main \\
  --score_file=$OUTPUT_FILE \\
  --top_k=20

echo "Inference completed. Results saved to: $OUTPUT_FILE"
"""
        return script
    
    def save_scripts(self):
        """Save all training and inference scripts."""
        scripts = {
            "train.sh": self.create_training_script(),
            "train_incremental.sh": self.create_incremental_training_script(),
            "inference.sh": self.create_inference_script(),
        }
        
        for filename, content in scripts.items():
            filepath = os.path.join(self.config.config_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
            os.chmod(filepath, 0o755)  # Make executable


class CustomDataProcessor:
    """Processes your organization's raw logs into Facade-compatible format."""
    
    def __init__(self, config: CustomConfig):
        self.config = config
    
    def process_logs_to_tfrecord(self, 
                                input_logs: List[Dict[str, Any]], 
                                output_path: str,
                                log_type: str = "action"):
        """Convert raw logs to TFRecord format."""
        
        with tf.io.TFRecordWriter(output_path) as writer:
            for log_entry in input_logs:
                if log_type == "action":
                    example = self._create_action_example(log_entry)
                elif log_type == "context":
                    example = self._create_context_example(log_entry)
                else:
                    raise ValueError(f"Unknown log_type: {log_type}")
                
                writer.write(example.SerializeToString())
    
    def _create_action_example(self, log_entry: Dict[str, Any]) -> tf.train.Example:
        """Create a TF Example for an action log entry."""
        features = {}
        
        # Required fields
        features["timestamp"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[log_entry["timestamp"]])
        )
        features["user_id"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[log_entry["user_id"].encode()])
        )
        features["action_type"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[log_entry["action_type"].encode()])
        )
        
        # Action-specific features
        if log_entry["action_type"] == "documents":
            features["document_id"] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[log_entry["resource_id"].encode()])
            )
            features["access_type"] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[log_entry.get("access_type", "view").encode()])
            )
        
        elif log_entry["action_type"] == "database":
            features["query_hash"] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[log_entry["query_hash"].encode()])
            )
            features["table_names"] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[t.encode() for t in log_entry.get("tables", [])])
            )
        
        # Add more action types as needed...
        
        return tf.train.Example(features=tf.train.Features(feature=features))
    
    def _create_context_example(self, log_entry: Dict[str, Any]) -> tf.train.Example:
        """Create a TF Example for a context log entry."""
        features = {}
        
        # User context
        features["user_id"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[log_entry["user_id"].encode()])
        )
        features["timestamp"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[log_entry["timestamp"]])
        )
        features["department"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[log_entry.get("department", "unknown").encode()])
        )
        features["role"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[log_entry.get("role", "user").encode()])
        )
        
        # Network context
        features["ip_address"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[log_entry.get("ip_address", "").encode()])
        )
        features["location"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[log_entry.get("location", "unknown").encode()])
        )
        
        return tf.train.Example(features=tf.train.Features(feature=features))


def main():
    """Example usage of the custom Facade implementation."""
    
    # Initialize configuration
    config = CustomConfig(
        embedding_dims=128,
        batch_size=256,
        learning_rate=0.04,
        action_types=["documents", "database", "api_calls", "file_access"],
        training_examples=3000000,
    )
    
    # Set up the training pipeline
    pipeline = CustomTrainingPipeline(config)
    pipeline.setup_directories()
    
    # Save model configuration
    config_path = os.path.join(config.config_dir, "config.textproto")
    pipeline.model_builder.save_config(config_path)
    
    # Create incremental training configuration
    config.incremental_training = True
    config.training_examples = 750000  # Smaller for incremental updates
    incremental_builder = CustomModelBuilder(config)
    incremental_config_path = os.path.join(config.config_dir, "config_incremental.textproto")
    incremental_builder.save_config(incremental_config_path)
    
    # Save training scripts
    pipeline.save_scripts()
    
    print("Custom Facade implementation setup completed!")
    print(f"Configuration saved to: {config_path}")
    print(f"Training scripts saved to: {config.config_dir}")
    print("\nNext steps:")
    print("1. Prepare your log data in TFRecord format")
    print("2. Create directive.textproto for your data sources")
    print("3. Run train.sh to start initial training")
    print("4. Use train_incremental.sh for quarterly updates")
    print("5. Use inference.sh for production scoring")


if __name__ == "__main__":
    main()