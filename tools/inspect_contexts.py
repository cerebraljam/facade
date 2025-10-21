#!/usr/bin/env python3
"""Quick script to inspect Context protobuf content in TFRecord files."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from protos import context_pb2
from common import time_utils

def inspect_context_tfrecord(filename: str, max_examples: int = 5, context_type_filter: str = None):
    """Inspect Context protobuf messages in a TFRecord file."""
    print(f"Inspecting Context TFRecord file: {filename}")
    if context_type_filter:
        print(f"Filtering for context type: {context_type_filter}")
    print("=" * 60)
    
    dataset = tf.data.TFRecordDataset([filename])
    
    # First, let's count the total records and collect type statistics
    print("Scanning file for statistics...")
    type_counts = {}
    total_records = 0
    
    for serialized_record in dataset:
        try:
            context = context_pb2.Context()
            context.ParseFromString(serialized_record.numpy())
            
            context_type = context.type
            type_counts[context_type] = type_counts.get(context_type, 0) + 1
            total_records += 1
            
        except Exception as e:
            print(f"Warning: Failed to parse record {total_records}: {e}")
            total_records += 1
    
    print(f"\nTotal records found: {total_records}")
    print("Context type distribution:")
    for context_type, count in sorted(type_counts.items()):
        print(f"  {context_type}: {count} records")
    
    # Now show examples
    print(f"\n{'='*60}")
    if context_type_filter:
        if context_type_filter not in type_counts:
            print(f"No records found with context type: {context_type_filter}")
            return
        print(f"Showing up to {max_examples} examples of type '{context_type_filter}':")
    else:
        print(f"Showing first {max_examples} records (any type):")
    print("="*60)
    
    # Reset dataset and show examples
    dataset = tf.data.TFRecordDataset([filename])
    examples_shown = 0
    
    for i, serialized_record in enumerate(dataset):
        if examples_shown >= max_examples:
            break
            
        try:
            context = context_pb2.Context()
            context.ParseFromString(serialized_record.numpy())
            
            # Apply filter if specified
            if context_type_filter and context.type != context_type_filter:
                continue
                
            examples_shown += 1
            print(f"\n--- Context {examples_shown} (record #{i+1}) ---")
            
            valid_from = time_utils.convert_proto_time_to_time(context.valid_from)
            
            print(f"Type: {context.type}")
            print(f"Principal: {context.principal}")
            print(f"Valid from: {valid_from}")
            print(f"Peer attributes ({len(context.peer_attributes)} total):")
            
            # Group attributes by name for better readability
            attrs_by_name = {}
            for attr in context.peer_attributes:
                if attr.name not in attrs_by_name:
                    attrs_by_name[attr.name] = []
                attrs_by_name[attr.name].append(attr)
            
            for attr_name, attrs in sorted(attrs_by_name.items()):
                print(f"  {attr_name} ({len(attrs)} peers):")
                for i, attr in enumerate(attrs[:5]):  # Show first 5 peers per attribute
                    direction_map = {
                        0: "D_UNSET",
                        1: "D_FORWARD", 
                        2: "D_BACKWARD"
                    }
                    direction_str = direction_map.get(attr.direction, f"UNKNOWN({attr.direction})")
                    print(f"    {i+1}. {attr.value} (weight: {attr.weight:.3f}, direction: {direction_str})")
                
                if len(attrs) > 5:
                    print(f"    ... and {len(attrs) - 5} more peers")
            
        except Exception as e:
            print(f"Failed to parse Context at record {i+1}: {e}")
            print("Raw bytes length:", len(serialized_record.numpy()))

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage: python inspect_contexts.py [filename] [max_examples] [context_type_filter]")
        print("  filename: path to .tfrecord file (default: sample/context.tfrecord)")
        print("  max_examples: number of examples to show (default: 5)")
        print("  context_type_filter: show only contexts of this type (optional)")
        print("Examples:")
        print("  python inspect_contexts.py")
        print("  python inspect_contexts.py sample/context.tfrecord")
        print("  python inspect_contexts.py sample/context.tfrecord 10")
        print("  python inspect_contexts.py sample/context.tfrecord 10 teams")
        sys.exit(0)
    
    filename = sys.argv[1] if len(sys.argv) > 1 else 'sample/context.tfrecord'
    max_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    context_type_filter = sys.argv[3] if len(sys.argv) > 3 else None
    inspect_context_tfrecord(filename, max_examples, context_type_filter)
