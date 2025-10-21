#!/usr/bin/env python3
"""Quick script to inspect Action protobuf content in TFRecord files."""

import argparse
import os
import sys
from datetime import datetime
from typing import Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

# Add the facade root directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from protos import action_pb2
from common import time_utils

def count_action_records(filename: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> int:
    """Count total records in an Action TFRecord file, optionally filtered by time range."""
    dataset = tf.data.TFRecordDataset([filename])
    count = 0
    for serialized_record in dataset:
        if start_time or end_time:
            try:
                action = action_pb2.Action()
                action.ParseFromString(serialized_record.numpy())
                action_time = time_utils.convert_proto_time_to_time(action.occurred_at)

                if start_time and action_time < start_time:
                    continue
                if end_time and action_time >= end_time:
                    continue
            except Exception:
                continue
        count += 1
    return count

def inspect_action_tfrecord(filename: str, max_examples: int = 5, action_type_filter: Optional[str] = None,
                           start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, count_only: bool = False):
    """Inspect Action protobuf messages in a TFRecord file."""
    if count_only:
        count = count_action_records(filename, start_time, end_time)
        print(count)
        return

    print(f"Inspecting Action TFRecord file: {filename}")
    if action_type_filter:
        print(f"Filtering for action type: {action_type_filter}")
    if start_time:
        print(f"Start time filter: {start_time}")
    if end_time:
        print(f"End time filter: {end_time}")
    print("=" * 60)
    
    dataset = tf.data.TFRecordDataset([filename])
    
    # First, let's count the total records and collect type statistics
    print("Scanning file for statistics...")
    type_counts = {}
    total_records = 0
    filtered_records = 0
    principals = set()
    resources = set()
    
    for serialized_record in dataset:
        try:
            action = action_pb2.Action()
            action.ParseFromString(serialized_record.numpy())
            
            # Apply time filtering
            action_time = time_utils.convert_proto_time_to_time(action.occurred_at)
            if start_time and action_time < start_time:
                total_records += 1
                continue
            if end_time and action_time > end_time:
                total_records += 1
                continue
            
            action_type = action.type
            type_counts[action_type] = type_counts.get(action_type, 0) + 1
            principals.add(action.principal)
            resources.add(action.resource_id)
            total_records += 1
            filtered_records += 1
            
        except Exception as e:
            print(f"Warning: Failed to parse record {total_records}: {e}")
            total_records += 1
    
    print(f"\nTotal records scanned: {total_records}")
    if start_time or end_time:
        print(f"Records matching time filter: {filtered_records}")
    print(f"Unique principals: {len(principals)}")
    print(f"Unique resources: {len(resources)}")
    print("Action type distribution:")
    for action_type, count in sorted(type_counts.items()):
        print(f"  {action_type}: {count} records")
    
    # Now show examples
    print(f"\n{'='*60}")
    if action_type_filter:
        if action_type_filter not in type_counts:
            print(f"No records found with action type: {action_type_filter}")
            return
        print(f"Showing up to {max_examples} examples of type '{action_type_filter}':")
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
            action = action_pb2.Action()
            action.ParseFromString(serialized_record.numpy())
            
            # Apply time filtering
            occurred_at = time_utils.convert_proto_time_to_time(action.occurred_at)
            if start_time and occurred_at < start_time:
                continue
            if end_time and occurred_at > end_time:
                continue
            
            # Apply action type filter if specified
            if action_type_filter and action.type != action_type_filter:
                continue
                
            examples_shown += 1
            print(f"\n--- Action {examples_shown} (record #{i+1}) ---")
            
            print(f"Type: {action.type}")
            print(f"ID (hex): {action.id.hex()}")  # Show hex representation of the hash
            print(f"Resource ID: {action.resource_id}")
            print(f"Principal: {action.principal}")
            print(f"Occurred at: {occurred_at}")
            
            # Display history key with interpretation
            history_key_str = action.history_key.decode('utf-8', 'ignore')
            print(f"History key: {action.history_key} ('{history_key_str}')")
            
        except Exception as e:
            print(f"Failed to parse Action at record {i+1}: {e}")
            print("Raw bytes length:", len(serialized_record.numpy()))

def parse_datetime(date_string: str) -> datetime:
    """Parse datetime string in various formats."""
    from datetime import timezone
    
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%SZ',
    ]
    
    for fmt in formats:
        try:
            parsed_dt = datetime.strptime(date_string, fmt)
            # Make timezone-aware (assume UTC) to match the Action timestamps
            return parsed_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse datetime: {date_string}. Supported formats: {formats}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inspect Action protobuf messages in TFRecord files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inspect_actions.py
  python inspect_actions.py --filename mercari/training_files/action.tfrecord
  python inspect_actions.py --filename mercari/training_files/action.tfrecord --max-examples 10
  python inspect_actions.py --filename mercari/training_files/action.tfrecord --action-type gdrive_doc_access
  python inspect_actions.py --filename mercari/training_files/action.tfrecord --start-time "2024-01-01" --end-time "2024-01-31"
  python inspect_actions.py --filename mercari/training_files/action.tfrecord --start-time "2024-01-01 10:00:00" --end-time "2024-01-01 18:00:00"
        """
    )
    
    parser.add_argument(
        '--filename', '-f',
        default='sample/action.tfrecord',
        help='Path to .tfrecord file (default: sample/action.tfrecord)'
    )
    
    parser.add_argument(
        '--max-examples', '-m',
        type=int,
        default=5,
        help='Number of examples to show (default: 5)'
    )
    
    parser.add_argument(
        '--action-type', '-t',
        help='Show only actions of this type (optional)'
    )
    
    parser.add_argument(
        '--start-time', '-s',
        type=parse_datetime,
        help='Start time filter (e.g., "2024-01-01" or "2024-01-01 10:00:00")'
    )
    
    parser.add_argument(
        '--end-time', '-e',
        type=parse_datetime,
        help='End time filter (e.g., "2024-01-31" or "2024-01-31 23:59:59")'
    )

    parser.add_argument(
        '--count-only', '-c',
        action='store_true',
        help='Only output the total record count (no other information)'
    )

    args = parser.parse_args()

    inspect_action_tfrecord(
        filename=args.filename,
        max_examples=args.max_examples,
        action_type_filter=args.action_type,
        start_time=args.start_time,
        end_time=args.end_time,
        count_only=args.count_only
    )
