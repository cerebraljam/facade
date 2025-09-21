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
# Original Facade project: Copyright 2025 Google Inc.

"""
Google Workspace log parser for Facade.
Parses JSONL logs from Google Workspace (exported from BigQuery) and creates TFRecord files.
"""

import json
import datetime
import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
import tensorflow as tf

# Import Facade components
from common import time_utils
from protos import action_pb2


@dataclass
class WorkspaceParserConfig:
    """Configuration for Google Workspace parser."""
    
    # Sharing threshold configuration
    total_employee_count: int = 2000
    shared_threshold_percentage: float = 3.0
    min_shared_threshold: int = 10
    max_shared_threshold: int = 500
    
    # Time parsing configuration  
    timezone: datetime.tzinfo = datetime.timezone.utc
    time_formats: List[str] = None
    
    def __post_init__(self):
        """Initialize time formats and compute threshold."""
        if self.time_formats is None:
            self.time_formats = [
                '%Y-%m-%dT%H:%M:%S.%fZ',  # ISO8601 with microseconds
                '%Y-%m-%dT%H:%M:%SZ',     # ISO8601 without microseconds  
                '%Y-%m-%dT%H:%M:%S.%f%z', # ISO8601 with timezone
                '%Y-%m-%dT%H:%M:%S%z',    # ISO8601 with timezone, no microseconds
                '%Y-%m-%d %H:%M:%S',      # Simple format
            ]
        
        # Calculate sharing threshold
        percentage_threshold = int(self.total_employee_count * self.shared_threshold_percentage / 100)
        self.computed_shared_threshold = max(
            self.min_shared_threshold,
            min(percentage_threshold, self.max_shared_threshold)
        )


class GoogleWorkspaceParser:
    """Parser for Google Workspace audit logs in JSONL format."""
    
    def __init__(self, config: WorkspaceParserConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def parse_time(self, time_str: str) -> Optional[datetime.datetime]:
        """Parse time string to UTC datetime following Facade conventions."""
        if not time_str:
            return None
            
        time_str = time_str.strip()
        
        # Try each format
        for fmt in self.config.time_formats:
            try:
                dt = datetime.datetime.strptime(time_str, fmt)
                
                # Facade assumes UTC - if no timezone, assign UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=self.config.timezone)
                    
                # Convert to UTC if not already
                if dt.tzinfo != datetime.timezone.utc:
                    dt = dt.astimezone(datetime.timezone.utc)
                    
                return dt
                
            except ValueError:
                continue
                
        self.logger.warning(f"Failed to parse time: {time_str}")
        return None
    
    def extract_resource_id(self, log_entry: Dict[str, Any]) -> Optional[str]:
        """Extract stable resource identifier from log entry."""
        possible_paths = [
            ['id', 'doc_id'],
            ['parameters', 'doc_id'],
            ['parameters', 'document_id'], 
            ['events', 'parameters', 'doc_id'],
            ['resource', 'resourceName'],
            ['protoPayload', 'resourceName'],
        ]
        
        for path in possible_paths:
            value = self._get_nested_value(log_entry, path)
            if value:
                return self._normalize_resource_id(value)
        return None
    
    def extract_principal(self, log_entry: Dict[str, Any]) -> Optional[str]:
        """Extract user principal from log entry."""
        possible_paths = [
            ['actor', 'email'],
            ['actor', 'user', 'email'],
            ['protoPayload', 'authenticationInfo', 'principalEmail'],
            ['user', 'email'],
            ['events', 'actor', 'email'],
        ]
        
        for path in possible_paths:
            value = self._get_nested_value(log_entry, path)
            if value and '@' in value:
                return value.lower()
        return None
    
    def extract_time(self, log_entry: Dict[str, Any]) -> Optional[str]:
        """Extract timestamp from log entry."""
        possible_paths = [
            ['time'],
            ['timestamp'], 
            ['events', 'time'],
            ['receiveTimestamp'],
        ]
        
        for path in possible_paths:
            value = self._get_nested_value(log_entry, path)
            if value:
                return value
        return None
    
    def count_document_sharing(self, jsonl_file_path: str) -> Dict[str, int]:
        """Count how many users each document is shared with."""
        sharing_counts = {}
        
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    log_entry = json.loads(line.strip())
                    resource_id = self.extract_resource_id(log_entry)
                    principal = self.extract_principal(log_entry)
                    
                    if resource_id and principal:
                        if resource_id not in sharing_counts:
                            sharing_counts[resource_id] = set()
                        sharing_counts[resource_id].add(principal)
                        
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
        
        return {doc_id: len(users) for doc_id, users in sharing_counts.items()}
    
    def parse_to_actions(self, jsonl_file_path: str) -> List[action_pb2.Action]:
        """Parse JSONL file to Facade Action protos."""
        
        # First pass: analyze document sharing
        self.logger.info("Analyzing document sharing patterns...")
        sharing_counts = self.count_document_sharing(jsonl_file_path)
        
        # Log filtering info
        filtered_docs = sum(1 for count in sharing_counts.values() 
                           if count > self.config.computed_shared_threshold)
        total_docs = len(sharing_counts)
        
        self.logger.info(
            f"Will exclude {filtered_docs}/{total_docs} documents "
            f"shared with >{self.config.computed_shared_threshold} users"
        )
        
        # Second pass: create Action protos
        actions = []
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    log_entry = json.loads(line.strip())
                    action = self._create_action(log_entry, line_num, sharing_counts)
                    if action:
                        actions.append(action)
                except:
                    continue
        
        self.logger.info(f"Created {len(actions)} actions")
        return actions
    
    def write_tfrecord(self, actions: List[action_pb2.Action], output_path: str):
        """Write actions to TFRecord file."""
        with tf.io.TFRecordWriter(output_path) as writer:
            for action in actions:
                writer.write(action.SerializeToString())
        
        self.logger.info(f"Wrote {len(actions)} actions to {output_path}")
    
    def parse_and_write_tfrecord(self, jsonl_file_path: str, tfrecord_output_path: str):
        """Complete pipeline: parse JSONL and write TFRecord."""
        actions = self.parse_to_actions(jsonl_file_path)
        self.write_tfrecord(actions, tfrecord_output_path)
    
    def _create_action(self, log_entry: Dict[str, Any], line_num: int, 
                      sharing_counts: Dict[str, int]) -> Optional[action_pb2.Action]:
        """Convert log entry to Action proto."""
        
        resource_id = self.extract_resource_id(log_entry)
        principal = self.extract_principal(log_entry)
        time_str = self.extract_time(log_entry)
        
        if not resource_id or not principal:
            return None
            
        # Parse time using Facade's UTC convention
        occurred_at = self.parse_time(time_str)
        if not occurred_at:
            return None
        
        # Apply sharing threshold filter
        if (resource_id in sharing_counts and
            sharing_counts[resource_id] > self.config.computed_shared_threshold):
            return None
        
        # Create Action proto
        action = action_pb2.Action()
        action.id = f"{line_num}_{resource_id}_{principal}".encode('utf-8')
        action.resource_id = resource_id.encode('utf-8')
        action.principal = principal.encode('utf-8')
        action.type = "doc_access"  # Standard type for Google Workspace documents
        action.history_key = resource_id  # Group by resource for history features
        
        # Convert time to Facade timestamp proto
        action.occurred_at.CopyFrom(time_utils.convert_to_proto_time(occurred_at))
        
        return action
    
    def _get_nested_value(self, data: Dict[str, Any], path: List[str]) -> Optional[str]:
        """Get value from nested dictionary."""
        current = data
        try:
            for key in path:
                current = current[key]
            return str(current) if current else None
        except (KeyError, TypeError):
            return None
    
    def _normalize_resource_id(self, resource_id: str) -> str:
        """Normalize resource ID to ensure stability."""
        # Remove Google Docs URL prefixes
        prefixes = [
            'https://docs.google.com/document/d/',
            'https://docs.google.com/spreadsheets/d/',
            'https://docs.google.com/presentation/d/',
            'https://drive.google.com/file/d/',
        ]
        
        normalized = resource_id
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        
        # Extract document ID (before any path components)
        if '/' in normalized:
            normalized = normalized.split('/')[0]
            
        return normalized


def create_workspace_parser_config(
    employee_count: int = 2000,
    shared_threshold_percentage: float = 3.0,
    **kwargs
) -> WorkspaceParserConfig:
    """Create parser configuration."""
    return WorkspaceParserConfig(
        total_employee_count=employee_count,
        shared_threshold_percentage=shared_threshold_percentage,
        **kwargs
    )


def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python gworkspace_parser.py <input_jsonl> <output_tfrecord>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Configure for 2000 employee company
    config = create_workspace_parser_config(
        employee_count=2000,
        shared_threshold_percentage=3.0
    )
    
    parser = GoogleWorkspaceParser(config)
    
    print(f"Parsing {input_file} to {output_file}")
    print(f"Using sharing threshold: {config.computed_shared_threshold} users")
    
    parser.parse_and_write_tfrecord(input_file, output_file)
    print("Done.")


if __name__ == "__main__":
    main()