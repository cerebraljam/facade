#!/usr/bin/env python3
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

"""
Simple script to parse Google Workspace logs and create TFRecord files for Facade.
"""

from acme.gworkspace_parser import GoogleWorkspaceParser, create_workspace_parser_config
import logging

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Configure parser for your company size
    config = create_workspace_parser_config(
        employee_count=2000,           # Your total employee count
        shared_threshold_percentage=3.0,  # 3% threshold like Google
        min_shared_threshold=10,       # Minimum threshold for small teams
        max_shared_threshold=500       # Maximum reasonable threshold
    )
    
    print(f"Configuration:")
    print(f"  Employee count: {config.total_employee_count}")
    print(f"  Threshold percentage: {config.shared_threshold_percentage}%")
    print(f"  Computed threshold: {config.computed_shared_threshold} users")
    print(f"  Documents shared with >{config.computed_shared_threshold} users will be excluded")
    print()
    
    # Create parser
    parser = GoogleWorkspaceParser(config)
    
    # Define file paths
    input_jsonl = "workspace_audit_logs.jsonl"    # Your BigQuery export
    output_tfrecord = "workspace_actions.tfrecord"  # Output for Facade
    
    # Parse and convert
    print(f"Processing {input_jsonl}...")
    parser.parse_and_write_tfrecord(input_jsonl, output_tfrecord)
    print(f"Created {output_tfrecord}")

if __name__ == "__main__":
    main()