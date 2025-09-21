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
# Based on the original Facade project: Copyright 2025 Google Inc.

"""
Configuration for parsing Google Workspace logs to TFRecord format.
Minimal implementation focused only on log parsing and TFRecord generation.
"""

import datetime
from .gworkspace_parser import create_workspace_parser_config


class GWSParserConfig:
    """Minimal configuration for Google Workspace log parser."""
    
    def __init__(self):
        # Company size for threshold calculation
        self.total_employees = 2000
        
        # Sharing threshold configuration
        # Documents shared with more than this percentage of employees are excluded
        self.shared_threshold_percentage = 3.0  # 3% like Google's approach
        
        # Computed threshold bounds
        self.min_shared_threshold = 15   # Minimum absolute threshold
        self.max_shared_threshold = 500  # Maximum absolute threshold
        
    def get_computed_threshold(self) -> int:
        """Calculate the sharing threshold based on company size."""
        percentage_threshold = int(self.total_employees * self.shared_threshold_percentage / 100)
        return max(
            self.min_shared_threshold,
            min(percentage_threshold, self.max_shared_threshold)
        )
    
    def create_workspace_parser(self):
        """Create configured Google Workspace parser."""
        config = create_workspace_parser_config(
            employee_count=self.total_employees,
            shared_threshold_percentage=self.shared_threshold_percentage,
            min_shared_threshold=self.min_shared_threshold,
            max_shared_threshold=self.max_shared_threshold
        )
        return config
    
    def print_threshold_info(self):
        """Print threshold configuration for verification."""
        threshold = self.get_computed_threshold()
        print(f"Company size: {self.total_employees} employees")
        print(f"Threshold percentage: {self.shared_threshold_percentage}%")
        print(f"Computed sharing threshold: {threshold} users")
        print(f"Documents shared with >{threshold} users will be excluded")


def create_acme_parser_config() -> GWSParserConfig:
    """Create ACME parser configuration."""
    return GWSParserConfig()