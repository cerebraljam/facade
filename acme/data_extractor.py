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
# Original Facade code by Google Inc.

"""
ACME Data Extractor for Facade Training Data

Efficiently extracts only the necessary fields from BigQuery for Facade training,
handling Google Workspace, Slack, GitHub, and GCP logs optimally.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from google.cloud import bigquery
import tensorflow as tf

from protos import action_pb2, context_pb2, timestamp_pb2
from batch import tfrecord_utils


@dataclass
class ExtractionConfig:
    """Configuration for data extraction"""
    project_id: str
    dataset_id: str
    start_date: datetime
    end_date: datetime
    output_dir: str
    max_records_per_query: int = 1000000  # Prevent runaway queries
    

class ACMEDataExtractor:
    """
    Extracts minimal required fields from BigQuery logs for Facade training.
    
    Optimizes for large-scale log processing by:
    - Extracting only essential fields  
    - Using date partitioning efficiently
    - Batching TFRecord writes
    - Filtering noisy/irrelevant data
    """
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.client = bigquery.Client(project=config.project_id)
        self.logger = logging.getLogger(__name__)
        
    def extract_all_data(self) -> Tuple[str, str]:
        """
        Extract all action and context data for the configured time range.
        
        Returns:
            Tuple of (action_tfrecord_path, context_tfrecord_path)
        """
        action_path = f"{self.config.output_dir}/actions.tfrecord"
        context_path = f"{self.config.output_dir}/contexts.tfrecord"
        
        # Extract actions from all sources
        self.logger.info("Extracting action data...")
        actions = []
        actions.extend(self._extract_drive_actions())
        actions.extend(self._extract_bigquery_actions()) 
        actions.extend(self._extract_http_actions())
        
        self.logger.info(f"Extracted {len(actions)} total actions")
        
        # Extract contexts from all sources  
        self.logger.info("Extracting context data...")
        contexts = []
        contexts.extend(self._extract_calendar_contexts())
        contexts.extend(self._extract_slack_contexts())
        contexts.extend(self._extract_github_contexts())
        contexts.extend(self._extract_gcp_contexts())
        
        self.logger.info(f"Extracted {len(contexts)} total contexts")
        
        # Write to TFRecords
        self._write_actions_tfrecord(actions, action_path)
        self._write_contexts_tfrecord(contexts, context_path)
        
        return action_path, context_path
    
    def _extract_drive_actions(self) -> List[action_pb2.Action]:
        """Extract Google Drive/Docs access actions"""
        
        query = f"""
        SELECT 
            -- Essential fields only
            CONCAT(actor_email, '_', doc_id, '_', UNIX_TIMESTAMP(event_time)) as unique_id,
            UNIX_TIMESTAMP(event_time) as timestamp,
            actor_email as principal,
            doc_id as resource_id,
            doc_title,
            event_type
        FROM `{self.config.project_id}.{self.config.dataset_id}.drive_activity`
        WHERE DATE(event_time) BETWEEN '{self.config.start_date.date()}' 
                                  AND '{self.config.end_date.date()}'
            AND event_type IN ('view', 'edit', 'download')  -- Focus on access events
            AND actor_email NOT LIKE '%service-account%'    -- Exclude service accounts
            AND doc_id IS NOT NULL
        LIMIT {self.config.max_records_per_query}
        """
        
        results = self.client.query(query).result()
        actions = []
        
        for row in results:
            action = action_pb2.Action()
            action.type = "drive"
            action.id = row.unique_id.encode('utf-8')
            action.resource_id = row.resource_id
            action.principal = row.principal
            action.occurred_at.seconds = row.timestamp
            action.history_key = row.resource_id.encode('utf-8')  # Group by document
            
            actions.append(action)
            
        self.logger.info(f"Extracted {len(actions)} Drive actions")
        return actions
    
    def _extract_bigquery_actions(self) -> List[action_pb2.Action]:
        """Extract BigQuery table access actions"""
        
        query = f"""
        SELECT
            CONCAT(principal_email, '_', table_id, '_', UNIX_TIMESTAMP(creation_time)) as unique_id,
            UNIX_TIMESTAMP(creation_time) as timestamp,
            principal_email as principal,
            CONCAT(project_id, '.', dataset_id, '.', table_id) as full_table_id,
            job_type
        FROM `{self.config.project_id}.{self.config.dataset_id}.bigquery_jobs`
        WHERE DATE(creation_time) BETWEEN '{self.config.start_date.date()}'
                                     AND '{self.config.end_date.date()}'
            AND job_type = 'QUERY'
            AND principal_email NOT LIKE '%service-account%'
            AND table_id IS NOT NULL
            AND error_result IS NULL  -- Only successful queries
        LIMIT {self.config.max_records_per_query}
        """
        
        results = self.client.query(query).result()
        actions = []
        
        for row in results:
            action = action_pb2.Action()
            action.type = "bigquery" 
            action.id = row.unique_id.encode('utf-8')
            action.resource_id = row.full_table_id
            action.principal = row.principal
            action.occurred_at.seconds = row.timestamp
            action.history_key = row.full_table_id.encode('utf-8')  # Group by table
            
            actions.append(action)
            
        self.logger.info(f"Extracted {len(actions)} BigQuery actions")
        return actions
    
    def _extract_http_actions(self) -> List[action_pb2.Action]:
        """Extract internal HTTP/API access actions"""
        
        query = f"""
        SELECT
            CONCAT(user_email, '_', host, '_', UNIX_TIMESTAMP(request_time)) as unique_id,
            UNIX_TIMESTAMP(request_time) as timestamp,
            user_email as principal,
            host as hostname,
            http_method,
            response_code
        FROM `{self.config.project_id}.{self.config.dataset_id}.http_requests`  
        WHERE DATE(request_time) BETWEEN '{self.config.start_date.date()}'
                                    AND '{self.config.end_date.date()}'
            AND host LIKE '%.acme.com'  -- Internal hosts only
            AND response_code < 400     -- Successful requests only
            AND user_email IS NOT NULL
            AND user_email NOT LIKE '%service-account%'
        LIMIT {self.config.max_records_per_query}
        """
        
        results = self.client.query(query).result()
        actions = []
        
        for row in results:
            action = action_pb2.Action()
            action.type = "http"
            action.id = row.unique_id.encode('utf-8') 
            action.resource_id = row.hostname
            action.principal = row.principal
            action.occurred_at.seconds = row.timestamp
            action.history_key = row.hostname.encode('utf-8')  # Group by hostname
            
            actions.append(action)
            
        self.logger.info(f"Extracted {len(actions)} HTTP actions")
        return actions
    
    def _extract_calendar_contexts(self) -> List[context_pb2.Context]:
        """Extract calendar meeting contexts for peer relationships"""
        
        query = f"""
        SELECT
            attendee_email as principal,
            event_id as meeting_id,
            UNIX_TIMESTAMP(event_start_time) as start_timestamp,
            UNIX_TIMESTAMP(event_end_time) as end_timestamp,
            attendee_count
        FROM `{self.config.project_id}.{self.config.dataset_id}.calendar_events`
        WHERE DATE(event_start_time) BETWEEN '{self.config.start_date.date()}'
                                        AND '{self.config.end_date.date()}'
            AND attendee_response_status = 'accepted'
            AND attendee_count BETWEEN 2 AND 50  -- Collaborative meetings only
            AND attendee_email NOT LIKE '%resource-%'  -- Exclude room resources
        """
        
        results = self.client.query(query).result()
        contexts = []
        
        for row in results:
            context = context_pb2.Context()
            context.type = "calendar"
            context.principal = row.principal
            context.valid_from.seconds = row.start_timestamp
            
            # Create peer attribute for meeting
            attr = context.peer_attributes.add()
            attr.name = "calendar_meeting"
            attr.value = row.meeting_id.encode('utf-8')
            attr.weight = 1.0 / row.attendee_count  # Weight by meeting size
            attr.direction = context_pb2.PeerAttribute.D_UNSET  # Undirected
            attr.time.seconds = row.end_timestamp  # Meeting end time
            
            contexts.append(context)
            
        self.logger.info(f"Extracted {len(contexts)} calendar contexts")
        return contexts
    
    def _extract_slack_contexts(self) -> List[context_pb2.Context]:
        """
        Extract Slack channel membership contexts from join/leave/view events.
        
        This approach is superior to message-based tracking because:
        1. Join events represent intentional collaboration choices
        2. Channel membership is more stable than daily message counts  
        3. Avoids privacy concerns with message content analysis
        4. View events show interest even without active participation
        """
        
        query = f"""
        SELECT
            user_email as principal,
            channel_id,
            UNIX_TIMESTAMP(event_timestamp) as timestamp,
            event_type,  -- 'channel_joined', 'channel_left', 'channel_viewed'
            channel_name,
            channel_member_count
        FROM `{self.config.project_id}.{self.config.dataset_id}.slack_channel_events`
        WHERE DATE(event_timestamp) BETWEEN '{self.config.start_date.date()}'
                                       AND '{self.config.end_date.date()}'
            AND event_type IN ('channel_joined', 'channel_viewed', 'channel_left')
            AND user_email IS NOT NULL
            AND user_email NOT LIKE '%slackbot%'  -- Exclude system accounts
            AND channel_type = 'public_channel'   -- Focus on public collaboration
            AND channel_member_count BETWEEN 3 AND 500  -- Meaningful team sizes
        ORDER BY user_email, channel_id, event_timestamp
        """
        
        results = self.client.query(query).result()
        contexts = []
        
        # Track active memberships to compute proper weights
        user_channel_state = {}  # (user, channel) -> latest_state
        
        for row in results:
            key = (row.principal, row.channel_id)
            
            # Update membership state tracking
            if row.event_type == 'channel_joined':
                user_channel_state[key] = {
                    'status': 'member',
                    'joined_at': row.timestamp,
                    'last_activity': row.timestamp,
                    'view_count': user_channel_state.get(key, {}).get('view_count', 0)
                }
            elif row.event_type == 'channel_viewed':
                if key not in user_channel_state:
                    user_channel_state[key] = {'status': 'viewer', 'view_count': 0}
                user_channel_state[key]['view_count'] += 1
                user_channel_state[key]['last_activity'] = row.timestamp
            elif row.event_type == 'channel_left':
                if key in user_channel_state:
                    user_channel_state[key]['status'] = 'left'
                    user_channel_state[key]['left_at'] = row.timestamp
            
            # Create context for each meaningful event
            if row.event_type in ['channel_joined', 'channel_viewed']:
                context = context_pb2.Context()
                context.type = "slack"
                context.principal = row.principal
                context.valid_from.seconds = row.timestamp
                
                # Calculate weight based on engagement type and channel size
                base_weight = 1.0
                if row.event_type == 'channel_joined':
                    base_weight = 3.0  # Joining shows stronger intent than viewing
                elif row.event_type == 'channel_viewed':
                    base_weight = 1.0  # Viewing shows interest
                
                # Adjust weight by channel size (smaller = stronger signal)
                size_factor = max(0.1, min(1.0, 10.0 / row.channel_member_count))
                final_weight = base_weight * size_factor
                
                # Create peer attribute for channel collaboration
                attr = context.peer_attributes.add()
                attr.name = "slack_channel"
                attr.value = row.channel_id.encode('utf-8')
                attr.weight = final_weight
                attr.direction = context_pb2.PeerAttribute.D_UNSET  # Undirected collaboration
                attr.time.seconds = row.timestamp
                
                contexts.append(context)
        
        # Also create contexts for current active memberships (snapshot approach)
        # This captures ongoing collaboration relationships
        active_memberships_query = f"""
        SELECT DISTINCT
            user_email as principal,
            channel_id,
            channel_name,
            UNIX_TIMESTAMP(CURRENT_TIMESTAMP()) as current_timestamp,
            channel_member_count,
            days_since_last_message
        FROM `{self.config.project_id}.{self.config.dataset_id}.slack_current_memberships`
        WHERE is_active_member = true
            AND channel_type = 'public_channel'
            AND user_email IS NOT NULL
            AND days_since_last_message <= 30  -- Active in last 30 days
        """
        
        try:
            membership_results = self.client.query(active_memberships_query).result()
            
            for row in membership_results:
                context = context_pb2.Context()
                context.type = "slack"  
                context.principal = row.principal
                context.valid_from.seconds = row.current_timestamp
                
                # Weight active membership by recency and channel size
                recency_factor = max(0.1, (30 - row.days_since_last_message) / 30.0)
                size_factor = max(0.1, min(1.0, 15.0 / row.channel_member_count))
                membership_weight = 2.0 * recency_factor * size_factor
                
                attr = context.peer_attributes.add()
                attr.name = "slack_channel_membership"  # Different attribute for ongoing membership
                attr.value = row.channel_id.encode('utf-8')
                attr.weight = membership_weight
                attr.direction = context_pb2.PeerAttribute.D_UNSET
                attr.time.seconds = row.current_timestamp
                
                contexts.append(context)
                
        except Exception as e:
            self.logger.warning(f"Could not extract current memberships: {e}")
            
        self.logger.info(f"Extracted {len(contexts)} Slack contexts (events + memberships)")
        return contexts
    
    def _extract_github_contexts(self) -> List[context_pb2.Context]:
        """Extract GitHub collaboration contexts"""
        
        query = f"""
        SELECT 
            actor_email as principal,
            repository_name,
            UNIX_TIMESTAMP(event_timestamp) as timestamp,
            event_type,
            pull_request_id
        FROM `{self.config.project_id}.{self.config.dataset_id}.github_events`
        WHERE DATE(event_timestamp) BETWEEN '{self.config.start_date.date()}'
                                       AND '{self.config.end_date.date()}'
            AND event_type IN ('commit', 'pull_request_created', 'pull_request_reviewed')
            AND actor_email IS NOT NULL
            AND repository_name LIKE 'acme/%'  -- ACME repositories only
        """
        
        results = self.client.query(query).result()
        contexts = []
        
        for row in results:
            context = context_pb2.Context()
            context.type = "github"
            context.principal = row.principal
            context.valid_from.seconds = row.timestamp
            
            # Repository collaboration
            repo_attr = context.peer_attributes.add()
            repo_attr.name = "github_repository"
            repo_attr.value = row.repository_name.encode('utf-8')
            repo_attr.weight = 1.0
            repo_attr.direction = context_pb2.PeerAttribute.D_UNSET
            repo_attr.time.seconds = row.timestamp
            
            # PR-specific relationships (directed)
            if row.pull_request_id:
                pr_attr = context.peer_attributes.add()
                pr_attr.name = "github_pull_request"
                pr_attr.value = str(row.pull_request_id).encode('utf-8')
                pr_attr.weight = 1.0
                pr_attr.time.seconds = row.timestamp
                
                # Set direction based on event type
                if row.event_type == 'pull_request_created':
                    pr_attr.direction = context_pb2.PeerAttribute.D_FORWARD  # Author
                elif row.event_type == 'pull_request_reviewed':
                    pr_attr.direction = context_pb2.PeerAttribute.D_BACKWARD  # Reviewer
                else:
                    pr_attr.direction = context_pb2.PeerAttribute.D_UNSET