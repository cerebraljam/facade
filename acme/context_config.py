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
ACME Custom Context Configuration for Multi-Platform Social Networks

This module defines context configurations for learning social networks from:
- Google Calendar meetings  
- Slack conversations
- GitHub collaborations
- GCP project memberships

The configuration automatically learns peer relationships across all platforms.
"""

from datetime import timedelta
from protos import context_source_config_pb2 as config_pb2


def create_acme_multi_platform_context_config():
    """
    Creates context configuration for ACME's multi-platform environment.
    
    This configuration will automatically learn:
    - Meeting collaborators from Calendar
    - Communication patterns from Slack  
    - Code collaboration from GitHub
    - Project team structures from GCP
    
    Returns:
        List[ContextSourceConfig]: Complete context configuration
    """
    
    # === CALENDAR CONTEXT ===
    calendar_config = config_pb2.ContextSourceConfig()
    calendar_config.type = "calendar"
    calendar_config.context_lookback.seconds = int(timedelta(days=90).total_seconds())
    
    # Calendar peer feature: meeting attendees
    calendar_peer = calendar_config.peer_feature_configs.add()
    calendar_peer.name = "calendar_meeting"
    calendar_peer.max_peers = 50  # Top 50 meeting collaborators
    
    # Use undirected graph - all attendees are peers
    calendar_peer.peer_attribute_graph_type = config_pb2.PeerFeatureConfig.PAGT_BIPARTITE_UNDIRECTED
    calendar_peer.bipartite_graph_config.edge_weighting_method = config_pb2.BipartiteGraph.EWM_DISCOUNTED_LATEST
    calendar_peer.bipartite_graph_config.discount_half_life.seconds = int(timedelta(days=30).total_seconds())
    
    # === SLACK CONTEXT ===  
    slack_config = config_pb2.ContextSourceConfig()
    slack_config.type = "slack"
    slack_config.context_lookback.seconds = int(timedelta(days=60).total_seconds())
    
    # Slack channel engagement (joins/views)
    slack_engagement_peer = slack_config.peer_feature_configs.add()
    slack_engagement_peer.name = "slack_channel"
    slack_engagement_peer.max_peers = 25
    slack_engagement_peer.peer_attribute_graph_type = config_pb2.PeerFeatureConfig.PAGT_BIPARTITE_UNDIRECTED
    slack_engagement_peer.bipartite_graph_config.edge_weighting_method = config_pb2.BipartiteGraph.EWM_DISCOUNTED_LATEST
    slack_engagement_peer.bipartite_graph_config.discount_half_life.seconds = int(timedelta(days=21).total_seconds())
    
    # Slack ongoing memberships (current active state)
    slack_membership_peer = slack_config.peer_feature_configs.add() 
    slack_membership_peer.name = "slack_channel_membership"
    slack_membership_peer.max_peers = 15  # Focus on most active channels
    slack_membership_peer.peer_attribute_graph_type = config_pb2.PeerFeatureConfig.PAGT_BIPARTITE_UNDIRECTED
    slack_membership_peer.bipartite_graph_config.edge_weighting_method = config_pb2.BipartiteGraph.EWM_LATEST
    
    # === GITHUB CONTEXT ===
    github_config = config_pb2.ContextSourceConfig() 
    github_config.type = "github"
    github_config.context_lookback.seconds = int(timedelta(days=180).total_seconds())
    
    # Repository collaborators
    github_repo_peer = github_config.peer_feature_configs.add()
    github_repo_peer.name = "github_repository"
    github_repo_peer.max_peers = 20
    github_repo_peer.peer_attribute_graph_type = config_pb2.PeerFeatureConfig.PAGT_BIPARTITE_UNDIRECTED
    github_repo_peer.bipartite_graph_config.edge_weighting_method = config_pb2.BipartiteGraph.EWM_DISCOUNTED_LATEST
    github_repo_peer.bipartite_graph_config.discount_half_life.seconds = int(timedelta(days=45).total_seconds())
    
    # Pull request reviewer relationships (directed)
    github_pr_peer = github_config.peer_feature_configs.add() 
    github_pr_peer.name = "github_pull_request"
    github_pr_peer.max_peers = 15
    github_pr_peer.peer_attribute_graph_type = config_pb2.PeerFeatureConfig.PAGT_BIPARTITE_DIRECTED
    github_pr_peer.bipartite_graph_config.edge_weighting_method = config_pb2.BipartiteGraph.EWM_LATEST
    
    # === GCP CONTEXT ===
    gcp_config = config_pb2.ContextSourceConfig()
    gcp_config.type = "gcp"  
    gcp_config.context_lookback.seconds = int(timedelta(days=120).total_seconds())
    
    # Project team members
    gcp_project_peer = gcp_config.peer_feature_configs.add()
    gcp_project_peer.name = "gcp_project"
    gcp_project_peer.max_peers = 25
    gcp_project_peer.peer_attribute_graph_type = config_pb2.PeerFeatureConfig.PAGT_BIPARTITE_UNDIRECTED
    gcp_project_peer.bipartite_graph_config.edge_weighting_method = config_pb2.BipartiteGraph.EWM_LATEST
    
    return [calendar_config, slack_config, github_config, gcp_config]


def create_context_proto_examples():
    """
    Example Context protos for each platform to show the data format.
    
    Returns:
        Dict[str, List]: Example context data for each platform
    """
    from protos import context_pb2, timestamp_pb2
    from datetime import datetime
    
    examples = {}
    
    # === CALENDAR EXAMPLE ===
    calendar_context = context_pb2.Context()
    calendar_context.type = "calendar"
    calendar_context.principal = "alice@acme.com"
    calendar_context.valid_from.seconds = int(datetime.now().timestamp())
    
    # Meeting with engineering team
    meeting_attr = calendar_context.peer_attributes.add()
    meeting_attr.name = "calendar_meeting"
    meeting_attr.value = b"meeting_12345"  # Meeting ID
    meeting_attr.weight = 1.0
    meeting_attr.direction = context_pb2.PeerAttribute.D_UNSET  # Undirected
    
    examples["calendar"] = [calendar_context]
    
    # === SLACK EXAMPLE ===
    slack_context = context_pb2.Context()
    slack_context.type = "slack"
    slack_context.principal = "alice@acme.com"
    slack_context.valid_from.seconds = int(datetime.now().timestamp())
    
    # Active in #engineering channel
    channel_attr = slack_context.peer_attributes.add()
    channel_attr.name = "slack_channel"
    channel_attr.value = b"C12345678"  # Slack channel ID
    channel_attr.weight = 5.0  # Number of messages this week
    channel_attr.direction = context_pb2.PeerAttribute.D_UNSET
    
    examples["slack"] = [slack_context]
    
    # === GITHUB EXAMPLE ===
    github_context = context_pb2.Context()
    github_context.type = "github"
    github_context.principal = "alice@acme.com"
    github_context.valid_from.seconds = int(datetime.now().timestamp())
    
    # Repository collaboration
    repo_attr = github_context.peer_attributes.add()
    repo_attr.name = "github_repository"
    repo_attr.value = b"acme/core-platform"  # Repository name
    repo_attr.weight = 1.0
    repo_attr.direction = context_pb2.PeerAttribute.D_UNSET
    
    # Pull request authorship (directed)
    pr_attr = github_context.peer_attributes.add()
    pr_attr.name = "github_pull_request"
    pr_attr.value = b"pr_567"  # Pull request ID
    pr_attr.weight = 1.0
    pr_attr.direction = context_pb2.PeerAttribute.D_FORWARD  # Alice is author
    
    examples["github"] = [github_context]
    
    # === GCP EXAMPLE ===
    gcp_context = context_pb2.Context()
    gcp_context.type = "gcp"
    gcp_context.principal = "alice@acme.com"
    gcp_context.valid_from.seconds = int(datetime.now().timestamp())
    
    # Project membership
    project_attr = gcp_context.peer_attributes.add()
    project_attr.name = "gcp_project"
    project_attr.value = b"acme-prod-12345"  # GCP project ID
    project_attr.weight = 1.0
    project_attr.direction = context_pb2.PeerAttribute.D_UNSET
    
    examples["gcp"] = [gcp_context]
    
    return examples


# === BIGQUERY EXTRACTION TEMPLATES ===

CALENDAR_EXTRACTION_QUERY = """
-- Extract Calendar contexts for Facade
SELECT 
    attendee_email as principal,
    CONCAT('cal_', event_id) as meeting_id,
    UNIX_TIMESTAMP(event_start_time) as valid_from_timestamp,
    UNIX_TIMESTAMP(event_end_time) as attribute_timestamp,
    organizer_email,
    attendee_count
FROM `{project}.{dataset}.calendar_events` 
WHERE event_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
    AND attendee_response_status IN ('accepted', 'tentative')
    AND attendee_count BETWEEN 2 AND 50  -- Focus on collaborative meetings
"""

SLACK_EXTRACTION_QUERY = """
-- Extract Slack channel engagement contexts from join/leave/view events
-- This approach captures intentional collaboration choices better than message counts
SELECT
    user_email as principal,
    channel_id,
    UNIX_TIMESTAMP(event_timestamp) as event_timestamp,
    event_type,  -- 'channel_joined', 'channel_left', 'channel_viewed'
    channel_member_count,
    channel_name
FROM `{project}.{dataset}.slack_channel_events`
WHERE event_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 60 DAY)
    AND event_type IN ('channel_joined', 'channel_viewed')  -- Focus on positive engagement
    AND channel_type = 'public_channel'  -- Exclude DMs for privacy
    AND user_email NOT LIKE '%slackbot%'  -- Exclude system accounts
    AND channel_member_count BETWEEN 3 AND 200  -- Meaningful collaboration groups

UNION ALL

-- Current active memberships for ongoing collaboration context
SELECT 
    user_email as principal,
    channel_id,
    UNIX_TIMESTAMP(CURRENT_TIMESTAMP()) as event_timestamp,
    'active_membership' as event_type,
    channel_member_count,
    channel_name
FROM `{project}.{dataset}.slack_current_memberships`
WHERE is_active_member = true
    AND channel_type = 'public_channel'
    AND days_since_last_activity <= 14  -- Active in last 2 weeks
"""

GITHUB_EXTRACTION_QUERY = """
-- Extract GitHub contexts for Facade
SELECT 
    actor_email as principal,
    repository_name,
    UNIX_TIMESTAMP(event_timestamp) as valid_from_timestamp,
    event_type,
    CASE 
        WHEN event_type = 'pull_request_author' THEN 'FORWARD'
        WHEN event_type = 'pull_request_reviewer' THEN 'BACKWARD'  
        ELSE 'UNSET'
    END as direction,
    pull_request_id
FROM `{project}.{dataset}.github_events`
WHERE event_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
    AND event_type IN ('commit', 'pull_request_author', 'pull_request_reviewer')
"""

GCP_EXTRACTION_QUERY = """
-- Extract GCP contexts for Facade
SELECT
    principal_email as principal, 
    project_id,
    UNIX_TIMESTAMP(binding_timestamp) as valid_from_timestamp,
    role_name
FROM `{project}.{dataset}.gcp_iam_bindings`
WHERE binding_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 120 DAY)
    AND action_type = 'role_granted'
    AND role_name LIKE '%Editor%' OR role_name LIKE '%Owner%'  -- Focus on active roles
"""