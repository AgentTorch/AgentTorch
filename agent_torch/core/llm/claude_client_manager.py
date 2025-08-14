"""
Singleton Claude Client Manager
===============================

Manages a single Claude client instance across the entire AgentTorch system
to eliminate redundant API client creation and improve performance.

Features:
- Singleton pattern for single client instance
- Thread-safe initialization
- Automatic API key validation
- Usage tracking and statistics
"""

import os
import threading
from typing import Optional, Dict, Any
from anthropic import Anthropic
from dotenv import load_dotenv


class ClaudeClientManager:
    """
    Singleton manager for Claude API client instances.
    
    Ensures only one Claude client is created and reused across
    all components of the AgentTorch system.
    """
    
    _instance: Optional['ClaudeClientManager'] = None
    _lock = threading.Lock()
    _client: Optional[Anthropic] = None
    _initialized = False
    
    def __new__(cls) -> 'ClaudeClientManager':
        """Create singleton instance with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the client manager (only once)."""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._setup_client()
                    self._usage_stats = {
                        'clients_requested': 0,
                        'unique_requesters': set(),
                        'api_calls_made': 0
                    }
                    self._initialized = True
    
    def _setup_client(self):
        """Set up the Anthropic client with proper API key validation."""
        # Load environment variables
        env_paths = [
            "agent_torch/core/llm/.env",
            ".env",
            os.path.expanduser("~/.env")
        ]
        
        for env_path in env_paths:
            if os.path.exists(env_path):
                load_dotenv(env_path)
                print(f"Claude client manager: Loading .env from: {env_path}")
                break
        
        # Get API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required for Claude client manager"
            )
        
        # Create single client instance
        self._client = Anthropic(api_key=api_key)
        print("Claude client manager: Singleton client initialized")
    
    @classmethod
    def get_client(cls, requester_name: str = "unknown") -> Anthropic:
        """
        Get the singleton Claude client instance.
        
        Args:
            requester_name: Name of the requesting component for tracking
            
        Returns:
            Anthropic client instance
        """
        instance = cls()
        
        # Update usage statistics
        instance._usage_stats['clients_requested'] += 1
        instance._usage_stats['unique_requesters'].add(requester_name)
        
        print(f"🔄 Claude client requested by: {requester_name}")
        
        return instance._client
    
    @classmethod
    def track_api_call(cls):
        """Track an API call for statistics."""
        instance = cls()
        instance._usage_stats['api_calls_made'] += 1
    
    @classmethod
    def get_usage_stats(cls) -> Dict[str, Any]:
        """Get usage statistics for monitoring."""
        instance = cls()
        return {
            'clients_requested': instance._usage_stats['clients_requested'],
            'unique_requesters': len(instance._usage_stats['unique_requesters']),
            'requester_names': list(instance._usage_stats['unique_requesters']),
            'api_calls_made': instance._usage_stats['api_calls_made'],
            'client_id': id(instance._client)
        }
    
    @classmethod
    def print_stats(cls):
        """Print usage statistics for debugging."""
        stats = cls.get_usage_stats()
        print("\n📊 Claude Client Manager Statistics:")
        print(f"   Client requests: {stats['clients_requested']}")
        print(f"   Unique requesters: {stats['unique_requesters']}")  
        print(f"   Requester components: {stats['requester_names']}")
        print(f"   API calls tracked: {stats['api_calls_made']}")
        print(f"   Client instance ID: {stats['client_id']}")
        print(f"   ✅ Single client reused across all components\n")


# Convenience function for backward compatibility
def get_claude_client(requester_name: str = "legacy") -> Anthropic:
    """Get Claude client instance - convenience function."""
    return ClaudeClientManager.get_client(requester_name) 