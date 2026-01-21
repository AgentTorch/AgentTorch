"""
Token Usage Tracking for LLM Calls
==================================

This module provides comprehensive token usage tracking and cost estimation
for various LLM providers including OpenAI, Anthropic, and others.

Key Features:
- Thread-safe token usage tracking
- Support for multiple LLM providers
- Cost estimation based on current pricing
- Session and global statistics
- Export capabilities (JSON, CSV)
- Real-time monitoring and reporting
"""

import json
import csv
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging
from collections import defaultdict
from enum import Enum
import numpy as np


class LLMProvider(Enum):
    """Enumeration of supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic" 
    GOOGLE = "google"
    
    @classmethod
    def from_string(cls, provider_str: str) -> 'LLMProvider':
        """Convert string to LLMProvider enum."""
        provider_str = provider_str.lower()
        for provider in cls:
            if provider.value == provider_str:
                return provider
        raise ValueError(f"Unknown provider: {provider_str}")
    
    def __str__(self) -> str:
        return self.value


def extract_token_usage(response, provider: Union[LLMProvider, str]) -> Optional[Dict[str, int]]:
    """
    Extract token usage information from LLM response.
    
    Args:
        response: Response object from the LLM call
        provider: Provider name (LLMProvider enum or string)
    
    Returns:
        Dict with prompt_tokens, completion_tokens, and total_tokens
    """
    # Convert string to enum if needed
    if isinstance(provider, str):
        try:
            provider = LLMProvider.from_string(provider)
        except ValueError:
            return None
    
    try:
        if provider == LLMProvider.OPENAI:
            # OpenAI responses typically have usage in the response metadata
            if hasattr(response, '_raw_response'):
                raw_response = response._raw_response
                if hasattr(raw_response, 'usage') and raw_response.usage:
                    usage = raw_response.usage
                    if hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
                        return {
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens
                        }
            
            # Alternative: check if response has usage attribute directly
            if hasattr(response, 'usage') and response.usage is not None:
                usage = response.usage
                if hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
                    return {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens
                    }
        
        elif provider == LLMProvider.ANTHROPIC:
            # Anthropic responses typically have usage in the response metadata
            if hasattr(response, '_raw_response'):
                raw_response = response._raw_response
                if hasattr(raw_response, 'usage') and raw_response.usage:
                    usage = raw_response.usage
                    if hasattr(usage, 'input_tokens') and hasattr(usage, 'output_tokens'):
                        return {
                            "prompt_tokens": usage.input_tokens,
                            "completion_tokens": usage.output_tokens,
                            "total_tokens": usage.input_tokens + usage.output_tokens
                        }
            
            # Alternative: check if response has usage attribute directly
            if hasattr(response, 'usage') and response.usage is not None:
                usage = response.usage
                if hasattr(usage, 'input_tokens') and hasattr(usage, 'output_tokens'):
                    return {
                        "prompt_tokens": usage.input_tokens,
                        "completion_tokens": usage.output_tokens,
                        "total_tokens": usage.input_tokens + usage.output_tokens
                    }
        
        elif provider == LLMProvider.GOOGLE:
            # usage is not present if caching is performed
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                if (hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens') 
                    and usage.prompt_tokens > 0):
                    result = {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": getattr(usage, 'total_tokens', usage.prompt_tokens + usage.completion_tokens)
                    }
                    return result
            elif hasattr(response, '_raw_response'):
                raw_response = response._raw_response
                if hasattr(raw_response, 'usage_metadata') and raw_response.usage_metadata:
                    usage_metadata = raw_response.usage_metadata
                    if hasattr(usage_metadata, 'prompt_token_count') and hasattr(usage_metadata, 'candidates_token_count'):
                        prompt_tokens = usage_metadata.prompt_token_count
                        completion_tokens = usage_metadata.candidates_token_count
                        return {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens
                        }
            logging.getLogger(__name__).warning(f"Failed to extract token usage - none of the strategies worked")
                    
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to extract token usage: {e}")
    
    return None


@dataclass
class TokenUsage:
    """Represents token usage for a single LLM call."""
    timestamp: datetime
    model: str
    provider: Union[LLMProvider, str]  # Accept both for backward compatibility
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float = 0.0
    request_id: Optional[str] = None
    operation: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def cost_per_1k_tokens(self) -> float:
        """Calculate cost per 1K tokens."""
        if self.total_tokens == 0:
            return 0.0
        return (self.cost_usd / self.total_tokens) * 1000


@dataclass
class TokenStats:
    """Aggregated token usage statistics."""
    total_requests: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    average_tokens_per_request: float = 0.0
    average_cost_per_request: float = 0.0
    models_used: Dict[str, int] = field(default_factory=dict)
    providers_used: Dict[str, int] = field(default_factory=dict)
    first_request: Optional[datetime] = None
    last_request: Optional[datetime] = None
    
    # Prompt token statistics
    prompt_tokens_mean: Optional[float] = None
    prompt_tokens_std: Optional[float] = None
    prompt_tokens_min: Optional[int] = None
    prompt_tokens_max: Optional[int] = None
    prompt_tokens_p25: Optional[float] = None
    prompt_tokens_p50: Optional[float] = None
    prompt_tokens_p75: Optional[float] = None
    prompt_tokens_p90: Optional[float] = None
    prompt_tokens_p95: Optional[float] = None
    prompt_tokens_p99: Optional[float] = None
    
    # Completion token statistics
    completion_tokens_mean: Optional[float] = None
    completion_tokens_std: Optional[float] = None
    completion_tokens_min: Optional[int] = None
    completion_tokens_max: Optional[int] = None
    completion_tokens_p25: Optional[float] = None
    completion_tokens_p50: Optional[float] = None
    completion_tokens_p75: Optional[float] = None
    completion_tokens_p90: Optional[float] = None
    completion_tokens_p95: Optional[float] = None
    completion_tokens_p99: Optional[float] = None
    
    def update_averages(self):
        """Update calculated averages."""
        if self.total_requests > 0:
            self.average_tokens_per_request = self.total_tokens / self.total_requests
            self.average_cost_per_request = self.total_cost_usd / self.total_requests
    
    def update_token_statistics(self, usage_data: List['TokenUsage']):
        """Update token statistics from usage data."""
        if not usage_data:
            return
        
        prompt_tokens = [usage.prompt_tokens for usage in usage_data]
        completion_tokens = [usage.completion_tokens for usage in usage_data]
        
        # Calculate prompt token statistics
        self.prompt_tokens_mean = float(np.mean(prompt_tokens))
        self.prompt_tokens_std = float(np.std(prompt_tokens))
        self.prompt_tokens_min = int(np.min(prompt_tokens))
        self.prompt_tokens_max = int(np.max(prompt_tokens))
        self.prompt_tokens_p25 = float(np.percentile(prompt_tokens, 25))
        self.prompt_tokens_p50 = float(np.percentile(prompt_tokens, 50))
        self.prompt_tokens_p75 = float(np.percentile(prompt_tokens, 75))
        self.prompt_tokens_p90 = float(np.percentile(prompt_tokens, 90))
        self.prompt_tokens_p95 = float(np.percentile(prompt_tokens, 95))
        self.prompt_tokens_p99 = float(np.percentile(prompt_tokens, 99))
        
        # Calculate completion token statistics
        self.completion_tokens_mean = float(np.mean(completion_tokens))
        self.completion_tokens_std = float(np.std(completion_tokens))
        self.completion_tokens_min = int(np.min(completion_tokens))
        self.completion_tokens_max = int(np.max(completion_tokens))
        self.completion_tokens_p25 = float(np.percentile(completion_tokens, 25))
        self.completion_tokens_p50 = float(np.percentile(completion_tokens, 50))
        self.completion_tokens_p75 = float(np.percentile(completion_tokens, 75))
        self.completion_tokens_p90 = float(np.percentile(completion_tokens, 90))
        self.completion_tokens_p95 = float(np.percentile(completion_tokens, 95))
        self.completion_tokens_p99 = float(np.percentile(completion_tokens, 99))


class TokenPricing:
    """Current pricing information for LLM providers."""
    
    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "openai": {
            "gpt-5": {"input": 0.00125, "output": 0.01},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-3.5-turbo-instruct": {"input": 0.0015, "output": 0.002},
        },
        "anthropic": {
            "claude-3-haiku-20240307": {"input": 0.0008, "output": 0.004},
            "claude-3-5-haiku-latest": {"input": 0.0008, "output": 0.004},
            "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
            "claude-3-5-sonnet-latest": {"input": 0.003, "output": 0.015},
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        },
        "google": {
            "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},
            "google/gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},  # With provider prefix
            "gemini-pro": {"input": 0.0005, "output": 0.0015},
            "gemini-1.5-pro": {"input": 0.0035, "output": 0.0105},
            "gemini-1.5-flash": {"input": 0.0001, "output": 0.0004},
        }
    }
    
    @classmethod
    def get_model_pricing(cls, provider: str, model: str) -> Dict[str, float]:
        """Get pricing for a specific model."""
        provider_pricing = cls.PRICING.get(provider.lower(), {})
        model_pricing = provider_pricing.get(model, {})
        
        if not model_pricing:
            # Try to match partial model names with improved logic
            model_lower = model.lower()
            
            # For exact matches first
            for available_model, pricing in provider_pricing.items():
                if available_model.lower() == model_lower:
                    return pricing
            
            # For partial matches - prioritize longer matches
            best_match = None
            best_match_length = 0
            
            for available_model, pricing in provider_pricing.items():
                available_lower = available_model.lower()
                
                # Check if they share common patterns
                if cls._models_match(model_lower, available_lower):
                    # Prefer longer common substring matches
                    common_length = cls._get_common_length(model_lower, available_lower)
                    if common_length > best_match_length:
                        best_match = pricing
                        best_match_length = common_length
            
            if best_match:
                return best_match
        
        return model_pricing
    
    @classmethod
    def _models_match(cls, model1: str, model2: str) -> bool:
        """Check if two model names represent the same model family."""
        # Extract base model identifiers
        model1_parts = model1.replace('-', ' ').split()
        model2_parts = model2.replace('-', ' ').split()
        
        # Check if they share common key identifiers
        common_parts = set(model1_parts) & set(model2_parts)
        
        # Models match if they share at least 2 common parts (e.g., "claude", "3")
        # and one of them includes the model family (e.g., "haiku", "sonnet")
        if len(common_parts) >= 2:
            model_families = {'haiku', 'sonnet', 'opus', 'gpt', 'turbo', 'mini', 'flash'}
            for family in model_families:
                if family in model1 and family in model2:
                    return True
            
            # Also match if they're very similar (like gpt-4 variations)
            if len(common_parts) >= 3:
                return True
        
        return False
    
    @classmethod
    def _get_common_length(cls, model1: str, model2: str) -> int:
        """Get the length of common substring between two model names."""
        # Find longest common substring
        max_length = 0
        for i in range(len(model1)):
            for j in range(len(model2)):
                length = 0
                while (i + length < len(model1) and 
                       j + length < len(model2) and 
                       model1[i + length] == model2[j + length]):
                    length += 1
                max_length = max(max_length, length)
        return max_length
    
    @classmethod
    def calculate_cost(cls, provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for a given usage."""
        pricing = cls.get_model_pricing(provider, model)
        if not pricing:
            # Log when pricing is not found for debugging
            logging.getLogger(__name__).debug(
                f"No pricing found for provider='{provider}', model='{model}'. "
                f"Available models for {provider}: {list(cls.PRICING.get(provider.lower(), {}).keys())}"
            )
            return 0.0
        
        input_cost = (prompt_tokens / 1000) * pricing.get("input", 0)
        output_cost = (completion_tokens / 1000) * pricing.get("output", 0)
        total_cost = input_cost + output_cost
        
        # Debug log the calculation
        logging.getLogger(__name__).debug(
            f"Cost calculation: {provider}/{model} - "
            f"Input: {prompt_tokens}*{pricing.get('input', 0)/1000}=${input_cost:.6f}, "
            f"Output: {completion_tokens}*{pricing.get('output', 0)/1000}=${output_cost:.6f}, "
            f"Total: ${total_cost:.6f}"
        )
        
        return total_cost


class TokenTracker:
    """
    Thread-safe token usage tracker with comprehensive logging and reporting.
    
    Usage:
        tracker = TokenTracker("experiment_1")
        tracker.track_usage("openai", "gpt-4", 100, 50, operation="classification")
        stats = tracker.get_session_stats()
    """
    
    _instances: Dict[str, 'TokenTracker'] = {}
    _lock = threading.Lock()
    
    def __init__(self, session_id: str = None, log_dir: str = None, auto_save: bool = True):
        """
        Initialize token tracker.
        
        Args:
            session_id: Unique identifier for this tracking session
            log_dir: Directory to save logs (defaults to './token_logs')
            auto_save: Whether to automatically save usage data
        """
        self.session_id = session_id or f"session_{int(time.time())}"
        self.log_dir = Path(log_dir or "./token_logs")
        self.log_dir.mkdir(exist_ok=True)
        self.auto_save = auto_save
        
        self.usage_history: List[TokenUsage] = []
        # WARNING: Do not access self.session_stats directly! 
        # Token statistics are populated lazily when get_session_stats() is called.
        # Always use get_session_stats() to ensure statistics are calculated.
        self.session_stats = TokenStats()
        self._lock = threading.Lock()
        
        self.logger = logging.getLogger(f"TokenTracker.{self.session_id}")
        self.logger.setLevel(logging.INFO)
        
        # Setup file handler for this session
        self.log_file = self.log_dir / f"tokens_{self.session_id}.log"
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        
        self.logger.info(f"Token tracking session started: {self.session_id}")
    
    @classmethod
    def get_instance(cls, session_id: str = None, **kwargs) -> 'TokenTracker':
        """Get or create a TokenTracker instance (singleton per session)."""
        if session_id is None:
            session_id = "default"
        
        with cls._lock:
            if session_id not in cls._instances:
                cls._instances[session_id] = cls(session_id=session_id, **kwargs)
            return cls._instances[session_id]
    
    def track_usage(
        self, 
        provider: Union[LLMProvider, str], 
        model: str, 
        prompt_tokens: int, 
        completion_tokens: int,
        operation: str = None,
        user_id: str = None,
        request_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> TokenUsage:
        """
        Track token usage for a single LLM call.
        
        Args:
            provider: LLM provider (LLMProvider enum or string like "openai", "anthropic")
            model: Model name (e.g., "gpt-4", "claude-3-5-sonnet-latest")
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            operation: Optional operation description
            user_id: Optional user identifier
            request_id: Optional request identifier
            metadata: Optional additional metadata
        
        Returns:
            TokenUsage object with cost calculation
        """
        # Convert string to enum if needed for consistency
        if isinstance(provider, str):
            try:
                provider_enum = LLMProvider.from_string(provider)
            except ValueError:
                # If conversion fails, keep as string for backward compatibility
                provider_enum = provider
        else:
            provider_enum = provider
            
        total_tokens = prompt_tokens + completion_tokens
        # Use string value for cost calculation (backward compatibility)
        provider_str = provider_enum.value if isinstance(provider_enum, LLMProvider) else provider_enum
        cost = TokenPricing.calculate_cost(provider_str, model, prompt_tokens, completion_tokens)
        
        usage = TokenUsage(
            timestamp=datetime.now(),
            provider=provider_enum,  # Use the processed provider (enum or string)
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            operation=operation,
            user_id=user_id,
            request_id=request_id,
            session_id=self.session_id,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.usage_history.append(usage)
            self._update_stats(usage)
        
        self.logger.info(
            f"Tracked usage: {provider_str}/{model} - "
            f"Tokens: {prompt_tokens}+{completion_tokens}={total_tokens}, "
            f"Cost: ${cost:.6f}, Operation: {operation}"
        )
        
        if self.auto_save:
            self._auto_save()
        
        return usage
    
    def _update_stats(self, usage: TokenUsage):
        """Update session statistics with new usage."""
        stats = self.session_stats
        
        stats.total_requests += 1
        stats.total_tokens += usage.total_tokens
        stats.total_prompt_tokens += usage.prompt_tokens
        stats.total_completion_tokens += usage.completion_tokens
        stats.total_cost_usd += usage.cost_usd
        
        # Update model and provider counts
        stats.models_used[usage.model] = stats.models_used.get(usage.model, 0) + 1
        # Convert provider to string for consistent storage
        provider_key = usage.provider.value if isinstance(usage.provider, LLMProvider) else str(usage.provider)
        stats.providers_used[provider_key] = stats.providers_used.get(provider_key, 0) + 1
        
        # Update timestamps
        if stats.first_request is None or usage.timestamp < stats.first_request:
            stats.first_request = usage.timestamp
        if stats.last_request is None or usage.timestamp > stats.last_request:
            stats.last_request = usage.timestamp
        
        stats.update_averages()
    
    def get_session_stats(self) -> TokenStats:
        """Get current session statistics."""
        with self._lock:
            # Update token statistics from current usage history
            self.session_stats.update_token_statistics(self.usage_history)
            return self.session_stats
    
    def get_usage_by_model(self) -> Dict[str, TokenStats]:
        """Get usage statistics grouped by model."""
        model_stats = defaultdict(lambda: TokenStats())
        model_usage = defaultdict(list)
        
        with self._lock:
            # First pass: group usage by model and calculate basic stats
            for usage in self.usage_history:
                model_usage[usage.model].append(usage)
                
                stats = model_stats[usage.model]
                stats.total_requests += 1
                stats.total_tokens += usage.total_tokens
                stats.total_prompt_tokens += usage.prompt_tokens
                stats.total_completion_tokens += usage.completion_tokens
                stats.total_cost_usd += usage.cost_usd
                
                if stats.first_request is None or usage.timestamp < stats.first_request:
                    stats.first_request = usage.timestamp
                if stats.last_request is None or usage.timestamp > stats.last_request:
                    stats.last_request = usage.timestamp
                
                stats.models_used[usage.model] = stats.models_used.get(usage.model, 0) + 1
                provider_key = usage.provider.value if isinstance(usage.provider, LLMProvider) else str(usage.provider)
                stats.providers_used[provider_key] = stats.providers_used.get(provider_key, 0) + 1
            
            # Second pass: calculate token statistics for each model
            for model, stats in model_stats.items():
                stats.update_averages()
                stats.update_token_statistics(model_usage[model])
        
        return dict(model_stats)
    
    def get_usage_by_operation(self) -> Dict[str, TokenStats]:
        """Get usage statistics grouped by operation."""
        operation_stats = defaultdict(lambda: TokenStats())
        operation_usage = defaultdict(list)
        
        with self._lock:
            # First pass: group usage by operation and calculate basic stats
            for usage in self.usage_history:
                operation = usage.operation or "unknown"
                operation_usage[operation].append(usage)
                
                stats = operation_stats[operation]
                stats.total_requests += 1
                stats.total_tokens += usage.total_tokens
                stats.total_prompt_tokens += usage.prompt_tokens
                stats.total_completion_tokens += usage.completion_tokens
                stats.total_cost_usd += usage.cost_usd
                
                if stats.first_request is None or usage.timestamp < stats.first_request:
                    stats.first_request = usage.timestamp
                if stats.last_request is None or usage.timestamp > stats.last_request:
                    stats.last_request = usage.timestamp
                
                stats.models_used[usage.model] = stats.models_used.get(usage.model, 0) + 1
                provider_key = usage.provider.value if isinstance(usage.provider, LLMProvider) else str(usage.provider)
                stats.providers_used[provider_key] = stats.providers_used.get(provider_key, 0) + 1
            
            # Second pass: calculate token statistics for each operation
            for operation, stats in operation_stats.items():
                stats.update_averages()
                stats.update_token_statistics(operation_usage[operation])
        
        return dict(operation_stats)
    
    def get_recent_usage(self, minutes: int = 60) -> List[TokenUsage]:
        """Get token usage from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        with self._lock:
            return [usage for usage in self.usage_history if usage.timestamp >= cutoff_time]
    
    def export_to_json(self, filename: str = None) -> str:
        """
        Export usage history and statistics to JSON file.
        
        The exported JSON includes:
        - session_stats: Overall session statistics
        - operation_stats: Statistics grouped by operation mode (train, eval, test, etc.)
        - usage_history: Individual usage records with full details
        
        Args:
            filename: Optional filename for export. If not provided, generates timestamp-based name.
            
        Returns:
            str: Path to the exported JSON file
        """
        filename = filename or f"token_usage_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.log_dir / filename
        
        # Get session stats with calculated statistics (this handles its own locking)
        session_stats = self.get_session_stats()
        
        # Get operation-specific statistics
        operation_stats = self.get_usage_by_operation()
        operation_stats_dict = {}
        for operation, stats in operation_stats.items():
            operation_stats_dict[operation] = asdict(stats)
        
        with self._lock:
            data = {
                "session_id": self.session_id,
                "export_timestamp": datetime.now().isoformat(),
                "session_stats": asdict(session_stats),
                "operation_stats": operation_stats_dict,
                "usage_history": []
            }
            
            for usage in self.usage_history:
                usage_dict = asdict(usage)
                usage_dict["timestamp"] = usage.timestamp.isoformat()
                # Convert LLMProvider enum to string for JSON serialization
                if isinstance(usage.provider, LLMProvider):
                    usage_dict["provider"] = usage.provider.value
                data["usage_history"].append(usage_dict)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Exported {len(self.usage_history)} usage records to {filepath}")
        return str(filepath)
    
    def export_to_csv(self, filename: str = None) -> str:
        """Export usage history to CSV file."""
        filename = filename or f"token_usage_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = self.log_dir / filename
        
        with self._lock:
            usage_dicts = []
            for usage in self.usage_history:
                usage_dict = asdict(usage)
                usage_dict["timestamp"] = usage.timestamp.isoformat()
                # Convert LLMProvider enum to string for CSV serialization
                if isinstance(usage.provider, LLMProvider):
                    usage_dict["provider"] = usage.provider.value
                usage_dict["metadata"] = json.dumps(usage_dict["metadata"])
                usage_dicts.append(usage_dict)
        
        if usage_dicts:
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=usage_dicts[0].keys())
                writer.writeheader()
                writer.writerows(usage_dicts)
        
        self.logger.info(f"Exported {len(self.usage_history)} usage records to {filepath}")
        return str(filepath)
    
    def print_summary(self, operation: Optional[str] = None):
        """Print a comprehensive usage summary.
        
        Args:
            operation: Optional operation filter. If specified, only show stats for that operation.
        """
        # Get the appropriate stats based on operation filter
        if operation:
            operation_stats = self.get_usage_by_operation()
            if operation not in operation_stats:
                print(f"\n{'='*60}")
                print(f"TOKEN USAGE SUMMARY ({operation.upper()}) - Session: {self.session_id}")
                print(f"{'='*60}")
                print(f"No operations found for '{operation}'.")
                print(f"{'='*60}\n")
                return
            stats = operation_stats[operation]
            title_suffix = f" ({operation.upper()})"
        else:
            stats = self.get_session_stats()
            title_suffix = ""
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"TOKEN USAGE SUMMARY{title_suffix} - Session: {self.session_id}")
        print(f"{'='*60}")
        
        if stats.first_request and stats.last_request:
            duration = stats.last_request - stats.first_request
            print(f"Duration: {duration}")
        
        print(f"Total Requests: {stats.total_requests:,}")
        print(f"Total Tokens: {stats.total_tokens:,}")
        print(f"  - Prompt Tokens: {stats.total_prompt_tokens:,}")
        print(f"  - Completion Tokens: {stats.total_completion_tokens:,}")
        print(f"Total Cost: ${stats.total_cost_usd:.4f}")
        print(f"Average Tokens per Request: {stats.average_tokens_per_request:.1f}")
        print(f"Average Cost per Request: ${stats.average_cost_per_request:.4f}")
        
        # Print token statistics if available
        if stats.prompt_tokens_mean is not None and stats.completion_tokens_mean is not None:
            print(f"\nPrompt Token Statistics:")
            print(f"  Mean: {stats.prompt_tokens_mean:.1f}, Std: {stats.prompt_tokens_std:.1f}")
            print(f"  Min: {stats.prompt_tokens_min}, Max: {stats.prompt_tokens_max}")
            print(f"  Percentiles - P25: {stats.prompt_tokens_p25:.0f}, P50: {stats.prompt_tokens_p50:.0f}, P75: {stats.prompt_tokens_p75:.0f}")
            print(f"  Percentiles - P90: {stats.prompt_tokens_p90:.0f}, P95: {stats.prompt_tokens_p95:.0f}, P99: {stats.prompt_tokens_p99:.0f}")
            
            print(f"\nCompletion Token Statistics:")
            print(f"  Mean: {stats.completion_tokens_mean:.1f}, Std: {stats.completion_tokens_std:.1f}")
            print(f"  Min: {stats.completion_tokens_min}, Max: {stats.completion_tokens_max}")
            print(f"  Percentiles - P25: {stats.completion_tokens_p25:.0f}, P50: {stats.completion_tokens_p50:.0f}, P75: {stats.completion_tokens_p75:.0f}")
            print(f"  Percentiles - P90: {stats.completion_tokens_p90:.0f}, P95: {stats.completion_tokens_p95:.0f}, P99: {stats.completion_tokens_p99:.0f}")
        
        if stats.models_used:
            print(f"\nModels Used:")
            for model, count in sorted(stats.models_used.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {model}: {count} requests")
        
        if stats.providers_used:
            print(f"\nProviders Used:")
            for provider, count in sorted(stats.providers_used.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {provider}: {count} requests")
        
        # Show usage by operation (only if not filtering by operation)
        if not operation:
            operation_stats = self.get_usage_by_operation()
            if len(operation_stats) > 1:  # Only show if multiple operations
                print(f"\nUsage by Operation:")
                for op, op_stats in sorted(operation_stats.items(), key=lambda x: x[1].total_cost_usd, reverse=True):
                    print(f"  - {op}: {op_stats.total_requests} requests, ${op_stats.total_cost_usd:.4f}")
        
        print(f"{'='*60}\n")
    
    def _auto_save(self):
        """Auto-save usage data if enabled."""
        if len(self.usage_history) % 10 == 0:  # Save every 10 requests
            try:
                self.export_to_json()
            except Exception as e:
                self.logger.error(f"Auto-save failed: {e}")
    
    def reset_session(self):
        """Reset the current session data."""
        with self._lock:
            self.usage_history.clear()
            self.session_stats = TokenStats()
        self.logger.info("Session data reset")


# Global instance for easy access
_default_tracker = None

def get_default_tracker(session_id: str = None, **kwargs) -> TokenTracker:
    """Get the default token tracker instance."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = TokenTracker.get_instance(session_id, **kwargs)
    return _default_tracker


def track_tokens(provider: Union[LLMProvider, str], model: str, prompt_tokens: int, completion_tokens: int, **kwargs) -> TokenUsage:
    """Convenience function to track tokens using the default tracker."""
    tracker = get_default_tracker()
    return tracker.track_usage(provider, model, prompt_tokens, completion_tokens, **kwargs)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing TokenTracker...")
    
    # Create tracker
    tracker = TokenTracker("test_session", auto_save=False)
    
    # Track some usage
    tracker.track_usage("openai", "gpt-4", 100, 50, operation="classification")
    tracker.track_usage("anthropic", "claude-3-5-sonnet-latest", 200, 75, operation="generation")
    tracker.track_usage("openai", "gpt-4o-mini", 150, 25, operation="classification")
    tracker.track_usage("anthropic", "claude-3-5-haiku-latest", 80, 30, operation="summarization")
    
    # Print summary
    tracker.print_summary()
    
    # Export data
    json_file = tracker.export_to_json()
    csv_file = tracker.export_to_csv()
    
    print(f"Data exported to:")
    print(f"  JSON: {json_file}")
    print(f"  CSV: {csv_file}")
