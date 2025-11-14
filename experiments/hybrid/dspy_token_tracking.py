"""
DSPy Token Tracking
===================

This module provides a wrapper around DSPy LM calls to enable comprehensive
token usage tracking during optimization processes like MIPRO.

Key Features:
- Transparent token tracking for all DSPy LM calls
- Integration with existing TokenTracker infrastructure  
- Support for Gemini, OpenAI, and Anthropic models
- Automatic cost calculation and reporting
- Thread-safe operation for parallel processing
"""

import dspy
from dspy.clients.lm import BaseLM
import logging
from typing import Dict, Any, Optional, List, Union
from functools import wraps
import time
import threading

# Import our existing token tracking infrastructure
from token_tracker import TokenTracker, extract_token_usage, LLMProvider

logger = logging.getLogger(__name__)

class TokenTrackingLM(BaseLM):
    """
    A wrapper around DSPy LM that tracks token usage for all calls.
    
    This class acts as a drop-in replacement for dspy.LM while transparently
    tracking all token usage through our existing TokenTracker infrastructure.
    
    Inherits from BaseLM to ensure proper DSPy recognition and compatibility.
    """
    
    def __init__(self, model: str, session_id: str = None, tracker: TokenTracker = None, **kwargs):
        """
        Initialize the token tracking LM wrapper.
        
        Args:
            model: Model identifier (e.g., 'gemini/gemini-2.5-flash')
            session_id: Session ID for token tracking (used if tracker not provided)
            tracker: Pre-created TokenTracker instance (optional)
            **kwargs: Additional arguments passed to the underlying LM
        """
        # Initialize the parent BaseLM class with required parameters
        super().__init__(model=model, **kwargs)
        
        # Create the underlying DSPy LM for actual calls
        self.lm = dspy.LM(model, **kwargs)
        
        # Use provided tracker or create/get one
        if tracker is not None:
            self.token_tracker = tracker
            self.session_id = tracker.session_id
        else:
            self.session_id = session_id or f"dspy_session_{int(time.time())}"
            self.token_tracker = TokenTracker.get_instance(
                session_id=self.session_id,
                log_dir=f"./expt2_models/token_logs"
            )
        
        # Determine provider from model name
        self.provider = self._determine_provider(model)
        
        # Track statistics
        self.call_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self._lock = threading.Lock()
        
        # Operation context for tracking different phases
        self._current_operation = "dspy_optimization"  # Default operation
        
        logger.info(f"Initialized TokenTrackingLM for {model} with session {self.session_id}")
    
    def _determine_provider(self, model: str) -> LLMProvider:
        """Determine the LLM provider from the model name."""
        model_lower = model.lower()
        
        if 'gemini' in model_lower or 'google' in model_lower:
            return LLMProvider.GOOGLE
        elif 'gpt' in model_lower or 'openai' in model_lower:
            return LLMProvider.OPENAI
        elif 'claude' in model_lower or 'anthropic' in model_lower:
            return LLMProvider.ANTHROPIC
        else:
            # Default to Google for unknown models since we're using Gemini
            logger.warning(f"Unknown provider for model {model}, defaulting to Google")
            return LLMProvider.GOOGLE
    
    def forward(self, prompt=None, messages=None, **kwargs):
        """
        Forward pass for the language model with token tracking.
        
        This is the required method that BaseLM expects subclasses to implement.
        The response should be identical to OpenAI response format.
        
        Args:
            prompt: The prompt to send to the LM
            messages: Alternative to prompt - list of message dicts
            **kwargs: Additional arguments for the LM call
            
        Returns:
            The LM response in OpenAI format
        """
        start_time = time.time()
        
        try:
            # Make the actual LM call through the underlying LM's forward method
            response = self.lm.forward(prompt=prompt, messages=messages, **kwargs)
            
            # Track the token usage
            self._track_usage(prompt, response, start_time, **kwargs)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in LM call: {e}")
            # Still try to track partial usage if possible
            self._track_error(prompt, e, start_time, **kwargs)
            raise
    
    def __call__(self, prompt=None, messages=None, **kwargs) -> Any:
        """
        Make an LM call with token tracking.
        
        This method is inherited from BaseLM and will call our forward method.
        
        Args:
            prompt: The prompt to send to the LM
            messages: Alternative to prompt - list of message dicts
            **kwargs: Additional arguments for the LM call
            
        Returns:
            The LM response with token usage tracked
        """
        # BaseLM.__call__ will call our forward method
        return super().__call__(prompt=prompt, messages=messages, **kwargs)
    
    async def aforward(self, prompt=None, messages=None, **kwargs):
        """
        Async forward pass for the language model with token tracking.
        
        This is the required async method that BaseLM expects subclasses to implement.
        
        Args:
            prompt: The prompt to send to the LM
            messages: Alternative to prompt - list of message dicts
            **kwargs: Additional arguments for the LM call
            
        Returns:
            The LM response in OpenAI format
        """
        start_time = time.time()
        
        try:
            # Make the actual async LM call through the underlying LM's aforward method
            response = await self.lm.aforward(prompt=prompt, messages=messages, **kwargs)
            
            # Track the token usage
            self._track_usage(prompt, response, start_time, **kwargs)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in async LM call: {e}")
            # Still try to track partial usage if possible
            self._track_error(prompt, e, start_time, **kwargs)
            raise
    
    def set_operation(self, operation: str):
        """Set the current operation context for token tracking."""
        self._current_operation = operation
        logger.debug(f"Token tracking operation set to: {operation}")
    
    def get_operation(self) -> str:
        """Get the current operation context."""
        return self._current_operation
    
    def _track_usage(self, prompt: Union[str, List[Dict[str, Any]]], response: Any, start_time: float, **kwargs):
        """Track token usage for a successful LM call."""
        try:
            # Extract token usage from the response
            usage_info = extract_token_usage(response, self.provider)
            
            # Extract operation from kwargs, fallback to current operation context
            operation = kwargs.pop("operation", self._current_operation)
            
            if usage_info:
                prompt_tokens = usage_info["prompt_tokens"]
                completion_tokens = usage_info["completion_tokens"]
                total_tokens = usage_info["total_tokens"]
                
                # Track usage with our TokenTracker
                usage = self.token_tracker.track_usage(
                    provider=self.provider,
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    operation=operation,
                    metadata={
                        "call_duration": time.time() - start_time,
                        "prompt_type": type(prompt).__name__,
                        "prompt_length": len(str(prompt)),
                        "response_type": type(response).__name__,
                        "dspy_call": True,
                        **kwargs
                    }
                )
                
                # Update internal statistics
                with self._lock:
                    self.call_count += 1
                    self.total_tokens += total_tokens
                    self.total_cost += usage.cost_usd
                
                logger.debug(f"Tracked DSPy call: {prompt_tokens}+{completion_tokens}={total_tokens} tokens, ${usage.cost_usd:.6f}")
                
            else:
                # Fallback: track a call without token details
                logger.warning(f"Could not extract token usage from {self.provider} response")
                with self._lock:
                    self.call_count += 1
                
        except Exception as e:
            logger.error(f"Error tracking token usage: {e}")
            with self._lock:
                self.call_count += 1
    
    def _track_error(self, prompt: Union[str, List[Dict[str, Any]]], error: Exception, start_time: float, **kwargs):
        """Track a failed LM call."""
        try:
            # Extract operation from kwargs, fallback to current operation context
            operation = kwargs.pop("operation", self._current_operation)
            error_operation = f"{operation}_error"
            
            # Track the failed call in metadata
            self.token_tracker.track_usage(
                provider=self.provider,
                model=self.model,
                prompt_tokens=0,  # Unknown for failed calls
                completion_tokens=0,
                operation=error_operation,
                metadata={
                    "call_duration": time.time() - start_time,
                    "prompt_type": type(prompt).__name__,
                    "prompt_length": len(str(prompt)),
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "dspy_call": True,
                    "failed": True,
                    **kwargs
                }
            )
            
            with self._lock:
                self.call_count += 1
                
        except Exception as track_error:
            logger.error(f"Error tracking failed call: {track_error}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current usage statistics for this LM instance."""
        with self._lock:
            return {
                "model": self.model,
                "session_id": self.session_id,
                "call_count": self.call_count,
                "total_tokens": self.total_tokens,
                "total_cost": self.total_cost,
                "avg_tokens_per_call": self.total_tokens / self.call_count if self.call_count > 0 else 0,
                "avg_cost_per_call": self.total_cost / self.call_count if self.call_count > 0 else 0
            }
    
    def print_summary(self):
        """Print a summary of token usage for this LM instance."""
        stats = self.get_stats()
        
        print(f"\n{'='*50}")
        print(f"DSPy LM Token Usage Summary")
        print(f"{'='*50}")
        print(f"Model: {stats['model']}")
        print(f"Session: {stats['session_id']}")
        print(f"Total Calls: {stats['call_count']:,}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print(f"Total Cost: ${stats['total_cost']:.4f}")
        print(f"Avg Tokens/Call: {stats['avg_tokens_per_call']:.1f}")
        print(f"Avg Cost/Call: ${stats['avg_cost_per_call']:.4f}")
        print(f"{'='*50}\n")
    
    def copy(self, **kwargs):
        """Returns a copy of the language model with possibly updated parameters."""
        # Create a copy using BaseLM's copy method
        new_instance = super().copy(**kwargs)
        
        # Copy our custom tracking attributes
        new_instance.session_id = self.session_id
        new_instance.token_tracker = self.token_tracker  # Share the same tracker instance
        new_instance.provider = self.provider
        new_instance.call_count = 0  # Reset call stats for new instance
        new_instance.total_tokens = 0
        new_instance.total_cost = 0.0
        new_instance._lock = threading.Lock()
        new_instance._current_operation = self._current_operation  # Preserve operation context
        new_instance.lm = self.lm  # Share the same underlying LM
        
        return new_instance
    
    # Forward other attributes to the underlying LM
    def __getattr__(self, name):
        return getattr(self.lm, name)


def configure_dspy_with_token_tracking(model: str, session_id: str = None, tracker: TokenTracker = None, **kwargs) -> TokenTrackingLM:
    """
    Configure DSPy with token tracking enabled.
    
    Args:
        model: Model identifier (e.g., 'gemini/gemini-2.5-flash')
        session_id: Session ID for token tracking (used if tracker not provided)
        tracker: Pre-created TokenTracker instance (optional)
        **kwargs: Additional arguments for the LM
        
    Returns:
        TokenTrackingLM instance configured for DSPy
    """
    # Create the token tracking LM
    tracking_lm = TokenTrackingLM(model=model, session_id=session_id, tracker=tracker, **kwargs)
    
    # Configure DSPy to use our tracking LM
    dspy.configure(lm=tracking_lm)
    
    logger.info(f"DSPy configured with token tracking for {model}")
    return tracking_lm


if __name__ == "__main__":
    # Test the token tracking wrapper
    print("Testing DSPy Token Tracking Wrapper...")
    
    # Create a token tracker first (like in mipro_skills.py)
    test_tracker = TokenTracker.get_instance(
        session_id='test_dspy_wrapper',
        log_dir='./test_token_logs'
    )
    
    # Configure DSPy with token tracking using the created tracker
    tracking_lm = configure_dspy_with_token_tracking(
        model='gemini/gemini-2.5-flash',
        tracker=test_tracker
    )
    
    # Test a simple call
    class SimpleSignature(dspy.Signature):
        """Answer a simple question."""
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
    
    predictor = dspy.Predict(SimpleSignature)
    
    # Make a few test calls
    print("Making test calls...")
    for i in range(3):
        result = predictor(question=f"What is {i+1} + {i+1}?")
        print(f"Q: What is {i+1} + {i+1}? A: {result.answer}")
    
    # Print summary
    tracking_lm.print_summary()
    test_tracker.print_summary()
