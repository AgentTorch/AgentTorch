"""
Claude LLM Backend for AgentTorch P3O Integration

Uses singleton client manager for optimal performance and resource usage.
"""

import os
import re
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .backend import LLMBackend
from .claude_client_manager import ClaudeClientManager


class ClaudeLocal(LLMBackend):
    """
    Local Claude backend for AgentTorch using singleton client manager.
    Optimized for performance with centralized client management.
    """

    def __init__(self, model_name="claude-3-haiku-20240307", temperature=0.1, max_tokens=1000, system_prompt=""):
        """Initialize Claude backend with singleton client."""
        self.backend = "claude"
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Use singleton client manager instead of creating new client
        self.client = ClaudeClientManager.get_client(f"ClaudeLocal_{model_name}")
        
        print(f"Claude backend initialized: {self.model_name}")

    def prompt(self, prompt_list: List[str]) -> List[str]:
        """Process multiple prompts using singleton client."""
        if not prompt_list:
            return []

        results = []
        
        # Process prompts with threading for efficiency
        try:
            with ThreadPoolExecutor(max_workers=min(10, len(prompt_list))) as executor:
                futures = {executor.submit(self._process_single_prompt, prompt): prompt 
                          for prompt in prompt_list}
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    
                    # Track API call in singleton manager
                    ClaudeClientManager.track_api_call()
                    
        except Exception as e:
            print(f"❌ Error in Claude backend: {e}")
            results = [f"Error: {str(e)}" for _ in prompt_list]

        return results

    def _process_single_prompt(self, prompt: str) -> str:
        """Process a single prompt through Claude API"""
        try:
            # DEBUG: Print concise prompt info per agent
            agent_info = "UNKNOWN"
            # Try to extract SOC code from prompt for identification
            import re
            soc_match = re.search(r'SOC (\d{2}-\d{4}\.\d{2})', prompt)
            if soc_match:
                agent_info = f"SOC {soc_match.group(1)}"
            
            # Extract job name if available from the actual prompt format
            job_match = re.search(r'You are a ([^(]+) \(SOC', prompt)
            job_name = job_match.group(1).strip() if job_match else "Unknown Job"
            
            # print(f"Agent [{agent_info}]: {job_name}")
            # print(f"  System: {self.system_prompt[:60]}..." if self.system_prompt else "  System: None")
            
            # DEBUG: Show the actual prompt to verify rich job data (longer snippet)
            # print(f"  📝 PROMPT: {prompt[:800]}..." if len(prompt) > 800 else f"  📝 PROMPT: {prompt}")
            # print("  " + "="*50)
            
            # Call Claude API using singleton client
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract text response
            raw_response = message.content[0].text if message.content else ""
            
            # Process and extract structured response  
            structured_response = self._extract_structured_response(raw_response)
            return structured_response
            
        except Exception as e:
            print(f"❌ Claude API error: {e}")
            return f"Claude API Error: {str(e)}"

    def inspect_history(self, file_dir="./memories") -> Dict[str, str]:
        """
        Inspect conversation history (not implemented for Claude)
        """
        print("⚠️ inspect_history not implemented for Claude backend")
        os.makedirs(file_dir, exist_ok=True)
        history_path = os.path.join(file_dir, "claude_history.md")
        with open(history_path, 'w') as f:
            f.write("Claude history inspection not implemented\\n")
        return {"path": history_path}

    def _extract_structured_response(self, raw_response: str) -> str:
        """Extract structured response from Claude's text output"""
        # Look for willingness score in various formats
        patterns = [
            r'"willingness":\s*(0?\.\d+|1\.0*|[01])',  # JSON format
            r'WILLINGNESS:\s*(0?\.\d+|1\.0*|[01])',    # Key-value format  
            r'willingness.*?(0?\.\d+|1\.0*)',          # Natural language
            r'^(0?\.\d+|1\.0*|[01])$',                 # Raw decimal number (like 0.75)
            r'\b(0?\.\d+|1\.0*)\b',                    # Any decimal in the text
        ]
        
        willingness_score = None
        for pattern in patterns:
            match = re.search(pattern, raw_response, re.IGNORECASE)
            if match:
                try:
                    willingness_score = float(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        # Default fallback if no score found
        if willingness_score is None:
            willingness_score = 0.5
        
        # Return structured format expected by AgentTorch
        reasoning = raw_response[:200] + "..." if len(raw_response) > 200 else raw_response
        return f"REASONING: {reasoning} | WILLINGNESS: {willingness_score}"


# Alias for backward compatibility
ClaudeLLM = ClaudeLocal 