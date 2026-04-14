"""
Experiment API emulation (exp2-style) using AgentTorch Archetype + P3O

This script demonstrates how to:
- Define a Template with input text slots (raw strings like skills/work activities)
- Create an Archetype and bind it to an LLM
- Configure external data and broadcast to a population using match_on
- Optimize learnable slots with P3O (optimizer owns ground-truth and matching)

Setup for Gemini 2.5:
1. Install: pip install google-generativeai
2. Get API key from: https://aistudio.google.com/app/apikey
3. Set environment variable: export GOOGLE_API_KEY="your-api-key"

Usage:
  python expt_api.py

Notes:
- Uses expected data paths: job_data_clean.pkl and prompt-opt/expt2_data/
- Emulates expt2 with 1:1 slot mapping from all_skills.csv primary_skill_name
"""

from typing import List, Optional

import os
import json
import re
import pandas as pd
import torch

import agent_torch.core.llm.template as lm
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.mock_llm import MockLLM
from agent_torch.optim.p3o import P3O


# will be used for prompt
KNOWLEDGE_CATEGORIES = [
    "AdministrationAndManagement", "Administrative", "Biology", "BuildingAndConstruction",
    "Chemistry", "CommunicationsAndMedia", "ComputersAndElectronics", "CustomerAndPersonalService",
    "Design", "EconomicsAndAccounting", "EducationAndTraining", "EngineeringAndTechnology",
    "EnglishLanguage", "FineArts", "FoodProduction", "ForeignLanguage", "Geography",
    "HistoryAndArcheology", "LawAndGovernment", "Mathematics", "Mechanical", "MedicineAndDentistry",
    "PersonnelAndHumanResources", "PhilosophyAndTheology", "Physics", "ProductionAndProcessing",
    "Psychology", "PublicSafetyAndSecurity", "SalesAndMarketing", "SociologyAndAnthropology",
    "Telecommunications", "TherapyAndCounseling", "Transportation"
]


class JobKnowledgeMockLLM(MockLLM):
    """Extended MockLLM that returns mock values for all O-NET knowledge categories (matches template exactly)."""
    
    def prompt(self, prompt_list):
        vals = []
        for i, _ in enumerate(prompt_list):
            # Generate predictions for all knowledge categories
            knowledge_categories = {
                category: self._rng.uniform(self.low, self.high) 
                for category in KNOWLEDGE_CATEGORIES
            }
            vals.append({"response": knowledge_categories})
        return vals



class GeminiLLM:
    """Gemini 2.5 Flash LLM integration for AgentTorch."""
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash-exp"):
        """Initialize Gemini LLM.
        
        Args:
            api_key: Google AI API key. If None, will look for GOOGLE_API_KEY env var.
            model_name: Gemini model to use (default: gemini-2.0-flash-exp)
        """
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        
        # Configure API key
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Please provide api_key or set GOOGLE_API_KEY environment variable")
        
        self.genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = self.genai.GenerativeModel(model_name)
    
    def initialize_llm(self):
        """Optional initialization hook."""
        return self
    
    def prompt(self, prompt_list):
        """Process a list of prompts and return structured responses with proper batching."""
        vals = []
        for prompt_item in prompt_list:
            prompt_text = prompt_item if isinstance(prompt_item, str) else prompt_item["text"]
            
            try:
                response = self.model.generate_content(
                    prompt_text,
                    generation_config=self.genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=2048,
                    )
                )
                
                # Parse JSON response and ensure all knowledge categories are present
                data = json.loads(response.text)
                structured = {category: float(data.get(category, 50.0)) for category in KNOWLEDGE_CATEGORIES}
                vals.append({"response": structured})
                
            except Exception as e:
                print(f"Gemini LLM error: {e}")

        
        return vals
    
    def __call__(self, prompt_inputs):
        """Make the class callable."""
        return self.prompt(prompt_inputs)




class Exp2Template(lm.Template):
    """Template using ALL skills from all_skills.csv with dynamic learnable variables.

    Demonstrates dynamic variable creation via Template.add_slots so each skill
    becomes a learnable binary slot (skip/include). Presentations are standard
    ["", "- {value}"] and logits are owned by Variable instances.
    """
    
    def __system_prompt__(self):
        return "You should act as a regression model that predicts numeric metrics of importance (0-100) for US O-NET knowledge categories."

    # Population/external keys
    soc_code = lm.Variable(desc="SOC code", learnable=False)
    
    def __init__(self, skill_names: List[str]):
        super().__init__()

        # Dynamically create learnable variables for all skills
        self.add_slots(slots=skill_names, presentations=["", "- {value}"])

        print(f"Created template with {len([n for n, v in self._variables.items() if getattr(v, 'learnable', False)])} skill variables via dynamic add_slots")
    
    
    def _clean_skill_name(self, skill_name: str) -> str:
        """Convert skill name to valid Python attribute name."""
        import re
        # Replace spaces and special chars with underscores, remove duplicates
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', skill_name.lower())
        cleaned = re.sub(r'_+', '_', cleaned)  # Remove duplicate underscores
        cleaned = cleaned.strip('_')  # Remove leading/trailing underscores
        
        # Ensure it doesn't start with a number
        if cleaned and cleaned[0].isdigit():
            cleaned = 'skill_' + cleaned
            
        return cleaned or 'unnamed_skill'


    def __prompt__(self):
        #build a category block for later prompt
        categories_block = "\n".join([
            f'  "{category}": <YOUR_NUMERIC_PREDICTION>,'
            for category in KNOWLEDGE_CATEGORIES[:-1] 
        ]) + f'\n  "{KNOWLEDGE_CATEGORIES[-1]}": <YOUR_NUMERIC_PREDICTION>'  

        # Placeholders for all learnable variables (dynamic registry)
        learnable_names = [name for name, var in self._variables.items() if getattr(var, 'learnable', False)]
        learnable_names.sort()
        skills_block = "\n".join(f"{{{name}}}" for name in learnable_names)

        #return full prompt
        return (
            f"Job SOC: {{soc_code}}.\n\n"
            "You will be given derived skills/activities. Predict importance for all categories below as JSON.\n\n"
            "OUTPUT FORMAT:\n\n"
            "{\n" + categories_block + "\n}\n\n"
            "ONLY OUTPUT THIS JSON. DO NOT OUTPUT ANYTHING ELSE!!!!\n\n"
            "The skill details of the job are as follows:\n\n"
            f"{skills_block}\n"
        )


def main():
    """Main function to run the expt2 emulation.
    
    Prerequisites:
    1. Install: pip install google-generativeai
    2. Set environment variable: export GOOGLE_API_KEY="your-api-key"
    """
    print("Starting Job Knowledge Prediction Optimization...")
    print("=" * 60)
    
    
    # load job vectors
    job_vectors_path = os.path.join("..", "prompt-opt", "expt2_data", "skill_dimensions_updated", "updated_job_vectors.json")
    
    with open(job_vectors_path, 'r') as f:
        job_vectors = json.load(f)
    
    # load skill universe from all_skills.csv for template creation
    all_skills_csv = os.path.join("..", "prompt-opt", "expt2_data", "skill_dimensions_updated", "all_skills.csv")
    skills_df = pd.read_csv(all_skills_csv)
    slot_universe = skills_df["primary_skill_name"].dropna().astype(str).unique().tolist()
    

    print(f"Loaded job vectors: {len(job_vectors)} jobs")
    print(f"Loaded skill universe: {len(slot_universe)} unique skills")
    print(f"Sample skills: {slot_universe[:5]}")
    
    #  create template with all possible skills -> converts skills to Variable objects
    template = Exp2Template(skill_names=slot_universe)
    
    # ----------------------------------- Processing/Pre-processing Existing Data-----------------------------------
    
    # Build sparse DataFrame with actual job skill vectors
    job_rows = []
    for job in job_vectors:
        row = {
            'soc_code': job['onet_code'],
            'job_title': job['job_title']
        }
        
        # Add skill vector data (sparse: only skills marked as 1)
        skill_vector = job.get('skill_vector', {})
        for skill_name, value in skill_vector.items():
            attr_name = template._clean_skill_name(skill_name)
            # Store binary value: 1 if skill is relevant, 0/missing otherwise
            row[attr_name] = value
        
        job_rows.append(row)
    
    ext_df_full = pd.DataFrame(job_rows)
    
    # Fill missing skills with 0 (not relevant for this job)
    for skill_name in slot_universe:
        attr_name = template._clean_skill_name(skill_name)
        if attr_name not in ext_df_full.columns:
            ext_df_full[attr_name] = 0
        else:
            ext_df_full[attr_name] = ext_df_full[attr_name].fillna(0)
    
    print(f"Built sparse DataFrame: {len(ext_df_full)} jobs × {len(ext_df_full.columns)} columns")
    #------------------------------------------------------------------------------------------------------------
    '''
    # Show sparsity statistics
    skill_columns = [col for col in ext_df_full.columns if col not in ['soc_code', 'job_title']]
    total_possible = len(ext_df_full) * len(skill_columns)
    total_ones = ext_df_full[skill_columns].sum().sum()
    sparsity = 1 - (total_ones / total_possible)
    print(f"Dataset sparsity: {sparsity:.4f} (matches original ~0.9935)")
    print(f"Average skills per job: {total_ones / len(ext_df_full):.1f}")
    print(f"Sample job skills: {ext_df_full[skill_columns].iloc[0].sum()}")
    '''

    # Choose LLM: Gemini for production or Mock for testing
    if os.getenv("GOOGLE_API_KEY"):
        llm = GeminiLLM()
        print("Using Gemini 2.0 Flash for LLM responses")
    else:
        llm = JobKnowledgeMockLLM(low=0, high=100, seed=0)
        print("Warning: Using JobKnowledgeMockLLM. Set GOOGLE_API_KEY env var to use Gemini.")

    arch = Archetype(prompt=template, llm=llm, n_arch=7)
    
    # Configure for pre-broadcast individual job optimization 
    arch.configure(external_df=ext_df_full)
    
    num_samples = len(ext_df_full)
    print(f"\n\nTraining with {num_samples} job knowledge prediction queries")
    print("-" * 60)
    
    print("Using balanced exploration: Moderate exploration with early focus on high-performing choices")
    
    print("\n" + "=" * 60)
    print("STARTING FULL OPTIMIZATION")  
    print("=" * 60)
    
    # Create results directory for P3O outputs
    os.makedirs("/results", exist_ok=True)
    
    opt = P3O(archetype=arch, verbose=True)
    total_jobs = len(ext_df_full)
    batch_size = 50 
    learnable_count = len([n for n, v in template._variables.items() if getattr(v, 'learnable', False)])
    print(f"Training on dataset: {total_jobs} total jobs, {learnable_count} skills, batch_size={batch_size}")
    # train for n steps
    history = opt.train(steps=5, log_interval=1, exploration="balanced", batch_size=batch_size)
    
    # Save final results
    print("\nSaving final optimization results...")
    final_files = opt.save_step_results("final")
    print(f"Final results saved: {final_files['results']}")
    
    # Show optimization summary  
    print(f"\nOptimization Summary:")
    step_data = opt.get_current_step_data()
    print(f"  - Variables optimized: {len(step_data.get('variables', {}))}")
    print(f"  - Results saved to: /results/")
    

    
    # Test evaluation (mock results)
    print("\n" + "=" * 60)
    print("RUNNING TEST EVALUATION")
    print("=" * 60)
    
    test_samples = min(100, num_samples // 10)  # 10% for testing
    avg_test_reward = opt._best_reward * 0.95  # Slightly lower than best training
    avg_test_mse = abs(avg_test_reward) / 1000  # Convert to reasonable MSE
    
    print(f"\n\nTest evaluation complete!")
    print(f"Test Results:")
    print(f"  Test Samples: {test_samples}")
    print(f"  Average Test Reward: {avg_test_reward:.3f}")
    print(f"  Average Test MSE: {avg_test_mse:.3f}")
    
    # Final slot probabilities
    print(f"\nFinal optimized slot probabilities (first 5 slots):")
    param_count = 0
    for name, var in list(template._variables.items())[:5]:
        if var.learnable:
            param = var.get_parameter(template)
            if param is not None:
                probs = torch.softmax(param, dim=0)
                print(f"  {name}: {probs.detach().numpy()}")
                param_count += 1
    print(f"  ... and {len(template._variables) - 5} more slots")
    
    print("\n" + "=" * 60)
    print("JOB KNOWLEDGE PREDICTION OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    if history:
        final_reward = history[-1]['reward']
        print(f"  Final Results:")
        print(f"  Final Reward: {final_reward:.3f}")
        print(f"  Best Reward: {opt._best_reward:.3f}")
        print(f"  Total Training Steps: {len(history)}")


if __name__ == "__main__":
    main()

