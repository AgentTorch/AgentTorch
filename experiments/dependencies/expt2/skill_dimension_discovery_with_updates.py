"""
Enhanced Skill Dimension Discovery with Updates
==============================================

This script discovers and updates skill dimensions for job analysis.
It can:
1. Discover new skill dimensions
2. Update/refine existing skill dimensions
3. Merge similar dimensions
4. Remove redundant dimensions
5. Improve dimension quality over iterations

The algorithm:
s = {}  # Start with empty set
for job in jobs:
    s = update_skill_dimensions(job, s)  # Can add new OR update existing
return s
"""

import logging
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Optional, Tuple
from pydantic import BaseModel
from llm_utils import call_llm_structured_output
import instructor
from anthropic import Anthropic
from openai import OpenAI
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

# Input data path
input_data_path = "./data/job_data_processed/job_data_processed.csv"
output_dir = "./expt2_data/skill_dimensions_updated"
skill_extraction_dump_dir = "./expt2_data/skill_extraction_dump"


class SkillDimension(BaseModel):
    """Represents a skill dimension with metadata."""
    dimension_name: str  # The skill dimension name (≤5 words)
    description: str     # Brief description of what this dimension represents
    confidence: float    # Confidence score (0-1) for this dimension discovery
    usage_count: Optional[int] = 0  # How many jobs use this dimension
    last_updated:  Optional[str] = ""  # When this dimension was last updated
    synonyms: Optional[List[str]] = []  # Alternative names for this dimension
    


class SkillDimensionResponse(BaseModel):
    """Response from LLM for skill dimension discovery/updates."""
    new_dimensions: List[SkillDimension] = []  # New dimensions to add
    updated_dimensions: List[SkillDimension] = []  # Existing dimensions to update
    removed_dimensions: List[str] = []  # Names of dimensions to remove
    reasoning: str  # Reasoning for the updates


class DimensionScore(BaseModel):
    """Represents a score for a skill dimension for a specific job."""
    dimension_name: str
    relevance_score: int  # Strict 0 or 1 indicating relevance
    reasoning: str  # Reasoning for the score


class JobVectorResponse(BaseModel):
    """Response from LLM for job vector creation."""
    dimension_scores: List[DimensionScore]  # Scores for each dimension
    summary: str  # Summary of the job's key characteristics


def safe_str_convert(val) -> str:
    """Safely convert value to string, handling pandas null values."""
    if pd.isna(val):
        return ""
    return str(val)


def safe_array_convert(val) -> List[str]:
    """Safely convert value to array, splitting by newlines."""
    if pd.isna(val):
        return []
    str_val = str(val)
    if not str_val.strip():
        return []
    # Split by newlines and filter out empty strings
    return [item.strip() for item in str_val.split('\n') if item.strip()]


def read_job_data(input_path: str, max_jobs: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read job data from CSV file and convert to list of job dictionaries."""
    logger.info(f"Reading job data from: {input_path}")
    
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} job samples")
        
        if max_jobs is not None:
            df = df[:max_jobs]
            logger.info(f"Processing first {max_jobs} jobs")
        
        jobs = []
        for i, (_, row) in enumerate(df.iterrows()):
            soc_code = safe_str_convert(row['soc_code'])
            job_name = safe_str_convert(row['job_name'])
            detailed_work_activities = safe_array_convert(row.get('DetailedWorkActivities', ''))
            skills = safe_array_convert(row.get('Skills', ''))
            
            job = {
                "onet_code": soc_code,
                "job_title": job_name,
                "detailed_work_activities": detailed_work_activities,
                "skills": skills
            }
            
            jobs.append(job)
            
            if i < 3:
                logger.info(f"Job {i+1}: {job_name}")
                logger.info(f"  Skills: {len(skills)} items")
                logger.info(f"  Work Activities: {len(detailed_work_activities)} items")
        
        logger.info(f"Processed {len(jobs)} jobs successfully")
        return jobs
        
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return []


def format_existing_dimensions(existing_dimensions: Dict[str, SkillDimension]) -> str:
    """Format existing dimensions for the prompt."""
    if not existing_dimensions:
        return "None"
    
    formatted = []
    for name, dim in existing_dimensions.items():
        formatted.append(f"- {name}: {dim.description} (confidence: {dim.confidence:.2f}, usage: {dim.usage_count})")
    
    return "\n".join(formatted)


def create_skill_update_prompt(job: Dict[str, Any], existing_dimensions: Dict[str, SkillDimension]) -> str:
    """
    Create a prompt for the LLM to discover new dimensions AND update existing ones.
    
    Args:
        job: Job dictionary with skills and work activities
        existing_dimensions: Dictionary of existing skill dimensions
        
    Returns:
        Formatted prompt string for LLM
    """
    skills_text = "\n".join([f"- {skill}" for skill in job['skills']])
    activities_text = "\n".join([f"- {activity}" for activity in job['detailed_work_activities']])
    existing_dims_text = format_existing_dimensions(existing_dimensions)
    
    prompt = f"""
You are an expert job analyst tasked with discovering and updating skill dimensions for job classification.

JOB INFORMATION:
Job Title: {job['job_title']}
ONET Code: {job['onet_code']}

JOB SKILLS:
{skills_text}

JOB DETAILED WORK ACTIVITIES:
{activities_text}

EXISTING SKILL DIMENSIONS so far:
{existing_dims_text}

CRITICAL CONSTRAINT:
The skill dimension set must represent ALL jobs analyzed so far, not just the current job. 
DO NOT REMOVE any existing dimensions as they may be essential for previous jobs.
Instead, focus on adding new dimensions or updating existing ones to better represent the current job.

TASK:
Analyze this job and the existing skill dimensions. You can:

1. ADD NEW DIMENSIONS: Identify new skill dimensions (≤5 words) not covered by existing ones
2. UPDATE EXISTING DIMENSIONS: Improve descriptions, confidence scores, or add synonyms
3. REMOVE DIMENSIONS: Remove dimensions that are redundant, too specific, or not generalizable;; OR in case you want to merge two dimensions into one - you would delete 2 and create 1

GUIDELINES:
- New dimensions should be ≤5 words, specific, and meaningful
- Updates should improve clarity and accuracy while preserving relevance to previous jobs
- Only remove dimensions if they are truly redundant or not generalizable across jobs
- When removing a dimension, provide clear reasoning why it should be removed

EXAMPLES:
- NEW: "Technical Programming", "Customer Service", "Data Analysis"
- UPDATE: Improve description of "Problem Solving" to be more specific
- REMOVE: "Specific Tool Knowledge" (too specific, better covered by broader technical skills)

Return a comprehensive update that may include new, updated, and merged dimensions.
Focus on creating a comprehensive set of skill dimensions that represents ALL jobs, not just the current one.
"""

    return prompt


def create_vector_scoring_prompt(job: Dict[str, Any], skill_dimensions: Dict[str, SkillDimension]) -> str:
    """
    Create a prompt for the LLM to score skill dimensions for a job with strict 0 or 1 values.
    
    Args:
        job: Job dictionary with skills and work activities
        skill_dimensions: Dictionary of all skill dimensions to score
        
    Returns:
        Formatted prompt string for LLM
    """
    skills_text = "\n".join([f"- {skill}" for skill in job['skills']])
    activities_text = "\n".join([f"- {activity}" for activity in job['detailed_work_activities']])
    
    dimensions_text = "\n".join([f"- {dim}" for dim in sorted(skill_dimensions.keys())])
    
    prompt = f"""
Score skill dimensions for this job with 0 or 1:

JOB: {job['job_title']} ({job['onet_code']})

SKILLS: 
{skills_text}
ACTIVITIES: {activities_text}

DIMENSIONS TO SCORE:
{dimensions_text}

RULES:
- 0: NOT relevant to this job
- 1: RELEVANT to this job (even minor relevance)

Score each dimension as 0 or 1. Provide brief reasoning for each score.
"""

    return prompt

def skill_extraction_dump(job_code: str, step: str, skill_dimensions: SkillDimensionResponse) -> str:
    """Dump skill dimensions response to JSON file with step information."""
    os.makedirs(skill_extraction_dump_dir, exist_ok=True)
    path_dump = os.path.join(skill_extraction_dump_dir, f"{job_code}.json")
    
    # Convert pydantic model to dict and add step information
    skill_dimensions_dump = skill_dimensions.model_dump()
    skill_dimensions_dump["step"] = step
    
    # Write to file
    with open(path_dump, 'w') as f:
        json.dump(skill_dimensions_dump, f, indent=2, default=str)
    logger.info(f"Saved skill extraction dump to: {path_dump}")
    
    return path_dump

def update_skill_dimensions_enhanced(job: Dict[str, Any], existing_dimensions: Dict[str, SkillDimension], step: int = 0) -> Dict[str, SkillDimensionResponse]:
    """
    Enhanced function to update skill dimensions by analyzing a job.
    Can add new dimensions, update existing ones, merge similar ones, or remove redundant ones.
    
    Args:
        job: Job dictionary with skills and work activities
        existing_dimensions: Current dictionary of skill dimensions
        
    Returns:
        Updated dictionary of skill dimensions
    """
    logger.info(f"Analyzing job: {job['job_title']}")
    
    # Create prompt for LLM
    prompt = create_skill_update_prompt(job, existing_dimensions)
    
    try:
        # Call LLM to get updates
        response: SkillDimensionResponse = call_llm_structured_output(
            prompt, 
            SkillDimensionResponse,
            model='claude-3-5-haiku-latest'
        )
    
        updated_dimensions = existing_dimensions.copy()
        
        # Process new dimensions
        for new_dim in response.new_dimensions:
            if new_dim.dimension_name not in updated_dimensions:
                new_dim.usage_count = 1  # This job gave rise to this dimension
                updated_dimensions[new_dim.dimension_name] = new_dim
                logger.info(f"  ADDED: {new_dim.dimension_name} (confidence: {new_dim.confidence:.2f})")
        
        # Process updated dimensions
        for updated_dim in response.updated_dimensions:
            if updated_dim.dimension_name in updated_dimensions:
                # Update existing dimension
                old_dim = updated_dimensions[updated_dim.dimension_name]
                updated_dim.usage_count = old_dim.usage_count + 1  # Increment usage count as this job also uses it
                updated_dimensions[updated_dim.dimension_name] = updated_dim
                logger.info(f"  UPDATED: {updated_dim.dimension_name} (usage: {updated_dim.usage_count})")
        
        # Process removed dimensions
        for dim_name in response.removed_dimensions:
            if dim_name in updated_dimensions:
                del updated_dimensions[dim_name]
                logger.info(f"  REMOVED: {dim_name}")

        skill_extraction_dump(job['onet_code'], f"{step}", response)
        
        # Note: Usage counts track how many jobs contributed to discovering/updating each dimension
        # This helps understand which dimensions emerged from more jobs during discovery
        
        logger.info(f"  Total dimensions after update: {len(updated_dimensions)}")
        logger.info(f"  Reasoning: {response.reasoning[:100]}...")
        
        return updated_dimensions
        
    except Exception as e:
        logger.error(f"Error calling LLM for job {job['job_title']}: {e}")
        return existing_dimensions


def discover_and_update_skill_dimensions(jobs: List[Dict[str, Any]], max_iterations: int = 3) -> Dict[str, SkillDimension]:
    """
    Discover and iteratively update skill dimensions across all jobs.
    
    Args:
        jobs: List of job dictionaries
        max_iterations: Maximum number of iterations for refinement
        
    Returns:
        Dictionary of refined skill dimensions
    """
    logger.info("Starting enhanced skill dimension discovery and updates...")
    logger.info("=" * 60)
    
    # Initialize empty dictionary of skill dimensions
    skill_dimensions = {}
    
    # Multiple iterations for refinement
    for iteration in range(max_iterations):
        logger.info(f"\n=== ITERATION {iteration + 1}/{max_iterations} ===")
        
        # Process each job
        for i, job in enumerate(jobs):
            logger.info(f"\nProcessing job {i+1}/{len(jobs)}")
            
            # Update skill dimensions based on this job
            skill_dimensions = update_skill_dimensions_enhanced(job, skill_dimensions, i)
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(jobs)} jobs processed, {len(skill_dimensions)} dimensions")
                logger.info("Current skill dimensions:")
                for dim_name, dim in skill_dimensions.items():
                    logger.info(f"  - {dim_name}: {dim.description} (confidence: {dim.confidence:.2f}, usage: {dim.usage_count})")
        
        # Analyze current state
        logger.info(f"\nAfter iteration {iteration + 1}:")
        logger.info(f"Total dimensions: {len(skill_dimensions)}")
        
        # Show top dimensions by usage
        sorted_dims = sorted(skill_dimensions.items(), key=lambda x: x[1].usage_count, reverse=True)
        logger.info("Top 5 dimensions by usage:")
        for i, (name, dim) in enumerate(sorted_dims[:5]):
            logger.info(f"  {i+1}. {name}: {dim.usage_count} uses (confidence: {dim.confidence:.2f})")
    
    logger.info("\n" + "=" * 60)
    logger.info("ENHANCED SKILL DIMENSION DISCOVERY COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final total unique skill dimensions: {len(skill_dimensions)}")
    
    # Display all discovered dimensions
    logger.info("\nFinal skill dimensions:")
    for i, (name, dim) in enumerate(sorted(skill_dimensions.items()), 1):
        logger.info(f"  {i:2d}. {name}: {dim.description} (usage: {dim.usage_count}, confidence: {dim.confidence:.2f})")
    
    return skill_dimensions


def create_job_vector_with_updated_dimensions(job: Dict[str, Any], skill_dimensions: Dict[str, SkillDimension]) -> Dict[str, float]:
    """
    Create a sparse vector representation for a job using LLM-based scoring with strict 0 or 1 values.
    
    Args:
        job: Job dictionary
        skill_dimensions: Dictionary of skill dimensions
        
    Returns:
        Dictionary mapping skill dimensions to relevance scores (0 or 1)
    """
    logger.info(f"Creating vector for job: {job['job_title']}")
    
    # Create prompt for LLM
    prompt = create_vector_scoring_prompt(job, skill_dimensions)
    
    try:
        response: JobVectorResponse = call_llm_structured_output(
            prompt, 
            JobVectorResponse,
            model='claude-3-5-haiku-latest'
        )
        
        # Convert to dictionary with strict 0 or 1 values
        vector = {}
        for score in response.dimension_scores:
            # Ensure the score is strictly 0 or 1
            relevance_score = 1 if score.relevance_score > 0 else 0
            vector[score.dimension_name] = relevance_score
            
            if relevance_score == 1:  # Only log relevant dimensions
                logger.info(f"  {score.dimension_name}: {relevance_score} - {score.reasoning[:50]}...")
        
        logger.info(f"  Summary: {response.summary[:100]}...")
        
        return vector
        
    except Exception as e:
        logger.error(f"Error calling LLM for vector creation: {e}")
        # Fallback to all zeros if LLM fails
        vector = {}
        for dim_name in skill_dimensions:
            vector[dim_name] = 0
        return vector


def process_single_job_vector(job_data: tuple) -> Dict[str, Any]:
    """
    Process a single job to create its skill vector.
    
    Args:
        job_data: Tuple of (job_index, job, skill_dimensions, lock)
        
    Returns:
        Job dictionary with added skill vector and sparsity
    """
    job_index, job, skill_dimensions = job_data
    
    logger.info(f"Creating vector for job {job_index+1}: {job['job_title']}")
    
    vector = create_job_vector_with_updated_dimensions(job, skill_dimensions)
    
    # No need to update usage counts here - will be calculated after all vectors are created
    
    job_with_vector = job.copy()
    job_with_vector['skill_vector'] = vector
    job_with_vector['original_index'] = job_index  # Store original index for sorting
    
    # Calculate sparsity (now with strict 0/1 values)
    non_zero_scores = sum(1 for score in vector.values() if score == 1)  # Count 1s
    sparsity = 1.0 - (non_zero_scores / len(skill_dimensions))
    job_with_vector['sparsity'] = sparsity
    
    logger.info(f"  Job {job_index+1} - Sparsity: {sparsity:.3f} ({non_zero_scores}/{len(skill_dimensions)} dimensions relevant)")
    
    return job_with_vector


def create_job_vectors_parallel(jobs: List[Dict[str, Any]], skill_dimensions: Dict[str, SkillDimension], max_workers: int = 5) -> List[Dict[str, Any]]:
    """
    Create skill vectors for all jobs using parallel processing.
    
    Args:
        jobs: List of job dictionaries
        skill_dimensions: Dictionary of skill dimensions
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of jobs with added skill vectors
    """
    logger.info(f"\nCreating sparse vectors for all jobs using {max_workers} parallel workers...")
    
    # Prepare job data for parallel processing
    job_data_list = [(i, job, skill_dimensions) for i, job in enumerate(jobs)]
    
    jobs_with_vectors = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs for processing
        future_to_job = {executor.submit(process_single_job_vector, job_data): job_data[0] for job_data in job_data_list}
        
        # Process completed jobs in order of completion
        for future in as_completed(future_to_job):
            job_index = future_to_job[future]
            try:
                job_with_vector = future.result()
                jobs_with_vectors.append(job_with_vector)
                logger.info(f"Completed job {job_index+1}/{len(jobs)}")
            except (ConnectionError, TimeoutError) as e:
                logger.error(f"Network error processing job {job_index+1}: {e}")
                # Add a fallback job with zero vector
                fallback_job = jobs[job_index].copy()
                fallback_job['skill_vector'] = {dim: 0 for dim in skill_dimensions}
                fallback_job['sparsity'] = 1.0
                fallback_job['original_index'] = job_index
                jobs_with_vectors.append(fallback_job)
            except Exception as e:
                logger.error(f"Unexpected error processing job {job_index+1}: {e}")
                # Add a fallback job with zero vector
                fallback_job = jobs[job_index].copy()
                fallback_job['skill_vector'] = {dim: 0 for dim in skill_dimensions}
                fallback_job['sparsity'] = 1.0
                fallback_job['original_index'] = job_index
                jobs_with_vectors.append(fallback_job)
    
    # Sort by original job index to maintain order
    jobs_with_vectors.sort(key=lambda x: x['original_index'])
    
    logger.info(f"Completed vector creation for {len(jobs_with_vectors)} jobs")
    return jobs_with_vectors


def save_skill_dimensions_only(skill_dimensions: Dict[str, SkillDimension], output_dir: str, filename: str = "skill_dimensions_before_vectors.json"):
    """Save skill dimensions to a JSON file before vector creation."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert skill dimensions to serializable format
    dimensions_data = {}
    for name, dim in skill_dimensions.items():
        dimensions_data[name] = {
            "description": dim.description,
            "confidence": dim.confidence,
            "usage_count": dim.usage_count,
            "last_updated": dim.last_updated,
            "synonyms": dim.synonyms
        }
    
    # Save skill dimensions
    dimensions_file = os.path.join(output_dir, filename)
    with open(dimensions_file, 'w') as f:
        json.dump(dimensions_data, f, indent=2)
    logger.info(f"Saved skill dimensions to: {dimensions_file}")
    logger.info(f"Total dimensions saved: {len(skill_dimensions)}")
    
    return dimensions_file


def save_updated_results(jobs_with_vectors: List[Dict[str, Any]], skill_dimensions: Dict[str, SkillDimension], output_dir: str):
    """Save results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert skill dimensions to serializable format
    dimensions_data = {}
    for name, dim in skill_dimensions.items():
        dimensions_data[name] = {
            "description": dim.description,
            "confidence": dim.confidence,
            "usage_count": dim.usage_count,
            "last_updated": dim.last_updated,
            "synonyms": dim.synonyms
        }
    
    # Save skill dimensions
    dimensions_file = os.path.join(output_dir, "updated_skill_dimensions.json")
    with open(dimensions_file, 'w') as f:
        json.dump(dimensions_data, f, indent=2)
    logger.info(f"Saved updated skill dimensions to: {dimensions_file}")
    
    # Save job vectors
    vectors_file = os.path.join(output_dir, "updated_job_vectors.json")
    with open(vectors_file, 'w') as f:
        json.dump(jobs_with_vectors, f, indent=2)
    logger.info(f"Saved updated job vectors to: {vectors_file}")
    
    # Create summary statistics
    summary = {
        "total_jobs": len(jobs_with_vectors),
        "total_dimensions": len(skill_dimensions),
        "average_sparsity": np.mean([job['sparsity'] for job in jobs_with_vectors]),
        "dimensions": list(skill_dimensions.keys()),
        "top_dimensions_by_usage": sorted(skill_dimensions.items(), key=lambda x: x[1].usage_count, reverse=True)[:10]
    }
    
    summary_file = os.path.join(output_dir, "updated_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved updated summary to: {summary_file}")

   
def main():
    """Main function to run enhanced skill dimension discovery and updates."""
    logger.info("Starting Enhanced Skill Dimension Discovery with Updates")
    logger.info("=" * 60)
    
    # Read job data
    jobs = read_job_data(input_data_path)  # Process all jobs
    
    if not jobs:
        logger.error("No jobs found. Exiting.")
        return
    
    # Discover and update skill dimensions
    skill_dimensions = discover_and_update_skill_dimensions(jobs, max_iterations=1)
    
    # Save skill dimensions before creating sparse vectors
    save_skill_dimensions_only(skill_dimensions, output_dir)
    
    # Create sparse vectors for all jobs
    jobs_with_vectors = create_job_vectors_parallel(jobs, skill_dimensions)
    
    # Save results
    save_updated_results(jobs_with_vectors, skill_dimensions, output_dir)
    
    # Final analysis
    logger.info("\n" + "=" * 60)
    logger.info("FINAL ANALYSIS")
    logger.info("=" * 60)
    
    sparsities = [job['sparsity'] for job in jobs_with_vectors]
    avg_sparsity = np.mean(sparsities)
    
    logger.info(f"Total jobs analyzed: {len(jobs_with_vectors)}")
    logger.info(f"Total skill dimensions: {len(skill_dimensions)}")
    logger.info(f"Average sparsity: {avg_sparsity:.3f}")
    
    # Show top dimensions by usage
    sorted_dims = sorted(skill_dimensions.items(), key=lambda x: x[1].usage_count, reverse=True)
    logger.info("\nTop 5 dimensions by usage:")
    for i, (name, dim) in enumerate(sorted_dims[:5]):
        logger.info(f"  {i+1}. {name}: {dim.usage_count} uses (confidence: {dim.confidence:.2f})")
    
    logger.info("\n" + "=" * 60)
    logger.info("ENHANCED SKILL DIMENSION DISCOVERY WITH UPDATES COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
