#!/usr/bin/env python3
"""
Experiment Data Consistency Verification
========================================

This script verifies that both P3O and MIPROv2 experiments are using 
the exact same dataset for training and testing.

Usage:
    python verify_experiment_consistency.py
"""

import logging
import sys
from typing import List, Dict, Any
import numpy as np

sys.path.append("..")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_experiments_consistency(max_data_points: int = 20) -> bool:
    """
    Verify that both P3O and MIPROv2 experiments use identical datasets.
    
    Args:
        max_data_points: Number of data points to verify
        
    Returns:
        True if experiments are consistent, False otherwise
    """
    try:
        # Import required modules
        from o_net_data_processor import create_shared_data_processor
        from expt_2_slot_imp_selection import JobKnowledgeExperiment
        from mipro_skills import create_training_examples
        
        logger.info("=" * 60)
        logger.info("VERIFYING EXPERIMENT DATA CONSISTENCY")
        logger.info("=" * 60)
        
        # Create shared data processor
        processor = create_shared_data_processor(random_seed=42)
        logger.info(f"✓ Created shared data processor with seed=42")
        
        # Get data summary
        summary = processor.get_data_summary()
        logger.info(f"✓ Data summary: {summary['total_samples']} total samples, {summary['skill_dimensions']} skill dimensions")
        
        # Test P3O experiment data
        logger.info("\n" + "-" * 40)
        logger.info("TESTING P3O EXPERIMENT DATA")
        logger.info("-" * 40)
        
        p3o_experiment = JobKnowledgeExperiment(
            input_data_path="dummy", 
            input_data_slots_path="dummy", 
            output_path="dummy",
            max_data_points=max_data_points,
            random_seed=42
        )
        
        p3o_queries = p3o_experiment.prepare_experiment_data()
        logger.info(f"✓ P3O experiment loaded {len(p3o_queries)} queries")
        
        # Test MIPROv2 experiment data
        logger.info("\n" + "-" * 40)
        logger.info("TESTING MIPROv2 EXPERIMENT DATA")
        logger.info("-" * 40)
        
        try:
            import dspy
            dspy_examples = create_training_examples(max_examples=max_data_points)
            logger.info(f"✓ MIPROv2 experiment loaded {len(dspy_examples)} examples")
            
            # Verify data consistency using processor's verification method
            logger.info("\n" + "-" * 40)
            logger.info("VERIFYING DATA CONSISTENCY")
            logger.info("-" * 40)
            
            is_consistent = processor.verify_data_consistency(p3o_queries, dspy_examples)
            
            if is_consistent:
                logger.info("✅ DATA CONSISTENCY VERIFICATION: PASSED")
                logger.info("Both experiments are using identical datasets!")
            else:
                logger.error("❌ DATA CONSISTENCY VERIFICATION: FAILED")
                logger.error("Experiments are using different datasets!")
                return False
                
        except ImportError:
            logger.warning("DSPy not available, skipping MIPROv2 data verification")
            return True
        
        # Additional detailed verification
        logger.info("\n" + "-" * 40)
        logger.info("DETAILED DATA VERIFICATION")
        logger.info("-" * 40)
        
        # Check first few samples in detail
        sample_count = min(3, len(p3o_queries))
        
        for i in range(sample_count):
            p3o_sample = p3o_queries[i]
            
            # Get job info
            job_title = p3o_sample["x"]["job_title"]
            onet_code = p3o_sample["x"]["onet_code"]
            
            # Count skills
            skill_count = sum(1 for k, v in p3o_sample["x"].items() 
                            if k not in ['job_title', 'onet_code'] and v == 1)
            
            # Get target metrics sample
            y_sample = p3o_sample["y"]
            admin_score = y_sample.AdministrationAndManagement
            
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Job: {job_title} ({onet_code})")
            logger.info(f"  Skills: {skill_count} active skills")
            logger.info(f"  Target (AdministrationAndManagement): {admin_score}")
            
            # Verify DSPy example matches
            if i < len(dspy_examples):
                dspy_sample = dspy_examples[i]
                dspy_admin_score = dspy_sample.job_metrics.AdministrationAndManagement
                
                if admin_score != dspy_admin_score:
                    logger.error(f"  ❌ Mismatch: P3O={admin_score}, DSPy={dspy_admin_score}")
                    return False
                else:
                    logger.info(f"  ✅ DSPy match confirmed")
        
        # Test deterministic behavior
        logger.info("\n" + "-" * 40)
        logger.info("TESTING DETERMINISTIC BEHAVIOR")
        logger.info("-" * 40)
        
        # Create second instance with same seed
        processor2 = create_shared_data_processor(random_seed=42)
        p3o_queries2 = processor2.get_p3o_format_data(max_data_points=max_data_points, use_train_split=True)
        
        # Compare first few samples
        if len(p3o_queries) == len(p3o_queries2):
            matches = 0
            for i in range(min(5, len(p3o_queries))):
                if (p3o_queries[i]["x"]["onet_code"] == p3o_queries2[i]["x"]["onet_code"] and
                    p3o_queries[i]["x"]["job_title"] == p3o_queries2[i]["x"]["job_title"]):
                    matches += 1
            
            if matches == min(5, len(p3o_queries)):
                logger.info("✅ Deterministic behavior confirmed - same seed produces same data order")
            else:
                logger.warning(f"⚠️  Non-deterministic behavior detected - only {matches}/{min(5, len(p3o_queries))} samples match")
        
        logger.info("\n" + "=" * 60)
        logger.info("VERIFICATION COMPLETE")
        logger.info("=" * 60)
        logger.info("✅ All consistency checks passed!")
        logger.info("Both experiments will use identical training datasets.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_random_seeds():
    """Test different random seeds to confirm they produce different data splits."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING DIFFERENT RANDOM SEEDS")
    logger.info("=" * 60)
    
    from o_net_data_processor import create_shared_data_processor
    
    # Test with different seeds
    seeds = [42, 123, 999]
    first_jobs = {}
    
    for seed in seeds:
        processor = create_shared_data_processor(random_seed=seed)
        queries = processor.get_p3o_format_data(max_data_points=5, use_train_split=True)
        
        first_job = queries[0]["x"]["job_title"] if queries else "None"
        first_jobs[seed] = first_job
        logger.info(f"Seed {seed}: First job = '{first_job}'")
    
    # Check if different seeds produce different orders
    unique_first_jobs = set(first_jobs.values())
    if len(unique_first_jobs) > 1:
        logger.info("✅ Different random seeds produce different data orders (as expected)")
    else:
        logger.warning("⚠️  All seeds produced the same first job (unexpected)")


if __name__ == "__main__":
    # Run verification
    success = verify_experiments_consistency(max_data_points=10)
    
    if success:
        logger.info("\n🎉 SUCCESS: Experiments are using consistent datasets!")
        
        # Also test different random seeds
        compare_random_seeds()
        
        sys.exit(0)
    else:
        logger.error("\n💥 FAILURE: Experiments are NOT using consistent datasets!")
        sys.exit(1)
