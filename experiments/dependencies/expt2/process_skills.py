#!/usr/bin/env python3
"""
Script to process skill dimensions JSON file and create rows for primary skills and synonyms.
Also provides analysis of unique values and duplicate counts in skill_name column.

Creates multiple rows:
- First row: skill_name = primary_skill_name (the key from JSON)
- Additional rows: skill_name = synonym, primary_skill_name = original key

Analysis features:
- Find unique values in skill_name column
- Show duplicate counts and most common skill names
- Generate detailed analysis report

Deduplication features:
- Smart duplicate removal with primary skill preference
- If skill_name appears as both primary and synonym: keep primary
- If skill_name only appears as synonyms: keep first occurrence
- Overwrite original file with deduplicated data
- Automatic analysis integration

Usage:
python expt2/process_skills.py "/Users/ferozahmad/hackspace/prompt-opt/expt2_data/skill_dimensions_updated/updated_skill_dimensions.json" --analyze --deduplicate

Output will be saved to: /Users/ferozahmad/hackspace/prompt-opt/expt2_data/skill_dimensions_updated/all_skills.csv


"""

import json
import csv
import sys
from pathlib import Path
from collections import Counter

output_file_path = "/Users/ferozahmad/hackspace/prompt-opt/expt2_data/skill_dimensions_updated/all_skills.csv"


def process_skills_json(input_file_path, output_file_path=None, suppress_stdout=False):
    """
    Process the skills JSON file and create structured output.
    
    Args:
        input_file_path (str): Path to the input JSON file
        output_file_path (str, optional): Path for output CSV file. If None, prints to stdout unless suppressed.
        suppress_stdout (bool, optional): If True, don't print CSV to stdout when no output file is specified.
    """
    try:
        # Read the JSON file
        with open(input_file_path, 'r', encoding='utf-8') as f:
            skills_data = json.load(f)
        
        # Prepare output data
        processed_rows = []
        
        # Process each skill
        for primary_skill_name, skill_info in skills_data.items():
            description = skill_info.get('description', '')
            confidence = skill_info.get('confidence', 0.0)
            usage_count = skill_info.get('usage_count', 0)
            last_updated = skill_info.get('last_updated', '')
            synonyms = skill_info.get('synonyms', [])
            
            # First row: skill_name = primary_skill_name
            processed_rows.append({
                'primary_skill_name': primary_skill_name.lower(),
                'skill_name': primary_skill_name.lower(),
                'description': description,
                'confidence': confidence,
                'usage_count': usage_count,
                'last_updated': last_updated,
                'is_primary': True
            })
            
            # Additional rows for each synonym
            for synonym in synonyms:
                processed_rows.append({
                    'primary_skill_name': primary_skill_name.lower(),
                    'skill_name': synonym.lower(),
                    'description': description,
                    'confidence': confidence,
                    'usage_count': usage_count,
                    'last_updated': last_updated,
                    'is_primary': False
                })
        
        # Output the data
        if output_file_path:
            # Write to CSV file
            fieldnames = ['primary_skill_name', 'skill_name', 'description', 'confidence', 
                         'usage_count', 'last_updated', 'is_primary']
            
            with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(processed_rows)
            
            print(f"Successfully processed {len(skills_data)} skills into {len(processed_rows)} rows.")
            print(f"Output saved to: {output_file_path}")
        elif not suppress_stdout:
            # Print to stdout only if not suppressed
            fieldnames = ['primary_skill_name', 'skill_name', 'description', 'confidence', 
                         'usage_count', 'last_updated', 'is_primary']
            
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(processed_rows)
        else:
            # Data processed but no output (for analysis only)
            print(f"Successfully processed {len(skills_data)} skills into {len(processed_rows)} rows.")
        
        return processed_rows
        
    except FileNotFoundError:
        print(f"Error: File not found: {input_file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None


def analyze_skill_names(processed_rows):
    """
    Analyze unique values in skill_name column and show duplicate counts.
    
    Args:
        processed_rows (list): List of processed skill rows
    
    Returns:
        dict: Analysis results containing unique counts and duplicates
    """
    if not processed_rows:
        return None
    
    # Extract all skill names
    skill_names = [row['skill_name'] for row in processed_rows]
    
    # Count occurrences
    skill_name_counts = Counter(skill_names)
    
    # Find unique and duplicate skill names
    unique_skills = list(skill_name_counts.keys())
    duplicates = {name: count for name, count in skill_name_counts.items() if count > 1}
    
    # Analysis results
    analysis = {
        'total_rows': len(processed_rows),
        'unique_skill_names': len(unique_skills),
        'total_skill_name_occurrences': len(skill_names),
        'duplicate_skill_names': len(duplicates),
        'duplicates_detail': duplicates,
        'skill_name_counts': skill_name_counts
    }
    
    return analysis


def print_analysis_report(analysis):
    """
    Print a formatted analysis report.
    
    Args:
        analysis (dict): Analysis results from analyze_skill_names
    """
    if not analysis:
        print("No analysis data available.")
        return
    
    print("\n" + "="*80)
    print("SKILL NAME ANALYSIS REPORT")
    print("="*80)
    
    print(f"Total rows processed: {analysis['total_rows']:,}")
    print(f"Unique skill names: {analysis['unique_skill_names']:,}")
    print(f"Total skill name occurrences: {analysis['total_skill_name_occurrences']:,}")
    print(f"Skill names with duplicates: {analysis['duplicate_skill_names']:,}")
    
    if analysis['duplicates_detail']:
        print(f"\nDUPLICATE SKILL NAMES (appears more than once):")
        print("-" * 60)
        
        # Sort duplicates by count (descending)
        sorted_duplicates = sorted(analysis['duplicates_detail'].items(), 
                                 key=lambda x: x[1], reverse=True)
        
        for skill_name, count in sorted_duplicates[:20]:  # Show top 20
            print(f"{skill_name:<50} | {count:>5} times")
        
        if len(sorted_duplicates) > 20:
            print(f"... and {len(sorted_duplicates) - 20} more duplicates")
    else:
        print("\nNo duplicate skill names found - all skill names are unique!")
    
    # Show most common skill names
    print(f"\nTOP 10 MOST COMMON SKILL NAMES:")
    print("-" * 60)
    most_common = analysis['skill_name_counts'].most_common(10)
    for skill_name, count in most_common:
        print(f"{skill_name:<50} | {count:>5} times")
    
    print("="*80)


def deduplicate_skill_names(processed_rows, analysis):
    """
    Remove duplicate skill names using smart logic:
    - If a skill_name has both primary and synonym entries, keep the primary
    - If a skill_name only has synonym entries, keep the first occurrence
    
    Args:
        processed_rows (list): List of processed skill rows
        analysis (dict): Analysis results containing duplicate information
    
    Returns:
        list: Deduplicated rows with intelligent duplicate resolution
    """
    if not processed_rows or not analysis:
        return processed_rows
    
    # Step 1: Group rows by skill_name
    skill_groups = {}
    for i, row in enumerate(processed_rows):
        skill_name = row['skill_name']
        if skill_name not in skill_groups:
            skill_groups[skill_name] = []
        skill_groups[skill_name].append((i, row))  # Store original index for debugging
    
    # Step 2: Apply smart selection logic
    deduplicated_rows = []
    primary_preferred = 0
    synonym_first_kept = 0
    total_duplicates_removed = 0
    
    for skill_name, group in skill_groups.items():
        if len(group) == 1:
            # No duplicates - keep the single row
            deduplicated_rows.append(group[0][1])
        else:
            # Multiple rows for this skill_name - apply smart logic
            total_duplicates_removed += len(group) - 1
            
            # Look for primary skill (is_primary=True)
            primary_rows = [row_data for idx, row_data in group if row_data['is_primary']]
            synonym_rows = [row_data for idx, row_data in group if not row_data['is_primary']]
            
            if primary_rows:
                # Case A: Primary exists - keep the primary, discard synonyms
                if len(primary_rows) > 1:
                    # Multiple primaries (shouldn't happen, but handle gracefully)
                    deduplicated_rows.append(primary_rows[0])
                    print(f"⚠️  Warning: Multiple primary entries for '{skill_name}' - kept first primary")
                else:
                    deduplicated_rows.append(primary_rows[0])
                primary_preferred += 1
            else:
                # Case B: Only synonyms - keep first occurrence
                deduplicated_rows.append(group[0][1])  # First in original order
                synonym_first_kept += 1
    
    # Sort deduplicated rows by their original order to maintain consistency
    original_indices = {}
    for i, row in enumerate(processed_rows):
        row_id = id(row)  # Use object id as unique identifier
        original_indices[row_id] = i
    
    # Sort deduplicated rows by their original position
    deduplicated_rows.sort(key=lambda row: original_indices.get(id(row), float('inf')))
    
    # Print detailed results
    print(f"\nSMART DEDUPLICATION RESULTS:")
    print("-" * 50)
    print(f"Original rows: {len(processed_rows):,}")
    print(f"Deduplicated rows: {len(deduplicated_rows):,}")
    print(f"Total duplicates removed: {total_duplicates_removed:,}")
    print(f"Unique skill names: {len(skill_groups):,}")
    print()
    print("Selection Strategy:")
    print(f"  • Primary skills preferred: {primary_preferred:,}")
    print(f"  • First synonym kept: {synonym_first_kept:,}")
    print(f"  • Unique skills (no duplicates): {len(skill_groups) - primary_preferred - synonym_first_kept:,}")
    
    return deduplicated_rows



def main():
    """Main function to handle command line execution."""
    if len(sys.argv) < 2:
        print("Usage: python process_skills.py <input_json_file> [--analyze] [--deduplicate]")
        print("Example: python process_skills.py updated_skill_dimensions.json")
        print("Example with analysis: python process_skills.py updated_skill_dimensions.json --analyze")
        print("Example with deduplication: python process_skills.py updated_skill_dimensions.json --analyze --deduplicate")
        print(f"Output file: {output_file_path}")
        sys.exit(1)
    
    # Parse command line arguments
    args = sys.argv[1:]
    analyze_flag = '--analyze' in args
    deduplicate_flag = '--deduplicate' in args
    
    # Remove flags from args
    if analyze_flag:
        args.remove('--analyze')
    if deduplicate_flag:
        args.remove('--deduplicate')
    
    input_file = args[0]
    
    # Ensure input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    # Process the file
    # Always suppress stdout since we're using a fixed output file
    suppress_output = True
    result = process_skills_json(input_file, output_file_path, suppress_stdout=suppress_output)
    
    if result is None:
        sys.exit(1)
    
    # Run analysis if requested
    analysis = None
    if analyze_flag:
        print("\nRunning skill name analysis...")
        analysis = analyze_skill_names(result)
        if analysis:
            print_analysis_report(analysis)
        else:
            print("Failed to analyze skill names.")
    
    # Run deduplication if requested
    if deduplicate_flag:
        if not analyze_flag:
            # Need analysis for deduplication, so run it if not already done
            print("\nRunning analysis for deduplication...")
            analysis = analyze_skill_names(result)
        
        if analysis:
            print("\nPerforming deduplication...")
            deduplicated_rows = deduplicate_skill_names(result, analysis)
            
            if deduplicated_rows:
                # Overwrite the same file with deduplicated data
                fieldnames = ['primary_skill_name', 'skill_name', 'description', 'confidence', 
                             'usage_count', 'last_updated', 'is_primary']
                
                try:
                    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(deduplicated_rows)
                    
                    print(f"\n✅ Deduplication complete! Original file overwritten: {output_file_path}")
                except Exception as e:
                    print(f"\n❌ Failed to overwrite file with deduplicated data: {e}")
            else:
                print("\n❌ Deduplication failed.")
        else:
            print("\n❌ Cannot deduplicate without analysis data.")


if __name__ == "__main__":
    main()
