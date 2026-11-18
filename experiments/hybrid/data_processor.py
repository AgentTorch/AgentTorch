import json
import pandas as pd
import os
from pathlib import Path
import re

data_input_path = "./data/job_data"
data_output_path = "./data/job_data_processed"

TESTING_MODE = False
TESTING_MODE_NUM_FILES = 10

def extract_job_name(name):
    """Extract job name from the full name by splitting on ' - ' and taking the second part"""
    if ' - ' in name:
        return name.split(' - ', 1)[1]
    return name

def clean_text(text):
    """Clean text by removing 'Related occupations' suffix and extra whitespace"""
    if isinstance(text, str):
        # Remove "Related occupations" suffix
        text = re.sub(r'Related occupations$', '', text).strip()
        return text
    return text

def process_json_file(file_path):
    """Process a single JSON file and return a dictionary for DataFrame row"""
    unique_detail_fields = set()
    numeric_prefix = 'NUMERIC_'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract basic information from summary
        summary = data.get('summary', {})
        details = data.get('details', {})
        
        # Create row dictionary
        row = {
            'soc_code': summary.get('soc_code', ''),
            'job_name': extract_job_name(summary.get('name', '')),
            'url': summary.get('url', ''),
            'timestamp': summary.get('timestamp', ''),
        }
        
        # Process summary fields (lists)
        summary_fields = [
            'Tasks', 'TechnologySkills', 'WorkActivities', 'DetailedWorkActivities',
            'WorkContext', 'Skills', 'Knowledge', 'Abilities', 'Interests',
            'WorkValues', 'WorkStyles', 'RelatedOccupations', 'ProfessionalAssociations'
        ]
        
        for field in summary_fields:
            field_data = summary.get(field, [])
            if isinstance(field_data, list):
                # Join list items with '; ' separator
                row[field] = '\n'.join([clean_text(item) for item in field_data])
            else:
                row[field] = str(field_data) if field_data else ''
        
        # Process details fields with importance scores
        details_fields = ['Knowledge']
        
        for field in details_fields:
            field_data = details.get(field, [])
            if isinstance(field_data, list):
                for item in field_data:
                    if isinstance(item, dict):
                        importance = item.get('Importance', '0')
                        if importance == '0':
                            continue
                        # Get the main field name (Skill, Knowledge, Ability, Work Activity)
                        field_key = None
                        for key in item.keys():
                            if key != 'Importance':
                                field_key = key
                                break
                        
                        if field_key:
                            # Extract the main part before the dash
                            text = item.get(field_key, '')
                            details_separator = '— '
                            if details_separator in text:
                                main_part = text.split(details_separator)[0].strip()
                            else:
                                main_part = text.strip()
                            
                            # Create column name: field_mainpart_importance
                            # clean_main_part = re.sub(r'[^a-zA-Z0-9\s]', '', main_part).strip()
                            # clean_main_part = re.sub(r'\s+', '_', clean_main_part)
                            if not main_part:
                                continue
                            main_part = ''.join(word.capitalize() for word in main_part.split())
                            column_name = f"{numeric_prefix}{field.lower()}_{main_part}"
                            unique_detail_fields.add(column_name)
                            
                            # Store the full text as value
                            row[column_name] = clean_text(importance)
        
        return row, unique_detail_fields
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, set()

def process_all_json_files():
    """Process all JSON files in the input directory and create a DataFrame"""
    input_path = Path(data_input_path)
    
    if not input_path.exists():
        print(f"Input directory {data_input_path} does not exist")
        return None, set()
    
    # Get all JSON files
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {data_input_path}")
        return None, set()
    
    if TESTING_MODE:
        json_files = json_files[:TESTING_MODE_NUM_FILES]
        print(f"TESTING MODE: Processing only {len(json_files)} JSON files")
    
    # Process each file
    rows = []
    unique_detail_fields = set()
    for i, file_path in enumerate(json_files):
        print(f"Processing {i+1}/{len(json_files)}: {file_path.name}")
        row, _unique_detail_fields = process_json_file(file_path)
        if row:
            rows.append(row)
            unique_detail_fields.update(_unique_detail_fields)
    
    print(f">> Unique detail fields: {len(unique_detail_fields)}")
    if rows:
        df = pd.DataFrame(rows)
        print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df, unique_detail_fields
    else:
        print("No valid data found")
        return None, set()

def save_dataframe(df, output_path, unique_detail_fields=None):
    """Save DataFrame to CSV and pickle files"""
    if df is None:
        print("No DataFrame to save")
        return
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = output_path / "job_data_processed.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved DataFrame to {csv_path}")
    
    # Save as pickle for faster loading
    pickle_path = output_path / "job_data_processed.pkl"
    df.to_pickle(pickle_path)
    print(f"Saved DataFrame to {pickle_path}")
    
    # Save unique detail fields to text file
    if unique_detail_fields:
        fields_path = output_path / "unique_detail_fields.txt"
        with open(fields_path, 'w', encoding='utf-8') as f:
            for field in sorted(unique_detail_fields):
                f.write(f"{field}\n")
        print(f"Saved {len(unique_detail_fields)} unique detail fields to {fields_path}")
    
    # Print some statistics
    print(f"\nDataFrame Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")

if __name__ == "__main__":
    # Process all JSON files
    df, unique_detail_fields = process_all_json_files()
    
    # Save the DataFrame
    if df is not None:
        save_dataframe(df, data_output_path, unique_detail_fields)
    else:
        print("Failed to create DataFrame")