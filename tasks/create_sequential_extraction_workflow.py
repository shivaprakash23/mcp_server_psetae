#!/usr/bin/env python3
"""
Create a sequential multi-split extraction workflow using the Admin Agent.
This script uses the new architectural approach to handle test, validation, and training splits sequentially.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the project root to Python path
BASE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(BASE_DIR))

from agents.admin_agent import AdminAgent

def create_sequential_workflow(start_date="2024-09-01", end_date="2025-03-31"):
    """Create a sequential extraction workflow for all splits using the Admin Agent."""
    
    # Server configuration
    server_url = "http://localhost:8080"
    admin_token = "b26b8c83-cf8f-46a2-aa66-11d182613c0b"
    
    # Initialize Admin Agent
    admin_agent = AdminAgent(admin_token, server_url)
    
    # Define GeoJSON files for each split
    geojson_files = {
        "test": os.path.join(BASE_DIR, "output", "geojson", "croptype_KA28_wgs84_test_622.geojson"),
        "validation": os.path.join(BASE_DIR, "output", "geojson", "croptype_KA28_wgs84_validation_622.geojson"),
        "training": os.path.join(BASE_DIR, "output", "geojson", "croptype_KA28_wgs84_training_622.geojson")
    }
    
    # Verify all GeoJSON files exist
    missing_files = []
    for split, geojson_path in geojson_files.items():
        if not os.path.exists(geojson_path):
            missing_files.append(f"{split}: {geojson_path}")
    
    if missing_files:
        print("ERROR: Missing GeoJSON files:")
        for missing in missing_files:
            print(f"  {missing}")
        return False
    
    # Output base directory
    output_base_dir = os.path.join(BASE_DIR, "output")
    
    print("Creating sequential extraction workflow...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output base directory: {output_base_dir}")
    print("GeoJSON files:")
    for split, path in geojson_files.items():
        print(f"  {split}: {os.path.basename(path)}")
    
    try:
        # Create the sequential extraction workflow
        workflow = admin_agent.create_sequential_extraction_workflow(
            geojson_files=geojson_files,
            output_base_dir=output_base_dir,
            start_date=start_date,
            end_date=end_date,
            bands=["VV", "VH"]
        )
        
        print(f"\nSequential extraction workflow created successfully!")
        print(f"Created {len(workflow)} extraction tasks:")
        
        for task_name, task_info in workflow.items():
            task_id = task_info.get("id")
            split = task_name.replace("_extraction_task", "")
            print(f"  {split.upper()} extraction task: {task_id}")
        
        print(f"\nWorkflow processing order:")
        print(f"1. TEST split extraction")
        print(f"2. VALIDATION split extraction") 
        print(f"3. TRAINING split extraction")
        
        print(f"\nOutput directories:")
        print(f"  TEST: {output_base_dir}/sentinel1_data/testdirectory")
        print(f"  VALIDATION: {output_base_dir}/sentinel1_data/validationdirectory")
        print(f"  TRAINING: {output_base_dir}/sentinel1_data/traindirectory")
        
        print(f"\nNext steps:")
        print(f"1. Start the Sentinel1DataExtractionAgent")
        print(f"2. The agent will process tasks sequentially in the correct order")
        print(f"3. Each split's data will be isolated to its own output directory")
        
        return True
        
    except Exception as e:
        print(f"Error creating sequential extraction workflow: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Create sequential multi-split extraction workflow")
    parser.add_argument("--start-date", default="2024-09-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-03-31", help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    success = create_sequential_workflow(args.start_date, args.end_date)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
