#!/usr/bin/env python3
"""
Create a single Sentinel-1 data extraction task for TRAINING split only.
This script creates one task at a time to avoid cross-task interference.
"""

import os
import sys
import json
import requests
import argparse
from pathlib import Path

# Add the project root to Python path
BASE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(BASE_DIR))

def create_training_extraction_task(start_date="2024-09-01", end_date="2025-03-31"):
    """Create a single extraction task for the TRAINING split."""
    
    # Server configuration
    server_url = "http://localhost:8080"
    admin_token = "c2b46016-7893-451a-b76d-c35f495bb17b"
    
    # GeoJSON and output configuration for TRAINING split
    geojson_file = "training_new_wgs84.geojson"
    geojson_path = os.path.join(BASE_DIR, "output", "geojson", geojson_file)
    output_dir = os.path.join(BASE_DIR, "output", "sentinel1_data", "traindirectory")
    
    # Verify GeoJSON file exists
    if not os.path.exists(geojson_path):
        print(f"ERROR: GeoJSON file not found: {geojson_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read tile coverage information
    coverage_file = os.path.join(BASE_DIR, "output", "tile_coverage", "sentinel1_coverage.txt")
    tile_info = {"tracks": ["63"], "acquisitions": {"63": 16}}  # Default values
    
    if os.path.exists(coverage_file):
        try:
            with open(coverage_file, 'r') as f:
                content = f.read()
                # Parse tile information from coverage file
                tracks = []
                acquisitions = {}
                for line in content.split('\n'):
                    if 'Track' in line and 'acquisitions' in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            track = parts[1].rstrip(':')
                            acq_count = int(parts[2])
                            tracks.append(track)
                            acquisitions[track] = acq_count
                
                if tracks:
                    tile_info = {"tracks": tracks, "acquisitions": acquisitions}
                    print(f"Found tile information: {len(tracks)} tracks")
                    for track, count in acquisitions.items():
                        print(f"  Track {track}: {count} acquisitions")
        except Exception as e:
            print(f"Warning: Could not read tile coverage file: {e}")
    
    # Task metadata
    task_metadata = {
        "geojson_path": geojson_path,
        "output_dir": output_dir,
        "start_date": start_date,
        "end_date": end_date,
        "bands": ["VV", "VH"],
        "task_type": "data_extraction",
        "tile_information": tile_info,
        "coverage_file": coverage_file,
        "split": "training"  # Explicitly mark this as training split
    }
    
    # Task creation payload
    task_data = {
        "title": f"Sentinel-1 TRAINING Data Extraction {start_date} to {end_date}",
        "description": f"Extract Sentinel-1 data for TRAINING split from {start_date} to {end_date} using {geojson_file}",
        "assigned_to": "sentinel1-data-extraction-agent",
        "priority": "high",
        "metadata": task_metadata
    }
    
    print(f"Creating TRAINING extraction task...")
    print(f"Using GeoJSON: {geojson_path}")
    print(f"Using output directory: {output_dir}")
    print(f"Dataset type: traindirectory (TRAINING split)")
    print(f"Output directory: {output_dir}")
    
    try:
        # Create the task
        response = requests.post(
            f"{server_url}/api/tasks/create",
            params={"token": admin_token},
            json=task_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            task_info = response.json()
            task_id = task_info.get("id")
            print(f"Task created successfully with ID: {task_id}")
            print(f"Task will extract Sentinel-1 data from {start_date} to {end_date}")
            print(f"Results will be saved to: {output_dir}")
            return True
        else:
            print(f"Failed to create task: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error creating task: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Create Sentinel-1 TRAINING extraction task")
    parser.add_argument("--start-date", default="2024-09-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-03-31", help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    success = create_training_extraction_task(args.start_date, args.end_date)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
