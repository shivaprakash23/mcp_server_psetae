#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create a Sentinel-1 data extraction task in the MCP server
Allows specifying date ranges via command-line arguments for flexibility
"""

import os
import json
import requests
import argparse
from datetime import datetime

# Server configuration
SERVER_URL = "http://localhost:8080"
ADMIN_TOKEN = "4c4e2130-6bd1-4c76-87b8-83da2b9a2504"  # Updated admin token

# Default paths
DEFAULT_GEOJSON_PATH = "D:\\Semester4\\ProjectVijayapur\\psetae\\psetae_all_github\\mcp_server_psetae\\output\\geojson\\croptype_KA28_wgs84_test_622.geojson"
DEFAULT_OUTPUT_DIR = "D:\\Semester4\\ProjectVijayapur\\psetae\\psetae_all_github\\mcp_server_psetae\\output\\sentinel1_data"
DEFAULT_COVERAGE_FILE = "D:\\Semester4\\ProjectVijayapur\\psetae\\psetae_all_github\\mcp_server_psetae\\output\\tile_coverage\\sentinel1_coverage.txt"

# Function to parse tile information from coverage file
def parse_tile_information(coverage_file):
    """Parse tile information from the coverage output file
    
    Args:
        coverage_file (str): Path to the coverage output file
        
    Returns:
        dict: Dictionary containing tile information
    """
    tile_info = {
        "tracks": [],
        "acquisitions": {}
    }
    
    try:
        if not os.path.exists(coverage_file):
            print(f"Warning: Coverage file {coverage_file} not found")
            return tile_info
        
        with open(coverage_file, 'r') as f:
            lines = f.readlines()
        
        # Parse track information
        for line in lines:
            if "Track" in line and "acquisitions" in line:
                # Extract track number and acquisition count
                parts = line.strip().split()
                track_num = parts[1].strip(':')
                acquisitions = int(parts[2])
                tile_info["tracks"].append(track_num)
                tile_info["acquisitions"][track_num] = acquisitions
        
        print(f"Found tile information: {len(tile_info['tracks'])} tracks")
        for track in tile_info["tracks"]:
            print(f"  Track {track}: {tile_info['acquisitions'][track]} acquisitions")
        
        return tile_info
    except Exception as e:
        print(f"Error parsing tile information: {str(e)}")
        return tile_info

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Create a Sentinel-1 data extraction task')
    parser.add_argument('--geojson', type=str, default=DEFAULT_GEOJSON_PATH,
                        help='Path to GeoJSON file')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for extracted data')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('--start-month', type=int, default=None,
                        help='Start month (1-12)')
    parser.add_argument('--start-year', type=int, default=None,
                        help='Start year (e.g., 2024)')
    parser.add_argument('--end-month', type=int, default=None,
                        help='End month (1-12)')
    parser.add_argument('--end-year', type=int, default=None,
                        help='End year (e.g., 2025)')
    parser.add_argument('--coverage-task-id', type=str, default=None,
                        help='ID of the coverage task to link with')
    parser.add_argument('--coverage-file', type=str, default=DEFAULT_COVERAGE_FILE,
                        help='Path to the coverage output file')
    parser.add_argument('--title-suffix', type=str, default='',
                        help='Optional suffix to add to the task title')
    return parser.parse_args()

# Create task
def create_task():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine date range
    start_date = args.start_date
    end_date = args.end_date
    
    # If dates not provided directly, construct from month/year
    if not start_date and args.start_month and args.start_year:
        start_date = f"{args.start_year}-{args.start_month:02d}-01"
    
    if not end_date and args.end_month and args.end_year:
        # Get last day of month
        if args.end_month == 12:
            next_month_year = args.end_year + 1
            next_month = 1
        else:
            next_month_year = args.end_year
            next_month = args.end_month + 1
        
        last_day = (datetime(next_month_year, next_month, 1).replace(day=1) - 
                    datetime.timedelta(days=1)).day
        end_date = f"{args.end_year}-{args.end_month:02d}-{last_day}"
    
    # Default dates if nothing provided
    if not start_date:
        start_date = "2024-09-01"  # Default to Sept 2024
    
    if not end_date:
        end_date = "2025-03-31"  # Default to March 2025
    
    # Determine dataset type (test, train, validation) based on geojson filename
    geojson_filename = os.path.basename(args.geojson).lower()
    if "test" in geojson_filename:
        dataset_type = "testdirectory"
    elif "train" in geojson_filename:
        dataset_type = "traindirectory"
    elif "valid" in geojson_filename or "val" in geojson_filename:
        dataset_type = "validationdirectory"
    else:
        # Default to test if can't determine
        dataset_type = "testdirectory"
        print(f"Warning: Could not determine dataset type from filename '{geojson_filename}'. Using '{dataset_type}' as default.")
    
    # Create dataset-specific output directory
    specific_output_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(specific_output_dir, exist_ok=True)
    
    print(f"Dataset type: {dataset_type} (determined from GeoJSON filename)")
    print(f"Output directory: {specific_output_dir}")
    
    # Get tile information from coverage file
    tile_info = parse_tile_information(args.coverage_file)
    
    # Task metadata
    metadata = {
        "geojson_path": args.geojson,
        "output_dir": specific_output_dir,
        "start_date": start_date,
        "end_date": end_date,
        "bands": ["VV", "VH"],
        "task_type": "data_extraction",
        "tile_information": tile_info  # Include tile information from coverage analysis
    }
    
    # Add coverage task ID if provided
    if args.coverage_task_id:
        metadata["coverage_task_id"] = args.coverage_task_id
        
    # Add coverage file path
    metadata["coverage_file"] = args.coverage_file
    
    # Create title with date range
    title_suffix = f" {args.title_suffix}" if args.title_suffix else ""
    title = f"Sentinel-1 Data Extraction {start_date} to {end_date}{title_suffix}"
    
    url = f"{SERVER_URL}/api/tasks/create"
    # Add token as query parameter
    params = {"token": ADMIN_TOKEN}
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "title": title,
        "description": f"Extract Sentinel-1 data from {start_date} to {end_date}",
        "assigned_to": "sentinel1-data-extraction-agent",
        "priority": 1,
        "metadata": metadata  # Pass metadata as a dictionary
    }
    
    response = requests.post(url, params=params, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        task_id = result.get("task_id") or result.get("id")  # Handle both possible field names
        print(f"Task created successfully with ID: {task_id}")
        print(f"Task will extract Sentinel-1 data from {metadata['start_date']} to {metadata['end_date']}")
        print(f"Results will be saved to: {metadata['output_dir']}")
        return task_id
    else:
        print(f"Failed to create task. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return None

if __name__ == "__main__":
    create_task()
