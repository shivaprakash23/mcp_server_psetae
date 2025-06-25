#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create a test task for the Sentinel1TileCoverageAgent
"""

import os
import sys
import argparse
import requests
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('create_test_task')

def create_task(token, server_url, geojson_path, output_dir, start_date, end_date):
    """Create a test task for the Sentinel1TileCoverageAgent
    
    Args:
        token (str): Admin token for authentication
        server_url (str): URL of the MCP server
        geojson_path (str): Path to the GeoJSON file
        output_dir (str): Output directory for tile coverage results
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        dict: Created task information
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create task
        response = requests.post(
            f"{server_url}/api/tasks/create",
            params={"token": token},
            json={
                "title": "Sentinel-1 Tile Coverage Analysis Test",
                "description": f"Test task for Sentinel-1 tile coverage analysis using {os.path.basename(geojson_path)} from {start_date} to {end_date}",
                "assigned_to": "sentinel1-tile-coverage-agent",
                "priority": 1,
                "metadata": {
                    "geojson_path": geojson_path,
                    "output_dir": output_dir,
                    "start_date": start_date,
                    "end_date": end_date,
                    "collection": "COPERNICUS/S1_GRD"
                }
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to create task: {response.text}")
            raise ValueError(f"Failed to create task: {response.text}")
        
        # Print full response for debugging
        print(f"Full server response: {response.text}")
        
        task_info = response.json()
        logger.info(f"Created task assigned to sentinel1-tile-coverage-agent")
        print(f"Task info: {task_info}")
        
        # Check if 'id' exists in the response
        if 'id' not in task_info:
            logger.warning(f"Task ID not found in response, using 'task_id' if available")
            task_info['id'] = task_info.get('task_id', 'unknown')
        
        return task_info
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        raise

def main():
    """Main function to create a test task"""
    parser = argparse.ArgumentParser(description='Create a test task for the Sentinel1TileCoverageAgent')
    parser.add_argument('--token', type=str, required=True, help='Admin token for authentication')
    parser.add_argument('--server-url', type=str, default="http://localhost:8081", help='MCP server URL')
    parser.add_argument('--geojson', type=str, required=True, help='Path to the GeoJSON file')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for tile coverage results')
    parser.add_argument('--start-date', type=str, default="2022-01-01", help='Start date for data extraction (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default="2022-12-31", help='End date for data extraction (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    try:
        # Create test task
        task_info = create_task(
            args.token,
            args.server_url,
            args.geojson,
            args.output_dir,
            args.start_date,
            args.end_date
        )
        
        # Print task information
        print("\n" + "="*50)
        print("Sentinel-1 Tile Coverage Test Task Created")
        print("="*50)
        print(f"\nTask ID: {task_info['id']}")
        print(f"Title: {task_info['title']}")
        print(f"Assigned to: {task_info['assigned_to']}")
        print(f"GeoJSON: {os.path.basename(args.geojson)}")
        print(f"Date Range: {args.start_date} to {args.end_date}")
        print(f"Output Directory: {args.output_dir}")
        print("\n" + "="*50)
        
    except Exception as e:
        logger.error(f"Error creating test task: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
