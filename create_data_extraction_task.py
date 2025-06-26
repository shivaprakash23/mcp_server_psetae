#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create a Sentinel-1 data extraction task in the MCP server
"""

import os
import json
import requests

# Server configuration
SERVER_URL = "http://localhost:8080"
ADMIN_TOKEN = "d29e611c-ce46-4c51-8cb4-05948b9dccdb"  # Updated admin token

# Paths
geojson_path = "D:\\Semester4\\ProjectVijayapur\\psetae\\psetae_all_github\\psetae_all5models\\2_data_extraction\\sentinel\\test_new_wgs84_test.geojson"
output_dir = "D:\\Semester4\\ProjectVijayapur\\psetae\\psetae_all_github\\mcp_server_psetae\\output\\sentinel1_data"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Task metadata
metadata = {
    "geojson_path": geojson_path,
    "output_dir": output_dir,
    "start_date": "2022-01-01",
    "end_date": "2022-12-31",
    "bands": ["VV", "VH"]
}

# Create task
def create_task():
    url = f"{SERVER_URL}/api/tasks/create"
    # Add token as query parameter
    params = {"token": ADMIN_TOKEN}
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "title": "Sentinel-1 Data Extraction Task",
        "description": f"Extract Sentinel-1 data from {metadata['start_date']} to {metadata['end_date']}",
        "assigned_to": "sentinel1-data-extraction-agent",
        "priority": 1,
        "metadata": metadata  # Pass metadata as a dictionary, not as a JSON string
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
