#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import requests
import json
from datetime import datetime

def create_reporting_task(server_url, token, model_dir, output_dir):
    """
    Create a reporting task for the Sentinel1ReportingAgent.
    
    Args:
        server_url (str): URL of the MCP server
        token (str): Authentication token for the MCP server
        model_dir (str): Directory containing model results
        output_dir (str): Directory to save the report
    
    Returns:
        dict: Created task information
    """
    # Create task metadata
    metadata = {
        "model_dir": model_dir,
        "output_dir": output_dir
    }
    
    # Create task payload
    current_date = datetime.now().strftime("%Y-%m-%d")
    task_payload = {
        "title": f"Sentinel-1 Model Results Report {current_date}",
        "description": f"Generate a summary report for Sentinel-1 model results from {model_dir}",
        "assigned_to": "sentinel1-reporting-agent",
        "priority": 1,
        "metadata": metadata
    }
    
    # Send request to create task
    response = requests.post(
        f"{server_url}/api/tasks/create",
        params={"token": token},
        json=task_payload
    )
    
    # Check response
    if response.status_code == 201:
        task = response.json()
        print(f"Created task: {task}")
        return task
    else:
        print(f"Failed to create task: {response.text}")
        return None

def main():
    """Main function to create a reporting task."""
    parser = argparse.ArgumentParser(description='Create a reporting task for Sentinel-1 model results')
    parser.add_argument('--token', required=True, help='Authentication token for the MCP server')
    parser.add_argument('--server-url', default='http://localhost:8080', help='URL of the MCP server')
    parser.add_argument('--model-dir', required=True, help='Directory containing model results')
    parser.add_argument('--output-dir', default='', help='Directory to save the report (default: model_dir/../reports)')
    
    args = parser.parse_args()
    
    create_reporting_task(args.server_url, args.token, args.model_dir, args.output_dir)

if __name__ == '__main__':
    main()
