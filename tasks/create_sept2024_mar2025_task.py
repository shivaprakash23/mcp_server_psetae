import requests
import json
import os

# Server configuration
SERVER_URL = "http://localhost:8080"
ADMIN_TOKEN = "d29e611c-ce46-4c51-8cb4-05948b9dccdb"  # Updated admin token

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "tile_coverage")
os.makedirs(output_dir, exist_ok=True)

# Task metadata
geojson_path = r"D:\Semester4\ProjectVijayapur\psetae\psetae_all_github\psetae_all5models\2_data_extraction\sentinel\test_new_wgs84_test.geojson"
metadata = {
    "geojson_path": geojson_path,
    "output_dir": output_dir,
    "start_date": "2024-09-01",
    "end_date": "2025-03-31",
    "collection": "COPERNICUS/S1_GRD"
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
        "title": "Sentinel-1 Tile Coverage Analysis Sept2024-Mar2025",
        "description": f"Analyze Sentinel-1 tile coverage from {metadata['start_date']} to {metadata['end_date']}",
        "assigned_to": "sentinel1-tile-coverage-agent",
        "priority": 1,
        "metadata": metadata  # Pass metadata as a dictionary, not as a JSON string
    }
    
    response = requests.post(url, params=params, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        task_id = result.get("task_id") or result.get("id")  # Handle both possible field names
        print(f"Task created successfully with ID: {task_id}")
        print(f"Task will analyze Sentinel-1 coverage from {metadata['start_date']} to {metadata['end_date']}")
        print(f"Results will be saved to: {metadata['output_dir']}")
        return task_id
    else:
        print(f"Failed to create task. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return None

if __name__ == "__main__":
    create_task()
