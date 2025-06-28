#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create a model training task for the Sentinel1ModelTrainingAgent
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import *

def create_model_training_task(token, agent_id, train_dir, val_dir, test_dir, output_dir, 
                              epochs=100, batch_size=32, learning_rate=0.0001, 
                              sensor="S1", input_dim=2, num_classes=10):
    """Create a model training task for the Sentinel1ModelTrainingAgent
    
    Args:
        token (str): Admin token for authentication
        agent_id (str): ID of the agent to assign the task to
        train_dir (str): Directory containing training data
        val_dir (str): Directory containing validation data
        test_dir (str): Directory containing test data
        output_dir (str): Directory to save model outputs
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for training
        sensor (str): Sensor type (S1 or S2)
        input_dim (int): Number of input dimensions
        num_classes (int): Number of output classes
        
    Returns:
        dict: Task information
    """
    # Create task metadata
    metadata = {
        "train_dir": train_dir,
        "val_dir": val_dir,
        "test_dir": test_dir,
        "output_dir": output_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "sensor": sensor,
        "input_dim": input_dim,
        "mlp1": [input_dim, 32, 64],  # Default MLP1 architecture
        "num_classes": num_classes
    }
    
    # Create task
    response = requests.post(
        f"http://localhost:{SERVER_PORT}/api/tasks/create",
        params={"token": token},
        json={
            "title": f"Sentinel-1 Model Training {datetime.now().strftime('%Y-%m-%d')}",
            "description": "Train a Sentinel-1 PSETAE model",
            "assigned_to": agent_id,
            "metadata": metadata  # Send as a dictionary, not a JSON string
        }
    )
    
    if response.status_code != 200:
        print(f"Failed to create task: {response.text}")
        return None
    
    task = response.json()
    print(f"Created task: {task}")
    
    return task

def main():
    """Main function to create a model training task"""
    parser = argparse.ArgumentParser(description='Create a model training task')
    parser.add_argument('--token', type=str, required=True, help='Admin token for authentication')
    parser.add_argument('--agent-id', type=str, default="sentinel1-model-training-agent", help='Agent ID')
    
    # Default paths for the PSETAE model training
    default_train_dir = os.path.join(BASE_DIR, "output", "sentinel1_data", "traindirectory")
    default_val_dir = os.path.join(BASE_DIR, "output", "sentinel1_data", "validationdirectory")
    default_test_dir = os.path.join(BASE_DIR, "output", "sentinel1_data", "testdirectory")
    default_output_dir = os.path.join(BASE_DIR, "output", "models", "sentinel1")
    
    parser.add_argument('--train-dir', type=str, default=default_train_dir, help='Directory containing training data')
    parser.add_argument('--val-dir', type=str, default=default_val_dir, help='Directory containing validation data')
    parser.add_argument('--test-dir', type=str, default=default_test_dir, help='Directory containing test data')
    parser.add_argument('--output-dir', type=str, default=default_output_dir, help='Directory to save model outputs')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('--sensor', type=str, default='S1', help='Sensor type (S1 or S2)')
    parser.add_argument('--input-dim', type=int, default=2, help='Number of input dimensions')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of output classes')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create task
    create_model_training_task(
        args.token,
        args.agent_id,
        args.train_dir,
        args.val_dir,
        args.test_dir,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.sensor,
        args.input_dim,
        args.num_classes
    )

if __name__ == "__main__":
    main()
