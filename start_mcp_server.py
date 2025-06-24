#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Start MCP Server for PSETAE
This script initializes and starts the MCP server for PSETAE.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from server.config import SERVER_PORT, SERVER_HOST

def main():
    """Main function to start the MCP server"""
    parser = argparse.ArgumentParser(description='Start the MCP server for PSETAE')
    parser.add_argument('--port', type=int, default=SERVER_PORT, help='Port to run the server on')
    parser.add_argument('--host', type=str, default=SERVER_HOST, help='Host to run the server on')
    parser.add_argument('--setup-workflow', action='store_true', help='Set up the complete PSETAE workflow')
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.join(Path(__file__).resolve().parent, "logs"), exist_ok=True)
    
    # Start the MCP server in a separate process
    server_process = subprocess.Popen(
        [sys.executable, os.path.join("server", "server.py"), 
         "--port", str(args.port), 
         "--host", args.host],
        cwd=Path(__file__).resolve().parent
    )
    
    print(f"Starting MCP server on {args.host}:{args.port}...")
    time.sleep(5)  # Wait for server to start
    
    # Get the admin token
    try:
        import requests
        response = requests.get(f"http://{args.host}:{args.port}/admin/token")
        if response.status_code == 200:
            admin_token = response.json()["admin_token"]
            print(f"Admin token retrieved: {admin_token}")
            
            # Start the Admin Agent
            admin_cmd = [
                sys.executable, 
                os.path.join("agents", "admin_agent.py"),
                "--token", admin_token,
                "--server-url", f"http://{args.host}:{args.port}"
            ]
            
            if args.setup_workflow:
                admin_cmd.append("--setup-workflow")
            
            admin_process = subprocess.Popen(
                admin_cmd,
                cwd=Path(__file__).resolve().parent
            )
            
            print("Admin Agent started!")
            
            # Keep the server running
            try:
                server_process.wait()
            except KeyboardInterrupt:
                print("\nShutting down MCP server...")
                server_process.terminate()
                if 'admin_process' in locals():
                    admin_process.terminate()
        else:
            print(f"Failed to retrieve admin token: {response.text}")
            server_process.terminate()
    except Exception as e:
        print(f"Error starting Admin Agent: {str(e)}")
        server_process.terminate()

if __name__ == "__main__":
    main()
