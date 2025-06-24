#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCP Server for PSETAE
This script initializes and runs the Model Context Protocol (MCP) server for PSETAE.
The server manages communication between agents and maintains the project state.
"""

import os
import sys
import argparse
import logging
import sqlite3
import uuid
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import *

# Initialize logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode='a') if os.path.exists(os.path.dirname(LOG_FILE)) else logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp_server")

# Initialize FastAPI app
app = FastAPI(title="MCP Server for PSETAE", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class Token(BaseModel):
    token: str
    agent_id: str
    agent_type: str
    created_at: datetime
    expires_at: datetime

class AgentRequest(BaseModel):
    agent_id: str
    agent_type: str
    token: str

class TaskRequest(BaseModel):
    title: str
    description: str
    assigned_to: str
    priority: int = 1
    metadata: Optional[Dict[str, Any]] = None

class MemoryEntry(BaseModel):
    agent_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

# Database initialization
def init_db(db_path):
    """Initialize the SQLite database with required tables"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tokens table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tokens (
        token TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        agent_type TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL,
        expires_at TIMESTAMP NOT NULL
    )
    ''')
    
    # Create tasks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        assigned_to TEXT NOT NULL,
        status TEXT NOT NULL,
        priority INTEGER NOT NULL,
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP NOT NULL,
        metadata TEXT
    )
    ''')
    
    # Create memory table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS memory (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL,
        metadata TEXT
    )
    ''')
    
    # Create project_context table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS project_context (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP NOT NULL
    )
    ''')
    
    # Insert admin token if it doesn't exist
    admin_token = str(uuid.uuid4())
    cursor.execute('''
    INSERT OR IGNORE INTO tokens (token, agent_id, agent_type, created_at, expires_at)
    VALUES (?, ?, ?, ?, ?)
    ''', (
        admin_token,
        ADMIN_AGENT_ID,
        "admin",
        datetime.now().isoformat(),
        (datetime.now() + timedelta(days=TOKEN_EXPIRY_DAYS)).isoformat()
    ))
    
    # Store admin token in project_context
    cursor.execute('''
    INSERT OR REPLACE INTO project_context (key, value, created_at, updated_at)
    VALUES (?, ?, ?, ?)
    ''', (
        "admin_token",
        admin_token,
        datetime.now().isoformat(),
        datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database initialized at {db_path}")
    logger.info(f"Admin token: {admin_token}")
    return admin_token

# Token validation
def validate_token(token: str, db_path: str):
    """Validate a token against the database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT agent_id, agent_type, expires_at FROM tokens
    WHERE token = ?
    ''', (token,))
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return None
    
    agent_id, agent_type, expires_at = result
    if datetime.fromisoformat(expires_at) < datetime.now():
        return None
    
    return {"agent_id": agent_id, "agent_type": agent_type}

# API routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "MCP Server for PSETAE is running"}

@app.get("/status")
async def status():
    """Server status endpoint"""
    return {
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0"
    }

@app.post("/api/validate-token")
async def validate_token_api(request: AgentRequest):
    """Validate a token and return agent information"""
    agent_info = validate_token(request.token, DB_PATH)
    if not agent_info:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return {
        "valid": True,
        "agent_id": agent_info["agent_id"],
        "agent_type": agent_info["agent_type"]
    }

@app.post("/api/create-worker")
async def create_worker(request: AgentRequest):
    """Create a new worker agent (admin only)"""
    agent_info = validate_token(request.token, DB_PATH)
    if not agent_info or agent_info["agent_type"] != "admin":
        raise HTTPException(status_code=401, detail="Unauthorized: Admin token required")
    
    if request.agent_type not in AGENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid agent type. Must be one of: {', '.join(AGENT_TYPES.keys())}")
    
    # Generate worker token
    worker_token = str(uuid.uuid4())
    
    # Store in database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO tokens (token, agent_id, agent_type, created_at, expires_at)
    VALUES (?, ?, ?, ?, ?)
    ''', (
        worker_token,
        request.agent_id,
        request.agent_type,
        datetime.now().isoformat(),
        (datetime.now() + timedelta(days=TOKEN_EXPIRY_DAYS)).isoformat()
    ))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Created worker agent: {request.agent_id} of type {request.agent_type}")
    
    return {
        "token": worker_token,
        "agent_id": request.agent_id,
        "agent_type": request.agent_type,
        "created_at": datetime.now().isoformat()
    }

@app.post("/api/tasks/create")
async def create_task(request: TaskRequest, token: str):
    """Create a new task"""
    agent_info = validate_token(token, DB_PATH)
    if not agent_info:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    task_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO tasks (id, title, description, assigned_to, status, priority, created_at, updated_at, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        task_id,
        request.title,
        request.description,
        request.assigned_to,
        "pending",
        request.priority,
        now,
        now,
        json.dumps(request.metadata) if request.metadata else None
    ))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Created task: {request.title} assigned to {request.assigned_to}")
    
    return {
        "task_id": task_id,
        "title": request.title,
        "assigned_to": request.assigned_to,
        "status": "pending",
        "created_at": now
    }

@app.get("/api/tasks/{agent_id}")
async def get_tasks(agent_id: str, token: str):
    """Get tasks for an agent"""
    agent_info = validate_token(token, DB_PATH)
    if not agent_info:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    if agent_info["agent_id"] != agent_id and agent_info["agent_type"] != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: Can only access own tasks or admin required")
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT * FROM tasks WHERE assigned_to = ? ORDER BY priority DESC, created_at ASC
    ''', (agent_id,))
    
    tasks = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return {"tasks": tasks}

@app.post("/api/memory/add")
async def add_memory(entry: MemoryEntry, token: str):
    """Add a memory entry"""
    agent_info = validate_token(token, DB_PATH)
    if not agent_info:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    if agent_info["agent_id"] != entry.agent_id and agent_info["agent_type"] != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: Can only add own memories or admin required")
    
    memory_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO memory (id, agent_id, content, created_at, metadata)
    VALUES (?, ?, ?, ?, ?)
    ''', (
        memory_id,
        entry.agent_id,
        entry.content,
        now,
        json.dumps(entry.metadata) if entry.metadata else None
    ))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Added memory for agent {entry.agent_id}")
    
    return {
        "memory_id": memory_id,
        "agent_id": entry.agent_id,
        "created_at": now
    }

@app.get("/api/memory/query")
async def query_memory(agent_id: str, query: str, token: str):
    """Query memory entries"""
    agent_info = validate_token(token, DB_PATH)
    if not agent_info:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Simple keyword search (in a real implementation, use a vector database or better search)
    cursor.execute('''
    SELECT * FROM memory 
    WHERE agent_id = ? AND content LIKE ?
    ORDER BY created_at DESC
    LIMIT 10
    ''', (agent_id, f"%{query}%"))
    
    memories = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return {"memories": memories}

@app.get("/admin/token")
async def get_admin_token():
    """Get the admin token (only for local development)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT value FROM project_context WHERE key = 'admin_token'
    ''')
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        raise HTTPException(status_code=404, detail="Admin token not found")
    
    return {"admin_token": result[0]}

# Main function
def main():
    """Main function to start the MCP server"""
    parser = argparse.ArgumentParser(description='Start the MCP server for PSETAE')
    parser.add_argument('--port', type=int, default=SERVER_PORT, help='Port to run the server on')
    parser.add_argument('--host', type=str, default=SERVER_HOST, help='Host to run the server on')
    parser.add_argument('--project-dir', type=str, default=str(BASE_DIR), help='Project directory')
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    
    # Initialize database
    admin_token = init_db(DB_PATH)
    
    # Print admin token
    print(f"\n{'='*50}")
    print(f"MCP Server for PSETAE initialized")
    print(f"Admin token: {admin_token}")
    print(f"Use this token to initialize the Admin Agent")
    print(f"{'='*50}\n")
    
    # Start server
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
