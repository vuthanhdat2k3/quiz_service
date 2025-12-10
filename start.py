#!/usr/bin/env python3
"""
Start script for Railway deployment
Handles PORT environment variable properly
"""
import os
import sys

if __name__ == "__main__":
    port = os.getenv("PORT", "8000")
    host = os.getenv("HOST", "0.0.0.0")
    
    # Import uvicorn
    import uvicorn
    
    # Run the app
    uvicorn.run(
        "app.main:app",
        host=host,
        port=int(port),
        log_level="info"
    )
