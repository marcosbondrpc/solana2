#!/usr/bin/env python3
"""
Main entry point for the MEV Detection System
DETECTION-ONLY: No execution or trading functionality
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def main():
    """Main entry point for the MEV detection system"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                  MEV DETECTION SYSTEM v2.0                      ║
║                    DETECTION-ONLY MODE                          ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print("Starting services...")
    
    # Check if running detection API
    if "--detector" in sys.argv or "--all" in sys.argv:
        print("✓ Starting Detection API on port 8000...")
        os.system("cd services/detector && python app.py &")
    
    # Check if running integration bridge
    if "--bridge" in sys.argv or "--all" in sys.argv:
        print("✓ Starting Integration Bridge on port 4000...")
        os.system("node integration-bridge.js &")
    
    # Check if running frontend
    if "--frontend" in sys.argv or "--all" in sys.argv:
        print("✓ Starting Frontend Dashboard on port 4001...")
        os.system("cd frontend2 && npm run dev &")
    
    if len(sys.argv) == 1:
        print("""
Usage: python start.py [options]

Options:
  --all        Start all services
  --detector   Start detection API only
  --bridge     Start integration bridge only  
  --frontend   Start frontend dashboard only

API Endpoints:
  Detection API:  http://localhost:8000
  Bridge API:     http://localhost:4000
  Dashboard:      http://localhost:4001
  WebSocket:      ws://localhost:4000/ws

Documentation:
  See docs/ folder for detailed documentation
  CLAUDE.md for AI development guide
  README.md for system overview
        """)
    else:
        print("\nServices started successfully!")
        print("\nAccess points:")
        print("  Detection API:  http://localhost:8000/docs")
        print("  Dashboard:      http://localhost:4001")
        print("  WebSocket:      ws://localhost:4000/ws")
        
        # Keep the script running
        try:
            await asyncio.sleep(float('inf'))
        except KeyboardInterrupt:
            print("\n\nShutting down services...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")