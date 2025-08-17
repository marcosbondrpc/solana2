#!/bin/bash

echo "═══════════════════════════════════════════════════════════════"
echo "     SOLANA NODE DASHBOARD - ULTRA PERFORMANCE MONITOR"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
    echo ""
fi

# Kill any existing processes on our ports
echo "Checking for existing processes..."
lsof -ti:42391 | xargs kill -9 2>/dev/null
lsof -ti:42392 | xargs kill -9 2>/dev/null

echo "Starting backend server on port 42392..."
node server/index.js &
SERVER_PID=$!

sleep 2

echo "Starting frontend on port 42391..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Dashboard is starting up..."
echo ""
echo "Frontend: http://0.0.0.0:42391"
echo "Backend:  http://0.0.0.0:42392"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo "═══════════════════════════════════════════════════════════════"

# Wait for Ctrl+C
trap "echo 'Stopping dashboard...'; kill $SERVER_PID $FRONTEND_PID 2>/dev/null; exit" INT

# Keep script running
wait