#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

trap 'kill $(jobs -p); exit' SIGINT SIGTERM

if [ -f "venv/bin/activate" ]; then
    echo "[Backend] Activating venv..."
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    echo "[Backend] Activating .venv..."
    source .venv/bin/activate
else
    echo "[Backend] Warning: No venv found, using system python..."
fi

echo "[Backend] Starting Server..."
python -m backend.main &
BACKEND_PID=$!

sleep 2

if [ -d "frontend" ]; then
    echo "[Frontend] Starting Client..."
    cd frontend
    npm run dev
else
    echo "[Error] Frontend directory not found!"
    kill $BACKEND_PID
    exit 1
fi

wait