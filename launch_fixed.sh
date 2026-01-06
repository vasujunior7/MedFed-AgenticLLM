#!/bin/bash

echo "🚀 Launching FED-MED Gradio Application..."
echo "This will take ~30 seconds to load the model..."
echo ""

cd /workspace/saumilya/vasu/FED-MED

# Kill any existing instance
pkill -f "python gradio_app.py" 2>/dev/null
sleep 2

# Launch the app
python gradio_app.py 2>&1 | tee -a gradio_output.log

