#!/bin/bash
# FED-MED Gradio App Launcher

echo "=========================================="
echo "🚀 FED-MED Gradio Application Launcher"
echo "=========================================="
echo ""

# Check if gradio is installed
if ! python -c "import gradio" 2>/dev/null; then
    echo "📦 Installing Gradio..."
    pip install -q gradio==4.12.0
fi

# Ensure shared state directory exists
mkdir -p /workspace/saumilya/vasu/FED-MED/static

# Launch app
echo "🎯 Launching FED-MED Gradio UI..."
echo ""
echo "Features:"
echo "  ✅ Interactive Medical Q&A"
echo "  ✅ Live Federated Learning Monitoring"
echo "  ✅ Animated Architecture Visualization"
echo "  ✅ Agent Decision Transparency"
echo ""
echo "=========================================="
echo ""

cd /workspace/saumilya/vasu/FED-MED
python gradio_app.py
