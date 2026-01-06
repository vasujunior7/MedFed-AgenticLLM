# 🎨 FED-MED Gradio Application Guide

## 🚀 Quick Start

```bash
# Launch the Gradio UI
./launch_gradio.sh

# Or manually:
python gradio_app.py
```

The app will be available at:
- **Local:** http://localhost:7860
- **Public:** Gradio will generate a public sharable link

---

## 📋 What You Get

### ✅ **Complete Interactive UI with:**

1. **💬 Medical Q&A Chat**
   - Ask medical questions in natural language
   - Get responses from fine-tuned federated model
   - Automatic medical safety disclaimers
   - Fast responses (7 sec) - model loaded once!

2. **📊 Live Monitoring Panel**
   - Current federated round status
   - Hospital metrics (samples, loss, weights)
   - Agent aggregation weights (real-time)
   - Client classifications (trusted/unstable/penalized)
   - Query counter

3. **🏗️ Animated Architecture Visualization**
   - D3.js live diagram showing data flows
   - Hospital nodes → Agent → Server
   - Animated training flows
   - Weight badges on hospitals
   - Round counter on server
   - Auto-updates every 15 seconds

4. **📚 Explainability Sections**
   - How federated learning works
   - How the agent makes decisions
   - How privacy is preserved
   - Performance metrics

---

## 🎯 Features

### **Interactive Inference (Fast!)**

The model is loaded **ONCE** at startup and stays in memory:
- First load: ~60 seconds
- Subsequent queries: ~7 seconds each
- **9x faster** than reloading every time!

```python
# Under the hood
inference_engine = InteractiveInferenceEngine(hospital="hospital_B", gpu=3)
# ↑ Loads model once at startup

response = inference_engine.query("What is diabetes?")
# ↑ Fast! Uses already-loaded model
```

### **Real-Time Monitoring**

Uses your **existing training results** from:
- `output-models/federated/metrics/training_history.json`
- Past federated training rounds (Rounds 1-3)
- Agent weights: Hospital A (23%), B (55%), C (23%)

**If you run new training:**
- UI will automatically update (polls shared state every 3 sec)
- New rounds will appear
- Weights will update
- Architecture diagram animates

### **Live Architecture Visualization**

D3.js SVG diagram that:
- Shows 3 hospitals (blue), agent (red), server (green)
- Displays current weights as badges
- Animates data flows during training
- Highlights active nodes
- Updates from shared state

---

## 📊 UI Layout

```
┌──────────────────────────────────────────────────────────────┐
│  🏥 FED-MED: Federated Medical AI                            │
├────────────────────────────┬─────────────────────────────────┤
│ 💬 CHAT (60%)              │ 📊 MONITORING (40%)             │
│                            │                                 │
│ User: What is diabetes?    │ Round: 3/3 (COMPLETE)          │
│ Bot: Diabetes is...        │ Global Loss: 0.1420            │
│      [disclaimer]          │                                 │
│                            │ Hospital A: 23% weight         │
│ [Ask Question Box]         │ Hospital B: 55% weight ⭐      │
│ [Submit Button]            │ Hospital C: 23% weight         │
│                            │                                 │
│ Example Questions:         │ Agent Decisions:               │
│ • Symptoms of diabetes?    │   Trusted: Hospital B          │
│ • High blood pressure?     │   Unstable: Hospital C         │
│ • Chest pain causes?       │   Queries: 42                  │
├────────────────────────────┴─────────────────────────────────┤
│ 🏗️ ARCHITECTURE VISUALIZATION                               │
│                                                              │
│  [Hospital A] ───►                                          │
│      23%            \                                        │
│  [Hospital B] ──────► [Agent] ────► [Server]               │
│      55% ⭐          /               Round 3                │
│  [Hospital C] ───►                                          │
│      23%                                                     │
│                                                              │
│  (Animated flows, real-time updates)                        │
├──────────────────────────────────────────────────────────────┤
│ 📚 EXPLAINABILITY (Expandable Accordions)                   │
│                                                              │
│ ▶ How Federated Learning Works                             │
│ ▶ How the Agent Makes Decisions                            │
│ ▶ How Privacy is Preserved                                 │
│ ▶ Performance Metrics                                       │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔧 Technical Details

### **File Structure**

```
FED-MED/
├── gradio_app.py                    ⭐ Main Gradio application
├── launch_gradio.sh                 ⭐ Launch script
├── src/
│   └── utils/
│       └── shared_state.py          ⭐ State management
├── static/
│   └── architecture_viz.html        ⭐ D3.js visualization
└── output-models/federated/
    └── metrics/
        └── training_history.json    📊 Training results (auto-loaded)
```

### **How It Works**

1. **Startup:**
   ```python
   # Load inference engine (60 sec)
   inference_engine = InteractiveInferenceEngine(hospital="hospital_B", gpu=3)
   
   # Load monitoring data
   monitor = FedMedMonitor()
   monitor._load_historical_results()  # Loads past training results
   ```

2. **User Queries:**
   ```python
   def chat_fn(message, history):
       response = inference_engine.query(message)  # Fast! 7 sec
       history.append((message, response))
       return history
   ```

3. **Monitoring Updates:**
   ```python
   # Auto-refresh every 3 seconds
   app.load(
       fn=get_monitoring_info,
       outputs=[round_display, hospital_status, agent_decisions],
       every=3
   )
   ```

4. **Architecture Viz:**
   ```javascript
   // Auto-poll state every 15 seconds
   setInterval(() => simulateTraining(), 15000);
   ```

### **Shared State System**

All components communicate via `/src/utils/shared_state.py`:

```python
from src.utils.shared_state import get_shared_state

state = get_shared_state()

# Training updates state
state.update_federated_round(2, "training")
state.update_hospital_status("hospital_A", "training", loss=0.35)
state.update_agent_weights({"hospital_A": 0.25, ...})

# UI reads state
data = state.get()
round_num = state.get("federated.current_round")
```

**State persists** to `shared_state.json` for recovery.

---

## 🎯 Usage Examples

### **Ask Medical Questions**

1. Type in chat: "What are the symptoms of diabetes?"
2. Click "Ask" or press Enter
3. Get response in ~7 seconds
4. See automatic medical disclaimer
5. Query counter increments in monitoring panel

### **View Monitoring**

Right panel shows:
- Round 3/3 (Complete)
- Global Loss: 0.1420
- Hospital B: 55% weight (best performer)
- Agent classified Hospital C as "unstable"

### **Watch Architecture**

Bottom visualization shows:
- Animated flows from hospitals → agent → server
- Weight badges update in real-time
- Nodes highlight during activity

### **Learn How It Works**

Expand accordions to read:
- Federated learning explanation
- Agentic aggregation algorithm
- Privacy preservation mechanisms
- Performance benchmarks

---

## 🔍 Monitoring New Training

If you run new federated training:

```bash
# Terminal 1: Run training with state updates
python src/training/federated_train_with_state.py

# Terminal 2: Gradio UI (already running)
# Watch it update automatically!
```

The UI will show:
- New rounds appearing
- Hospital status changing (idle → training → completed)
- Weights updating after each round
- Architecture flows animating
- Global loss improving

---

## 🎨 Customization

### **Change Hospital Model**

Edit `gradio_app.py`:
```python
# Use Hospital A's model instead of B
inference_engine = InteractiveInferenceEngine(hospital="hospital_A", gpu=3)
```

### **Adjust Refresh Rate**

```python
# Change from 3 seconds to 5 seconds
app.load(
    fn=get_monitoring_info,
    outputs=[...],
    every=5  # ← Change this
)
```

### **Customize Visualization**

Edit `static/architecture_viz.html`:
```javascript
// Change animation speed
setInterval(() => simulateTraining(), 10000);  // 10 sec instead of 15
```

---

## 🐛 Troubleshooting

### **Model Not Loading**

```
Error: Cannot load adapter from output-models/federated/hospital_B/final
```

**Fix:** Run federated training first:
```bash
python src/training/federated_train.py --rounds 3 --gpu 3
```

### **Port Already in Use**

```
Error: Port 7860 is already in use
```

**Fix:** Change port in `gradio_app.py`:
```python
app.launch(server_port=7861)  # Use different port
```

### **Visualization Not Showing**

**Fix:** Ensure `static/architecture_viz.html` exists:
```bash
ls static/architecture_viz.html
```

### **Slow Inference**

**Fix:** Model is reloading. Check initialization:
```
✅ Model loaded! Ready for fast inference.
```

If you see this repeatedly, model isn't staying in memory.

---

## 📊 Example Session

```
Terminal Output:
==================================================
🚀 INITIALIZING FED-MED GRADIO APPLICATION
==================================================

🔄 Loading model from hospital_B...
✅ Model loaded! Ready for fast inference.

==================================================
✅ INITIALIZATION COMPLETE
==================================================

==================================================
🚀 LAUNCHING FED-MED GRADIO APPLICATION
==================================================

Features:
  ✅ Interactive medical Q&A (fast inference)
  ✅ Real-time federated learning monitoring
  ✅ Live architecture visualization
  ✅ Agent decision transparency
  ✅ Comprehensive explainability

==================================================

Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://abc123.gradio.live

Open in browser! 🌐
```

---

## 🎓 Learning Resources

### **Understanding the Code**

1. **Shared State:** `/src/utils/shared_state.py`
   - Thread-safe JSON state manager
   - Enables real-time communication

2. **Inference Engine:** `gradio_app.py` → `InteractiveInferenceEngine`
   - One-time model loading
   - Fast query processing

3. **Monitoring:** `gradio_app.py` → `FedMedMonitor`
   - Loads historical results
   - Provides monitoring data

4. **Visualization:** `/static/architecture_viz.html`
   - D3.js animations
   - Live state polling

### **Extending the UI**

Want to add features?

```python
# Add new tab
with gr.Tab("Training Control"):
    start_btn = gr.Button("Start Training")
    start_btn.click(fn=start_training, outputs=status_text)

# Add new metric
loss_chart = gr.Plot(label="Loss Over Time")
app.load(fn=get_loss_chart, outputs=loss_chart, every=5)
```

---

## 🏆 What Makes This Special

✅ **Fast Inference** - Model loaded once, 9x speedup  
✅ **Real-Time Updates** - See system state live  
✅ **Beautiful Visualization** - D3.js animated diagram  
✅ **Agent Transparency** - See decision-making process  
✅ **Explainability** - Understand every component  
✅ **Privacy Focus** - Show how data stays local  
✅ **Production Ready** - Clean code, error handling  

---

## 📞 Support

**Issues?**
- Check `/src/utils/shared_state.py` for state
- View `shared_state.json` for current values
- Check console output for errors

**Questions?**
- Read explainability sections in UI
- Review code comments in `gradio_app.py`
- Check `QUICK_ANSWERS.md` for FAQ

---

**Built with ❤️ for showcasing privacy-preserving medical AI**

*FED-MED Gradio Application - Interactive, Transparent, Production-Ready*
