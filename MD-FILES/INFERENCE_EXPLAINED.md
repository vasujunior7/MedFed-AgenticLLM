# 🔍 How Inference Works - Complete Explanation

## ❓ Your Questions Answered

### 1. **Why Does It Take So Long?**

**Problem:** Model loads fresh every time (20-30 seconds)

**Why:**
```
Every time you run: python inference.py "question"
├── Loads Mistral-7B base model (~7GB) ⏱️ 15 seconds
├── Loads LoRA adapter (~13MB)         ⏱️ 2 seconds  
├── Generates answer                   ⏱️ 5-10 seconds
└── Exits (model unloaded from memory)
```

**Solutions:**

✅ **Option 1: Interactive Mode** (BEST for multiple queries)
```bash
python inference_interactive.py
# Loads model ONCE, then you can ask unlimited questions!
# First load: 20 sec, Each answer after: 5-10 sec only
```

✅ **Option 2: Batch Mode**
```bash
python inference.py "Question 1" "Question 2" "Question 3"
# Load once, answer all questions
```

✅ **Option 3: API Server** (for production)
- Load model on startup
- Keep in memory
- Serve via REST API
- Answer instantly (<5 sec per query)

---

## 🧠 How Inference Actually Works

### The Complete Pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                  TRAINING (What We Did)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Round 1: Hospital A, B, C train locally with LoRA        │
│           ↓                                                 │
│  Agent computes weights: A=0.288, B=0.355, C=0.357        │
│           ↓                                                 │
│  Round 2: Continue training with global knowledge          │
│           ↓                                                 │
│  Round 3: Final training                                   │
│           ↓                                                 │
│  Final Agent Weights: A=0.227, B=0.547, C=0.225           │
│                                                             │
│  Result: Each hospital has fine-tuned LoRA adapter         │
│          - hospital_A/final/  (trained on 4,520 samples)   │
│          - hospital_B/final/  (trained on 2,521 samples)   │
│          - hospital_C/final/  (trained on 2,959 samples)   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                INFERENCE (What Happens Now)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Load base Mistral-7B (NOT trained, just base model)    │
│                                                             │
│  2. Load LoRA adapter from ONE hospital:                    │
│     → Currently: hospital_B (best performer)                │
│     → Loss: 0.0416 (lowest)                                │
│     → Agent weight: 0.547 (highest)                        │
│                                                             │
│  3. Base Model + LoRA = Fine-tuned Medical Model           │
│                                                             │
│  4. Generate answer to your question                        │
│                                                             │
│  5. Add safety disclaimer                                   │
└─────────────────────────────────────────────────────────────┘
```

### Yes! It Uses Federated Training Results

**What the model learned:**
- ✅ Trained on 10,000 medical Q&A samples across 3 hospitals
- ✅ 3 federated rounds (hospitals didn't share data, only LoRA weights)
- ✅ Agent-weighted aggregation (smart weighting, not just averaging)
- ✅ Final LoRA adapters contain all the medical knowledge

**What you're using:**
- Base Mistral-7B + Hospital B's LoRA adapter
- Hospital B had the best performance (lowest loss, highest agent weight)
- Contains medical knowledge from federated training

---

## 🏥 Hospital Selection - How It Works

### Current Logic:

```python
# In inference.py
if args.hospital:
    # User explicitly chose: --hospital hospital_A
    adapter_path = f"output-models/federated/{args.hospital}/final"
else:
    # AUTO-SELECT: Use hospital_B (best performer)
    adapter_path = "output-models/federated/hospital_B/final"
```

### Why Hospital B?

From Round 3 results:
```
Hospital A: Loss = 0.3217, Agent Weight = 0.227 (worst performance)
Hospital B: Loss = 0.0416, Agent Weight = 0.547 ⭐ BEST
Hospital C: Loss = 0.2043, Agent Weight = 0.225 (unstable)
```

**Agent Analysis:**
- Hospital B: Low loss + stable training = highest weight (0.547)
- Hospital A: Loss increased (0.0375 → 0.3217) = penalized
- Hospital C: High variance detected = penalized

---

## 🤖 Role of Agents

### During Training (Rounds 1-3):

```python
# Agent's job: Compute smart aggregation weights
agent = AgenticAggregator(loss_weight=0.6, variance_weight=0.4)

# For each round:
weights = agent.compute_aggregation_weights(
    client_metrics=[
        {'hospital': 'A', 'loss': 0.3217, 'variance': 0.02},
        {'hospital': 'B', 'loss': 0.0416, 'variance': 0.01},
        {'hospital': 'C', 'loss': 0.2043, 'variance': 0.15}  # unstable!
    ]
)

# Agent scores:
# - 60% based on loss (lower = better)
# - 40% based on stability (less variance = better)
# - Penalizes unstable clients
# - Detects malicious behavior (increasing loss)

Result: weights = [0.227, 0.547, 0.225]  # B gets highest!
```

### During Inference (Now):

**Agents are NOT used during inference!**

- No aggregation happening
- No weight computation
- Just loading pre-trained LoRA adapter
- Agent's work is "baked into" the final models

The agent already did its job during training to:
1. Select which hospitals contribute more to global knowledge
2. Penalize unstable/bad performers
3. Create high-quality LoRA adapters

Now we just use the best one (Hospital B)!

---

## 📊 Complete Flow Diagram

```
USER QUESTION: "I have chest pain"
        ↓
┌───────────────────────────────────┐
│  inference.py loads model         │
│  - Base: Mistral-7B (3.7B params) │
│  - LoRA: hospital_B/final         │
│  - Total VRAM: 3.86 GB            │
└───────────────────────────────────┘
        ↓
┌───────────────────────────────────┐
│  Format prompt:                   │
│  [INST] You are a medical AI...   │
│  Question: I have chest pain      │
│  [/INST]                          │
└───────────────────────────────────┘
        ↓
┌───────────────────────────────────┐
│  Generate with LoRA adapter:      │
│  - Model has learned medical      │
│    knowledge from 10k samples     │
│  - Uses federated training        │
│  - Temperature=0.7 (creative)     │
│  - Max tokens=256                 │
└───────────────────────────────────┘
        ↓
┌───────────────────────────────────┐
│  Safety guardrails check:         │
│  - No diagnosis claims            │
│  - No dangerous advice            │
│  - Add disclaimer                 │
└───────────────────────────────────┘
        ↓
RESPONSE: "Chest pain can indicate...
          Please consult a doctor... ⚠️"
```

---

## 🚀 Performance Optimization

### Current (inference.py):
- **First query:** 20-30 seconds (model loading)
- **Each query:** 20-30 seconds (reloads every time!)
- **Memory:** Cleared after each run

### Optimized (inference_interactive.py):
- **First load:** 20-30 seconds (one-time)
- **Each query after:** 5-10 seconds only! ⚡
- **Memory:** Model stays loaded

### Example:
```bash
# OLD WAY (SLOW):
python inference.py "Question 1"  # 25 seconds
python inference.py "Question 2"  # 25 seconds
python inference.py "Question 3"  # 25 seconds
# Total: 75 seconds for 3 questions

# NEW WAY (FAST):
python inference_interactive.py
# Load once: 25 seconds
# Question 1: 7 seconds
# Question 2: 7 seconds  
# Question 3: 7 seconds
# Total: 46 seconds for 3 questions (39% faster!)
```

---

## 🎯 Key Takeaways

1. **Model loads every time** = slow (inference.py)
   - Use interactive mode to avoid this!

2. **Yes, it uses federated training results**
   - LoRA adapters contain learned medical knowledge
   - Trained on 10k samples across 3 hospitals
   - No raw data was shared (privacy preserved)

3. **Hospital selection**: Currently hospital_B (best performer)
   - You can choose: `--hospital hospital_A`
   - Or use default (auto-selects best)

4. **Agents**: Used during TRAINING, not inference
   - Computed smart weights (0.227, 0.547, 0.225)
   - Penalized unstable clients
   - Created high-quality LoRA adapters

5. **How it answers**:
   - Base model (Mistral-7B) + LoRA weights
   - LoRA contains medical knowledge from federated learning
   - Safety checks + disclaimer added

---

## 🛠️ Try It Now!

**Fast interactive mode:**
```bash
python inference_interactive.py
# Ask multiple questions without reloading!
```

**Single query mode:**
```bash
python inference.py "What causes diabetes?"
```

**Choose specific hospital:**
```bash
python inference_interactive.py --hospital hospital_A
python inference_interactive.py --hospital hospital_C
```

**Compare all hospitals:**
```bash
# Hospital A (4,520 samples, weight=0.227)
python inference.py --hospital hospital_A "Symptoms of flu?"

# Hospital B (2,521 samples, weight=0.547) ⭐ BEST
python inference.py --hospital hospital_B "Symptoms of flu?"

# Hospital C (2,959 samples, weight=0.225)
python inference.py --hospital hospital_C "Symptoms of flu?"
```
