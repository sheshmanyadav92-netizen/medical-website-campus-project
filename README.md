## 🔗 Live Demo
https://huggingface.co/spaces/Avimanyu8008/medtriage-env

# 🏥 AI Medical Triage System (OpenEnv RL Project

This project follows OpenEnv-style environment design with tasks, reward, and evaluation loop.

## 👋 Hi, I’m Krishna

This project started as a simple idea:
**“Can we build a system that thinks like a first-level doctor?”**

Not to replace doctors — but to **quickly classify cases as emergency vs non-emergency** and guide users.

But I didn’t stop there.

I turned this into an **RL-style evaluation system using OpenEnv**, where:

* the AI *acts like an agent*
* patient symptoms are *inputs*
* decisions are *actions*
* correctness is *rewarded*

So this is not just an app —
👉 it’s a **mini AI environment with grading + scoring**

---

# 🚨 Problem

In real life:

* People panic for minor issues
* People ignore serious symptoms
* Hospitals get overloaded

We need a **fast triage layer** that:

* understands symptoms
* prioritizes severity
* gives instant guidance

---

# 💡 Solution

I built a system that:

1. Takes **natural language symptoms**
2. Applies **rule-based reasoning (fast + reliable)**
3. Uses **LLM for explanation (human-like thinking)**
4. Classifies into:

   * Emergency 🚨
   * Non-emergency ✅
5. Provides:

   * Decision
   * Advice
   * Explanation

---

# 🧠 What makes this project different?

Most people built:
❌ Chatbots
❌ Simple symptom checkers

I built:

👉 **A complete AI evaluation environment**

### ✔ Agent-style system

* Input → Symptoms
* Output → Decision
* Behavior → Reasoning

### ✔ Task system (Easy / Medium / Hard)

* Single symptom → Easy
* Multiple symptoms → Medium
* Complex combinations → Hard

### ✔ Automated Grader

* Checks if AI output matches expected decision
* No manual checking

### ✔ Reward Logic

* Correct → +1
* Wrong → 0

### ✔ Final Score

* Shows performance of the AI system

👉 This is exactly how RL environments are structured.

---

# 🎮 OpenEnv Integration

This project follows OpenEnv principles:

| Requirement | Implementation            |
| ----------- | ------------------------- |
| Environment | `inference.py`            |
| Tasks       | Defined inside code       |
| Agent       | `run_model()`             |
| Actions     | Emergency / Non-emergency |
| Reward      | 0 or 1                    |
| Evaluation  | Final score               |

---

# ⚙️ How it works (simple)

### Step 1: Rule-based decision

Fast logic for safety:

* chest pain → emergency
* breathing issue → emergency
* fever → non-emergency

👉 This ensures **reliability**

---

### Step 2: LLM explanation

Using:

* Qwen 2.5 (via HuggingFace)

It explains:

* WHY the decision was made
* Like a real doctor

👉 This adds **intelligence + trust**

---

### Step 3: Output format

```
[START]
Patient symptoms: ...

Step 1: Analyze symptoms  
Step 2: ...

Explanation: ...

Final: Emergency / Non-emergency  
Advice: ...

[END]
```

---

### Step 4: Evaluation system

Example:

```
Input: fever  
Expected: non-emergency  
Reward: 1  

Final Score: 4/4
```

👉 This turns the project into a **testable AI system**

---

# 🧪 Difficulty Levels

### 🟢 Easy

* fever
* headache

### 🟡 Medium

* chest pain + breathing
* dizziness + weakness

### 🔴 Hard

* long natural language inputs
* multiple mixed symptoms
* real-world messy descriptions

👉 The system handles ALL of them.

---

# 🖥️ UI (Gradio)

* Simple interface
* Enter symptoms
* Get instant decision

Built using:

* Gradio
* HuggingFace Spaces

---

# 📂 Project Structure

```
MedTriageEnvs/
│
├── app.py              # UI
├── inference.py       # Core logic + RL tasks
├── openenv.yaml       # Environment config
├── requirements.txt   # Dependencies
├── env/               # Modular structure
├── tasks/             # Task definitions
```

---

# 🚀 How to run

```bash
pip install -r requirements.txt
python app.py
```

---

# 🔥 Why this project stands out

### 1. Not just UI — full system

Most submissions stop at frontend
👉 This has backend + logic + evaluation

---

### 2. RL thinking applied

Even without full RL training:

* Agent
* Tasks
* Rewards
* Evaluation

👉 This is how real AI systems are tested

---

### 3. Safe + practical

* Rule-based core (no hallucination risk)
* LLM only for explanation

---

### 4. Scalable design

Can be extended to:

* multilingual support
* voice input
* hospital integration

---

### 5. Real-world impact

This is not a toy project.

👉 This can actually be used as:

* hospital triage assistant
* telemedicine pre-check system

---

# 📌 What I learned

* How to combine **rules + LLM**
* How to design **evaluation systems**
* How to think like an **AI system designer**
* How to structure projects for **real-world use**

---

# 🏁 Final Thought

I didn’t just build an app.

I built a **mini AI system that can be tested, evaluated, and improved like a real model.**

That’s what makes this project different.

---

## 👨‍💻 Built by

**TEAM : Attack on Titan**

## 👨‍💻 Team Members
**Avimanyu**
**Krishna**


---
