# 🏥 AI Medical Triage System (OpenEnv RL Project)

This project follows OpenEnv-style environment design with tasks, rewards, and an evaluation loop.
It is an educational prototype, not a diagnostic tool or a clinically validated triage service.

It demonstrates a rule-first symptom classification environment where:

* the AI *acts like an agent*
* patient symptoms are *inputs*
* decisions are *actions*
* correctness is *rewarded*

The code is derived from the MIT-licensed [ai-medical-triage project](https://github.com/engineerkrish/ai-medical-triage); see [LICENSE](LICENSE) for the required notice.

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

The system:

1. Takes **natural language symptoms**
2. Applies **conservative rule-based triage**
3. Uses an optional LLM for supplementary explanations; the deterministic rules own urgency and advice
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

This project provides:

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
| Tasks       | `inference.py`            |
| Agent       | `run_model()`             |
| Actions     | Emergency / Clinical review / Non-emergency |
| Reward      | 0 or 1                    |
| Evaluation  | Final score               |

---

# ⚙️ How it works (simple)

### Step 1: Check predefined emergency warning signs

Predefined emergency warning signs always take priority. Only a narrow allowlist
of fever/headache phrases is classified as non-emergency; unknown or mixed
symptoms are sent for clinical review. These prototype rules are incomplete and
do not guarantee safety.

---

### Step 2: Optional LLM explanation

The model can only provide a supplementary explanation for allowlisted
non-emergency phrases. It is disabled by default and never controls the final
decision or advice. Emergency cases bypass the model entirely.

---

### Step 3: Output format

```
[START]
Patient symptoms: ...

Step 1: Check predefined emergency warning signs
Step 2: Apply the conservative rule-based triage policy

Explanation: ...

Final: Emergency / Clinical review / Non-emergency
Advice: ...

[END]
```

---

### Step 4: Evaluation system

Example:

```
Input: fever  
Expected: Non-emergency
Reward: 1  

Final Score: 5/5
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

* mixed symptoms and instruction-injection examples

Unmatched or complex inputs require clinical review; the system does not claim
to handle all real-world symptoms.

---

# 📂 Project Structure

```
inference.py                   # Rule-based classifier, output, grading, tasks
triage_prompts.py              # Optional explanation prompt
benchmark.py                   # Reproducible synthetic rule-only benchmark
server/app.py                  # FastAPI/OpenEnv scaffold
tasks/__init__.py              # Tasks package
tests/test_triage.py           # Offline unit tests
docs/safety-evaluation.md      # Guardrails, benchmark, prompt, feedback template
openenv.yaml                   # Environment configuration
requirements.txt               # Runtime dependencies
```

---

# 🚀 How to run

```bash
pip install -r requirements.txt
python inference.py
```

Run the deterministic benchmark and unit tests:

```bash
python benchmark.py
python -m unittest discover -s tests -v
```

See [Safety policy and evaluation](docs/safety-evaluation.md) for guardrails,
the prompt artifact, benchmark scope, and the user-validation feedback template.

---

# 🔥 Project scope

### 1. Rule-first evaluation system

The project combines triage rules with a reproducible synthetic evaluation suite.

---

### 2. RL thinking applied

Even without full RL training:

* Agent
* Tasks
* Rewards
* Evaluation

👉 This is how real AI systems are tested

---

### 3. Conservative prototype

* Rule-based final decision and advice
* Optional LLM explanation is supplementary and disabled by default
* Not clinically validated; seek professional care for concerning symptoms

---

### 4. Scalable design

Can be extended to:

* multilingual support
* voice input
* hospital integration

---

### 5. Limitations

This is an educational prototype and is not ready for hospital, telemedicine,
or other clinical use.

---

# 📌 Project goals

* Demonstrate **rules with optional LLM explanations**
* Make behavior measurable with **synthetic evaluation cases**
* Document **guardrails, limitations, and validation needs**

---

# 🏁 Limitations

The system uses a small, incomplete keyword ruleset and has not been clinically
validated. It is not intended for real-world medical triage or treatment decisions.

---
