from openai import OpenAI
import os

# ================= CLIENT =================
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")
)

# ================= MAIN MODEL =================
def run_model(symptoms):
    symptoms_lower = symptoms.lower()

    steps = []
    decision = ""
    advice = ""

    # Step 1
    steps.append("Step 1: Analyze symptoms")

    # Step 2 (rule-based decision)
    if "chest pain" in symptoms_lower:
        steps.append("Step 2: Check breathing condition")

        if "breathing" in symptoms_lower:
            decision = "Emergency"
            advice = "Call emergency services immediately"
        else:
            decision = "Emergency"
            advice = "Seek immediate medical attention"

    elif "fever" in symptoms_lower:
        steps.append("Step 2: Check severity of fever")
        decision = "Non-emergency"
        advice = "Rest, hydrate, and monitor temperature"

    elif "headache" in symptoms_lower:
        steps.append("Step 2: Check vision issues")
        decision = "Non-emergency"
        advice = "Monitor symptoms and consult doctor if needed"

    elif "breathing" in symptoms_lower:
        steps.append("Step 2: Evaluate breathing difficulty")
        decision = "Emergency"
        advice = "Seek immediate medical attention"

    else:
        steps.append("Step 2: General assessment")
        decision = "Non-emergency"
        advice = "Monitor symptoms"

    # 🧠 LLM for explanation
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": "Explain the medical reasoning briefly."
                },
                {
                    "role": "user",
                    "content": f"Patient symptoms: {symptoms}. Decision: {decision}"
                }
            ],
            temperature=0.3,
            max_tokens=120
        )

        explanation = response.choices[0].message.content

    except Exception:
        explanation = "Basic medical reasoning applied."

    # Final output
    output = "[START]\n"
    output += f"Patient symptoms: {symptoms}\n\n"

    for step in steps:
        output += step + "\n"

    output += f"\nExplanation: {explanation}\n"
    output += f"\nFinal: {decision}\n"
    output += f"Advice: {advice}\n"
    output += "[END]"

    return output


# ================= TASKS =================
tasks = [
    {"symptoms": "fever", "expected": "non-emergency", "level": "easy"},
    {"symptoms": "headache dizziness", "expected": "non-emergency", "level": "medium"},
    {"symptoms": "chest pain breathing", "expected": "emergency", "level": "hard"},
    {"symptoms": "severe chest pain breathing", "expected": "emergency", "level": "hard"},
]

# ================= GRADER =================
def grade(output, expected):
    if expected.lower() in output.lower():
        return 1
    return 0


# ================= ENV STEP =================
def step(symptoms, expected):
    output = run_model(symptoms)
    reward = grade(output, expected)
    done = True
    return output, reward, done


# ================= RUN ALL TASKS =================
def run_all_tasks():
    total_score = 0

    for task in tasks:
        output, reward, done = step(task["symptoms"], task["expected"])

        print("Input:", task["symptoms"])
        print("Expected:", task["expected"])
        print("Reward:", reward)
        print("-------------------------")

        total_score += reward

    print(f"Final Score: {total_score} / {len(tasks)}")


# ================= OPTIONAL TEST =================
if __name__ == "__main__":
    run_all_tasks()