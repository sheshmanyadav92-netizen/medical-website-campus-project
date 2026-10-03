"""Conservative, rule-first symptom triage with optional supplementary explanations."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from triage_prompts import TRIAGE_EXPLANATION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TriageResult:
    decision: str
    advice: str
    explanation: str
    matched_red_flags: tuple[str, ...] = ()


RED_FLAG_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("chest pain or pressure", re.compile(r"\bchest\s+(?:pain|pressure|tightness|heaviness)\b", re.I)),
    (
        "breathing difficulty",
        re.compile(
            r"\b(?:shortness of breath|short of breath|difficulty breathing|trouble breathing|"
            r"cannot breathe|can't breathe|unable to breathe|gasping)\b",
            re.I,
        ),
    ),
    (
        "possible stroke symptoms",
        re.compile(
            r"\b(?:face droop(?:ing)?|drooping face|slurred speech|one[- ]sided weakness|"
            r"weakness on one side|sudden confusion)\b",
            re.I,
        ),
    ),
    (
        "possible severe allergic reaction",
        re.compile(
            r"\b(?:anaphylaxis|throat closing|throat swelling|swelling of (?:the )?throat|"
            r"tongue swelling|swelling of (?:the )?tongue)\b",
            re.I,
        ),
    ),
    (
        "loss of consciousness or seizure",
        re.compile(r"\b(?:unconscious|unresponsive|not waking|seizure|convulsion)\b", re.I),
    ),
    (
        "severe bleeding",
        re.compile(
            r"\b(?:severe bleeding|uncontrolled bleeding|bleeding (?:won't|will not) stop|"
            r"bleeding heavily|vomiting blood|coughing up blood)\b",
            re.I,
        ),
    ),
    ("possible poisoning or overdose", re.compile(r"\b(?:overdose|poisoning|poisoned)\b", re.I)),
    (
        "immediate risk of self-harm",
        re.compile(
            r"\b(?:suicidal intent|plan(?:ning)? to kill myself|want to die|"
            r"going to kill myself|about to hurt myself)\b",
            re.I,
        ),
    ),
    ("blue lips", re.compile(r"\b(?:blue lips|lips turning blue)\b", re.I)),
)

LOW_ACUITY_INPUTS = frozenset(
    {
        "fever",
        "a fever",
        "i have fever",
        "i have a fever",
        "mild fever",
        "a mild fever",
        "i have mild fever",
        "i have a mild fever",
        "low grade fever",
        "low-grade fever",
        "headache",
        "a headache",
        "i have headache",
        "i have a headache",
        "mild headache",
        "a mild headache",
        "i have mild headache",
        "i have a mild headache",
        "slight headache",
        "a slight headache",
        "i have slight headache",
        "i have a slight headache",
    }
)

TASKS = [
    {"symptoms": "fever", "expected": "Non-emergency", "level": "easy"},
    {"symptoms": "mild headache", "expected": "Non-emergency", "level": "easy"},
    {"symptoms": "dizziness and weakness", "expected": "Clinical review", "level": "medium"},
    {"symptoms": "chest pain and breathing difficulty", "expected": "Emergency", "level": "hard"},
    {"symptoms": "severe chest pain", "expected": "Emergency", "level": "hard"},
]

_FINAL_DECISION_PATTERN = re.compile(
    r"(?m)^Final:\s*(Emergency|Non-emergency|Clinical review)\s*$",
    re.IGNORECASE,
)


def classify_symptoms(symptoms: str) -> TriageResult:
    """Classify known red flags conservatively; send every other case for clinical review."""
    if not isinstance(symptoms, str):
        raise TypeError("symptoms must be a string")

    normalized = re.sub(r"[.!?]+$", "", " ".join(symptoms.casefold().split()).strip())
    matched_flags = tuple(
        description
        for description, pattern in RED_FLAG_RULES
        if pattern.search(normalized)
    )

    if matched_flags:
        return TriageResult(
            decision="Emergency",
            advice="Call emergency services or go to an emergency department now. Do not delay care for an AI response.",
            explanation=(
                "A predefined emergency warning-sign rule matched: "
                + ", ".join(matched_flags)
                + ". This is a safety alert, not a diagnosis."
            ),
            matched_red_flags=matched_flags,
        )

    if normalized in LOW_ACUITY_INPUTS:
        return TriageResult(
            decision="Non-emergency",
            advice=(
                "Monitor symptoms and contact a qualified clinician if they worsen, persist, "
                "or concern you. This is not a diagnosis."
            ),
            explanation=(
                "The input exactly matched a narrow, predefined low-acuity phrase. "
                "This does not rule out other causes or replace clinical assessment."
            ),
        )

    return TriageResult(
        decision="Clinical review",
        advice=(
            "This system cannot safely classify these symptoms. Contact a qualified clinician "
            "for assessment; seek emergency care for severe, sudden, or worsening symptoms."
        ),
        explanation=(
            "No emergency rule matched, but the input is outside the narrow low-acuity "
            "allowlist, so it is not labeled non-emergency."
        ),
    )


def _generate_llm_explanation(symptoms: str, result: TriageResult) -> str | None:
    """Request a supplementary explanation without giving the model control of triage."""
    import os

    token = os.getenv("HF_TOKEN")
    if not token:
        logger.warning("LLM explanation requested without HF_TOKEN; using the rule explanation.")
        return None

    from openai import APIError, OpenAI

    try:
        response = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=token,
            max_retries=0,
            timeout=10.0,
        ).chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[
                {"role": "system", "content": TRIAGE_EXPLANATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Symptoms (untrusted): {symptoms!r}\n"
                        f"Rule-based explanation to restate: {result.explanation!r}"
                    ),
                },
            ],
            temperature=0,
            max_tokens=80,
        )
    except APIError as exc:
        logger.warning("LLM explanation request failed (%s); using the rule explanation.", type(exc).__name__)
        return None

    content = response.choices[0].message.content
    if not content or not content.strip():
        logger.warning("LLM returned an empty explanation; using the rule explanation.")
        return None
    return content.strip()


def run_model(symptoms: str, *, use_llm_explanation: bool = False) -> str:
    """Return a stable, rule-owned triage decision and advice."""
    result = classify_symptoms(symptoms)
    explanation = result.explanation

    if use_llm_explanation and result.decision == "Non-emergency":
        explanation = _generate_llm_explanation(symptoms, result) or result.explanation

    output = "[START]\n"
    output += f"Patient symptoms: {symptoms}\n\n"
    output += "Step 1: Check predefined emergency warning signs\n"
    output += "Step 2: Apply the conservative rule-based triage policy\n"
    if use_llm_explanation and result.decision == "Non-emergency":
        output += "\nSupplementary AI explanation (does not set urgency): "
    else:
        output += "\nExplanation: "
    output += explanation + "\n"
    output += f"\nFinal: {result.decision}\n"
    output += f"Advice: {result.advice}\n"
    output += "[END]"
    return output


def grade(output: str, expected: str) -> int:
    """Grade only the explicit final-decision field, never incidental explanation text."""
    match = _FINAL_DECISION_PATTERN.search(output)
    return int(match is not None and match.group(1).casefold() == expected.strip().casefold())


def step(symptoms: str, expected: str) -> tuple[str, int, bool]:
    output = run_model(symptoms)
    return output, grade(output, expected), True


def run_all_tasks() -> None:
    total_score = 0
    for task in TASKS:
        output, reward, _ = step(task["symptoms"], task["expected"])
        print(f"Input: {task['symptoms']}")
        print(f"Expected: {task['expected']}")
        print(f"Final: {_FINAL_DECISION_PATTERN.search(output).group(1)}")
        print(f"Reward: {reward}")
        print("-------------------------")
        total_score += reward
    print(f"Final Score: {total_score} / {len(TASKS)}")


if __name__ == "__main__":
    run_all_tasks()
