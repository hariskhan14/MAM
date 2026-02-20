"""
MAM Interactive Chat API
Provides an interactive CLI to query the full multi-modal medical diagnostic pipeline.

Run:
    python -m use_api
"""

import os
import sys

from pipeline.modality_selection import modality_selection
from pipeline.type_classification import type_classification
from pipeline.role_generation import generate_role
from pipeline.meeting import roles_meeting
from pipeline.diagnosis import final_diagnosis
from pipeline.review import review_all
from pipeline.memory import memory


def run_pipeline(question, file_name, history_id=1):
    """Run the full MAM pipeline for a single question and return the diagnosis."""

    # 1. Modality detection
    modality = modality_selection(question, file_name)
    print(f"[Modality] {modality}")

    # 2. Type classification
    try:
        type_name = type_classification(modality, question, file_name)
    except Exception as e:
        print(f"[Warning] Type classification failed: {e}")
        type_name = 'general'
    if not type_name:
        type_name = 'general'
    print(f"[Type] {type_name}")

    # 3. Role generation
    roles_generated = None
    try:
        roles_generated = generate_role(type_name, modality, question, file_name)
        print("[Roles] Specialist team assembled.")
    except Exception as e:
        print(f"[Warning] Role generation failed: {e}")

    # 4. Multi-agent meeting
    meeting_record = ''
    if roles_generated is not None:
        try:
            print("[Meeting] Starting multi-agent discussion...")
            meeting_record = roles_meeting(question, file_name, modality, type_name, roles_generated, '')
        except Exception as e:
            print(f"[Warning] Meeting failed: {e}")

    # 5. Final diagnosis
    diagnosis = None
    try:
        diagnosis = final_diagnosis(question, file_name, modality, type_name, meeting_record)
    except Exception as e:
        print(f"[Warning] Diagnosis generation failed: {e}")

    # 6. Review
    if diagnosis is not None:
        try:
            review = review_all(question, file_name, modality, type_name, diagnosis)
            print(f"[Review] {review}")
        except Exception as e:
            print(f"[Warning] Review failed: {e}")

    # 7. Save to history
    if diagnosis is not None:
        try:
            os.makedirs('./history', exist_ok=True)
            memory(history_id, question, file_name, modality, diagnosis)
        except Exception as e:
            print(f"[Warning] Memory save failed: {e}")

    return diagnosis


def chat_loop():
    print("=" * 60)
    print("  MAM: Multi-Modal Multi-Agent Medical Diagnosis System")
    print("=" * 60)
    print("Supported file types:")
    print("  Image : .jpg / .jpeg / .png")
    print("  Audio : .wav / .mp3")
    print("  Video : .mp4")
    print("  Text  : no file required")
    print("Type 'exit' or 'quit' to stop.\n")

    history_id = 1

    while True:
        try:
            question = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if question.lower() in ('exit', 'quit'):
            print("Exiting.")
            break
        if not question:
            continue

        file_name = input("File path (press Enter for text-only): ").strip()

        print()
        diagnosis = run_pipeline(question, file_name, history_id)

        if diagnosis:
            print("\n" + "-" * 60)
            print("DIAGNOSIS:")
            print(diagnosis)
            print("-" * 60)
            history_id += 1
        else:
            print("[Error] No diagnosis produced.")

        print()


if __name__ == '__main__':
    chat_loop()