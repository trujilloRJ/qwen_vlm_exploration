import json
import os
import re
from typing import TypedDict

import tqdm


class TextvqaAnswer(TypedDict):
    question_id: int
    question: str
    answers: list[str]
    predicted_answer: str


def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower().strip())


def calculate_vqa_score(prediction, answers: list[str]) -> float:
    """
    Calculates the official soft VQA evaluation metric.
    Expects both prediction and ground_truth_list elements to be pre-cleaned.
    """
    # Count how many humans provided the exact same answer as the model
    matching_human_count = answers.count(prediction)
    
    # Apply the official formula: min(matching_count / 3, 1.0)
    score = min(matching_human_count / 3.0, 1.0)
    return score


if __name__ == "__main__":
    model_name = "Qwen3.5-2B"
    dataset_name = "textvqa"
    res_folder = r"C:\jtr\side_projects\LLMs\qwen_vlm_exploration\_results"
    
    results_path = os.path.join(res_folder, model_name, f"{dataset_name}.json")
    with open(results_path, "r") as f:
        predictions: list[TextvqaAnswer] = json.load(f)

    n_samples = len(predictions)
    results = []
    for result in tqdm.tqdm(predictions, desc="Evaluating...", total=n_samples):
        answers = result["answers"]
        predicted_answer = result["predicted_answer"]

        clean_prediction = clean_text(predicted_answer)
        clean_answers = [clean_text(ans) for ans in answers]
        score = calculate_vqa_score(clean_prediction, clean_answers)

        results.append({
            "question_id": result["question_id"],
            "question": result["question"],
            "score": score
        })

    # Calculate the average score
    average_score = sum(r["score"] for r in results) / n_samples
    print(f"Average VQA Score: {average_score:.4f}")

    # Save metrics
    out_path = os.path.join("_results", model_name, f"{dataset_name}_metrics.json")
    with open(out_path, "w") as f: 
        json.dump({
            "average_score": average_score,
            "per_question_scores": results
        }, f, indent=4)
