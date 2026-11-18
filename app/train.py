"""Train and evaluate the HR policy GPT-2 classifier."""

import json
from pathlib import Path

from app.core import (
    DEFAULT_MODEL_DIR,
    PROJECT_ROOT,
    classify_text,
    load_hr_dataset,
    majority_baseline,
    split_hr_dataset,
    train_hr_classifier,
)


def evaluate(output_dir: str | Path = DEFAULT_MODEL_DIR) -> dict:
    model_dir = Path(output_dir)
    tokenizer, model = _load_saved(model_dir)
    records = load_hr_dataset()
    _, validation = split_hr_dataset(records)
    correct = 0
    predictions = []
    for record in validation:
        predicted = classify_text(record["text"], tokenizer, model, model_dir)
        correct += predicted == record["label"]
        predictions.append(
            {
                "text": record["text"],
                "expected": record["label"],
                "predicted": predicted,
            }
        )

    accuracy = correct / len(validation)
    baseline_category, baseline_accuracy = majority_baseline(validation)
    evidence = {
        "accuracy": accuracy,
        "baseline_category": baseline_category,
        "baseline_accuracy": baseline_accuracy,
        "validation_examples": len(validation),
        "predictions": predictions,
    }
    evidence_path = model_dir / "evaluation.json"
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    public_evidence = PROJECT_ROOT / "data" / "evaluation.json"
    public_evidence.parent.mkdir(parents=True, exist_ok=True)
    public_evidence.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


def _load_saved(model_dir: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    model.eval()
    return tokenizer, model


def main() -> None:
    training = train_hr_classifier()
    model_dir = Path(training["output_dir"])
    evidence = evaluate(model_dir)
    print(json.dumps({**training, **evidence}, indent=2))


if __name__ == "__main__":
    main()
