"""Train and evaluate the HR policy GPT-2 classifier."""

import json
from pathlib import Path

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from app.core import (
    CATEGORIES,
    DEFAULT_MODEL_DIR,
    PROJECT_ROOT,
    CATEGORIES,
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
        "training": {
            "base_model": "gpt2",
            "train_examples": len(split_hr_dataset(records)[0]),
            "epochs": 20,
            "per_device_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "max_sequence_length": 128,
            "seed": 42,
            "checkpoint_selection": "lowest validation loss",
        },
        "historical_baseline_accuracy": 0.2,
        "predictions": predictions,
    }
    expected = [record["label"] for record in validation]
    predicted = [item["predicted"] for item in predictions]
    precision, recall, f1, support = precision_recall_fscore_support(
        expected,
        predicted,
        labels=list(CATEGORIES),
        zero_division=0,
    )
    evidence["macro_f1"] = float(f1.mean())
    evidence["category_metrics"] = {
        category: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, category in enumerate(CATEGORIES)
    }
    matrix = confusion_matrix(expected, predicted, labels=list(CATEGORIES))
    evidence["confusion_matrix"] = {
        "labels": list(CATEGORIES),
        "matrix": matrix.tolist(),
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
