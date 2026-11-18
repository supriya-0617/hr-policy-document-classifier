"""HR policy classification data, training, and inference utilities."""

import csv
import os
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    from transformers import Trainer, TrainingArguments
except ImportError:
    Trainer = None
    TrainingArguments = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "hr_policy_examples.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "hr-gpt2"

CATEGORIES = (
    "Leave Policy",
    "Expense Claim",
    "Recruitment",
    "Employee Benefits",
    "Workplace Conduct",
)
CATEGORY_TOKENS = {category: f"<HR_{category.upper().replace(' ', '_')}>" for category in CATEGORIES}


def load_hr_dataset(path: str | Path = DATASET_PATH) -> list[dict[str, str]]:
    """Load and validate the manually labelled HR paragraph dataset."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"HR dataset not found: {dataset_path}")

    records: list[dict[str, str]] = []
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"text", "label", "split"}
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f"Dataset columns must be {sorted(required)}")
        for row in reader:
            text = row["text"].strip()
            label = row["label"].strip()
            split = row["split"].strip()
            if not text:
                raise ValueError("Dataset contains an empty paragraph")
            if label not in CATEGORIES:
                raise ValueError(f"Invalid HR label: {label}")
            if split not in {"train", "validation"}:
                raise ValueError(f"Invalid dataset split: {split}")
            records.append({"text": text, "label": label, "split": split})
    if not records:
        raise ValueError("HR dataset is empty")
    return records


def split_hr_dataset(records: Iterable[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train = [record for record in records if record["split"] == "train"]
    validation = [record for record in records if record["split"] == "validation"]
    if not train or not validation:
        raise ValueError("Both train and validation splits are required")
    return train, validation


def split_paragraphs(text: str) -> list[str]:
    """Split uploaded text into non-empty sections for classification."""
    if not text or not text.strip():
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    if len(paragraphs) == 1:
        paragraphs = [line.strip() for line in normalized.splitlines() if line.strip()]
    return paragraphs


def extend_gpt2_tokenizer(model_name: str = "gpt2"):
    """Load GPT-2 and add all category tokens, resizing token embeddings."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens = list(CATEGORY_TOKENS.values())
    added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    for category, token in CATEGORY_TOKENS.items():
        if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id:
            raise RuntimeError(f"Tokenizer did not recognize {category} token")
    if added != len(special_tokens):
        # Models loaded from a directory may already contain these tokens.
        if all(token in tokenizer.get_vocab() for token in special_tokens) is False:
            raise RuntimeError("Could not add the HR category tokens")
    return tokenizer, model


def prepare_training_data(records: Iterable[dict[str, str]], tokenizer, max_length: int = 256):
    """Create a masked causal-language-modeling dataset for category prediction."""
    from datasets import Dataset

    encoded_rows: list[dict] = []
    for record in records:
        text = record["text"].strip()
        label = record["label"].strip()
        token = CATEGORY_TOKENS[label]
        prompt = f"HR Policy Text:\n{text}\nCategory:"
        full_text = f"{prompt}{token}"
        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]
        labels = list(input_ids)
        category_id = tokenizer.convert_tokens_to_ids(token)
        try:
            category_index = input_ids.index(category_id)
        except ValueError as exc:
            raise ValueError(f"Category token was truncated for label {label}") from exc
        for index in range(category_index):
            labels[index] = -100
        for index in range(category_index + 1, len(labels)):
            labels[index] = -100
        encoded_rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": encoded["attention_mask"],
                "labels": labels,
            }
        )
    return Dataset.from_list(encoded_rows)


def train_hr_classifier(
    output_dir: str | Path = DEFAULT_MODEL_DIR,
    model_name: str = "gpt2",
    epochs: float = 3.0,
    batch_size: int = 2,
) -> dict:
    """Fine-tune GPT-2 with the Hugging Face Trainer class."""
    if Trainer is None or TrainingArguments is None:
        raise RuntimeError("Transformers is required for model training")

    records = load_hr_dataset()
    train_records, validation_records = split_hr_dataset(records)
    tokenizer, model = extend_gpt2_tokenizer(model_name)
    train_dataset = prepare_training_data(train_records, tokenizer)
    validation_dataset = prepare_training_data(validation_records, tokenizer)

    random.seed(42)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        seed=42,
        report_to=[],
        remove_unused_columns=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    metrics = trainer.evaluate()

    best_dir = output_path / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    return {
        "output_dir": str(best_dir),
        "train_examples": len(train_records),
        "validation_examples": len(validation_records),
        "eval_loss": float(metrics.get("eval_loss", 0.0)),
    }


def load_classifier(model_dir: str | Path = DEFAULT_MODEL_DIR):
    """Load a trained classifier model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run: python -m app.train"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_pretrained(str(path))
    model.eval()
    return tokenizer, model


def classify_text(
    text: str,
    tokenizer=None,
    model=None,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> str:
    """Classify text using the next-token prediction of the fine-tuned GPT-2 model."""
    if not text or not text.strip():
        raise ValueError("Cannot classify empty text")
    if tokenizer is None or model is None:
        tokenizer, model = load_classifier(model_dir)

    import torch

    prompt = f"HR Policy Text:\n{text.strip()}\nCategory:"
    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits[:, -1, :]

    category_ids = [tokenizer.convert_tokens_to_ids(token) for token in CATEGORY_TOKENS.values()]
    scores = logits[0, category_ids]
    predicted_index = int(torch.argmax(scores).item())
    predicted_token = list(CATEGORY_TOKENS.values())[predicted_index]
    if category_ids[predicted_index] >= model.get_input_embeddings().weight.shape[0]:
        raise RuntimeError("Predicted category token is outside the model vocabulary")
    return next(category for category, token in CATEGORY_TOKENS.items() if token == predicted_token)


def classify_paragraphs(
    text: str,
    tokenizer=None,
    model=None,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> list[dict[str, str]]:
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        raise ValueError("Document contains no text paragraphs")
    return [
        {"text": paragraph, "category": classify_text(paragraph, tokenizer, model, model_dir)}
        for paragraph in paragraphs
    ]


def majority_baseline(records: Iterable[dict[str, str]]) -> tuple[str, float]:
    rows = list(records)
    counts = Counter(row["label"] for row in rows)
    predicted = counts.most_common(1)[0][0]
    accuracy = sum(row["label"] == predicted for row in rows) / len(rows)
    return predicted, accuracy


# Backwards-compatible milestone entry points used by the drill.
def milestone_1() -> list[dict[str, str]]:
    return load_hr_dataset()


def milestone_2(model_name: str = "gpt2"):
    return extend_gpt2_tokenizer(model_name)


def milestone_3(output_dir: str | Path = DEFAULT_MODEL_DIR, model_name: str = "gpt2"):
    return train_hr_classifier(output_dir, model_name, epochs=3.0)


def milestone_4(text: str, tokenizer=None, model=None, model_dir: str | Path = DEFAULT_MODEL_DIR) -> str:
    return classify_text(text, tokenizer, model, model_dir)


def milestone_5(text: str, tokenizer=None, model=None, model_dir: str | Path = DEFAULT_MODEL_DIR):
    return classify_paragraphs(text, tokenizer, model, model_dir)
