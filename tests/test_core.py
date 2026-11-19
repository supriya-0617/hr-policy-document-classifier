"""Tests for the HR classifier milestones."""

from collections import Counter

import pytest

from app import core


def test_milestone_1_dataset_is_complete_and_labelled():
    records = core.load_hr_dataset()
    train, validation = core.split_hr_dataset(records)
    assert len(records) == 125
    assert len(train) == 100
    assert len(validation) == 25
    assert set(record["label"] for record in records) == set(core.CATEGORIES)
    assert all(record["text"].strip() for record in records)
    for category in core.CATEGORIES:
        assert sum(record["label"] == category for record in train) == 20
        assert sum(record["label"] == category for record in validation) == 5


def test_milestone_2_category_tokens_are_unique_and_complete():
    assert len(core.CATEGORY_TOKENS) == 5
    assert set(core.CATEGORY_TOKENS) == set(core.CATEGORIES)
    assert len(set(core.CATEGORY_TOKENS.values())) == 5
    assert all(token.startswith("<HR_") and token.endswith(">") for token in core.CATEGORY_TOKENS.values())


def test_milestone_2_training_preparation_masks_prompt(monkeypatch):
    class MockTokenizer:
        vocab = {
            "<HR_LEAVE_POLICY>": 100,
            "<HR_EXPENSE_CLAIM>": 101,
        }

        def convert_tokens_to_ids(self, token):
            return self.vocab[token]

        def __call__(self, text, truncation, max_length, padding, return_tensors):
            category_id = self.vocab[next(token for token in self.vocab if token in text)]
            ids = list(range(3)) + [category_id]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    dataset = core.prepare_training_data(
        [{"text": "A leave policy paragraph.", "label": "Leave Policy", "split": "train"}],
        MockTokenizer(),
    )
    row = dataset[0]
    assert row["labels"][:3] == [-100, -100, -100]
    assert row["labels"][3] == 100


def test_milestone_3_trainer_configuration_is_used(monkeypatch):
    observed = {}

    class MockTokenizer:
        def save_pretrained(self, path):
            observed["saved"] = path

    class MockModel:
        def parameters(self):
            return []

        class Embeddings:
            class Weight:
                requires_grad = False
            weight = Weight()

        def get_input_embeddings(self):
            return self.Embeddings()

    class MockTrainer:
        def __init__(self, **kwargs):
            observed["trainer"] = kwargs

        def train(self):
            return None

        def evaluate(self):
            return {"eval_loss": 0.25}

        def save_model(self, path):
            observed["model"] = path

    monkeypatch.setattr(core, "split_hr_dataset", lambda records: (records[:100], records[100:]))
    monkeypatch.setattr(core, "extend_gpt2_tokenizer", lambda model_name: (MockTokenizer(), MockModel()))
    monkeypatch.setattr(core, "prepare_training_data", lambda records, tokenizer: records)

    class MockArgs:
        def __init__(self, **kwargs):
            observed["args"] = kwargs

    monkeypatch.setattr(core, "Trainer", MockTrainer)
    monkeypatch.setattr(core, "TrainingArguments", MockArgs)
    result = core.train_hr_classifier("/tmp/hr-gpt2-test")
    assert result["train_examples"] == 100
    assert result["validation_examples"] == 25
    assert observed["args"]["eval_strategy"] == "epoch"
    assert observed["trainer"]["train_dataset"] is not None


def test_milestone_4_invalid_and_paragraph_inputs():
    with pytest.raises(ValueError, match="empty"):
        core.classify_text(" ")
    assert core.split_paragraphs("") == []
    assert core.split_paragraphs("First section.\n\nSecond section.") == ["First section.", "Second section."]
    assert core.split_paragraphs("One line\nSecond line") == ["One line", "Second line"]


def test_inference_decodes_generated_hr_token():
    import torch

    class MockTokenizer:
        eos_token_id = 0

        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

        def convert_tokens_to_ids(self, token):
            return list(core.CATEGORY_TOKENS.values()).index(token) + 10

    class MockModel:
        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1))])

        def generate(self, input_ids, prefix_allowed_tokens_fn, **kwargs):
            allowed = prefix_allowed_tokens_fn(0, input_ids[0])
            assert allowed == list(range(10, 15))
            return torch.tensor([[1, 2, 12]])

    assert core.classify_text("A never-seen policy paragraph.", MockTokenizer(), MockModel()) == "Recruitment"


def test_milestone_5_paragraph_processing(monkeypatch):
    monkeypatch.setattr(core, "classify_text", lambda text, tokenizer=None, model=None, model_dir=None: "Leave Policy")
    results = core.classify_paragraphs("Annual leave rules.\n\nBenefit enrollment.")
    assert results == [
        {"text": "Annual leave rules.", "category": "Leave Policy"},
        {"text": "Benefit enrollment.", "category": "Leave Policy"},
    ]


def test_majority_baseline_is_real_and_deterministic():
    records = [{"label": "Leave Policy"}, {"label": "Expense Claim"}, {"label": "Leave Policy"}]
    predicted, accuracy = core.majority_baseline(records)
    assert predicted == "Leave Policy"
    assert accuracy == pytest.approx(2 / 3)
