# HR Policy Document Classifier

This project fine-tunes GPT-2 to classify internal HR policy paragraphs into five categories:

- Leave Policy
- Expense Claim
- Recruitment
- Employee Benefits
- Workplace Conduct

The Streamlit interface accepts a UTF-8 text file, separates it into paragraphs, and uses the fine-tuned model to classify every section.

## Dataset

The manually labelled dataset is `data/hr_policy_examples.csv`. It contains 125 unique, original paragraphs: 100 for training and 25 held out for validation, with 20 training and 5 validation examples per category. The dataset was expanded by 75 realistic examples to provide category balance and wording variety; no examples were duplicated. The `split` column keeps the sets distinct, and the loader rejects validation text found in training.

## GPT-2 extension

`app/core.py` loads `gpt2` with `AutoTokenizer` and `AutoModelForCausalLM`, then adds these special tokens:

- `<HR_LEAVE_POLICY>`
- `<HR_EXPENSE_CLAIM>`
- `<HR_RECRUITMENT>`
- `<HR_EMPLOYEE_BENEFITS>`
- `<HR_WORKPLACE_CONDUCT>`

It resizes GPT-2's input and output embedding matrix after adding the tokens. Each newly added token embedding is initialized from the mean of the base GPT-2 subword embeddings for its category words. Training formats each example as a classification instruction, paragraph, and category token; prompt and padding positions are masked with `-100`, so Trainer optimizes the category-token next-token prediction.

## Training

Training uses Hugging Face `Trainer` and `TrainingArguments`. The reproducible CPU-friendly run uses GPT-2, 20 epochs, batch size 4, gradient accumulation 2, learning rate `1e-3`, weight decay `0.01`, warmup ratio `0.05`, sequence length 128, and seed 42. The GPT-2 transformer layers remain frozen while the resized token embedding matrix is fine-tuned. Checkpoints are evaluated each epoch, and the lowest-validation-loss checkpoint is saved with the tokenizer to `models/hr-gpt2/best/`.

```bash
python -m app.train
```

The script repeats the held-out validation evaluation and writes `data/evaluation.json`.

## Inference

`classify_text()` uses the same instruction prompt as training and generates one next token from a constrained set containing only the five HR category tokens. It converts the generated special token to its readable label and raises an explicit error if no recognized token is generated. Streamlit uses this saved-model inference for each uploaded paragraph.

## Installation

Use Python 3.12:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` only if using the optional LLM settings; the GPT-2 classifier does not require an API key.

## Run

```bash
streamlit run app/main.py
```

## Tests

```bash
pytest -q
```

Unit tests cover the dataset, labels, special-token setup, masked training data, Trainer configuration, paragraph processing, inference input validation, and the majority baseline. The suite uses mocks rather than downloading or training GPT-2.

## Evaluation and baseline

The latest saved-model GPT-2 run was measured on all 25 held-out paragraphs, without training on their text. It achieved **84% accuracy (21/25)** and **0.842 macro-F1**, compared with the historical 20% result and the current balanced-set majority-class baseline of **20% (5/25)**, which predicts Leave Policy. Per-category precision / recall / F1:

| Category | Precision | Recall | F1 |
|---|---:|---:|---:|
| Leave Policy | 0.80 | 0.80 | 0.80 |
| Expense Claim | 1.00 | 0.60 | 0.75 |
| Recruitment | 1.00 | 0.80 | 0.89 |
| Employee Benefits | 0.63 | 1.00 | 0.77 |
| Workplace Conduct | 1.00 | 1.00 | 1.00 |

Confusion matrix (rows are actual; columns are predicted, in the category order above):

```text
[[4, 0, 0, 1, 0],
 [0, 3, 0, 2, 0],
 [1, 0, 4, 0, 0],
 [0, 0, 0, 5, 0],
 [0, 0, 0, 0, 5]]
```

The original pipeline used only 40 training paragraphs, trained the full GPT-2 model for three epochs, and reached 20% accuracy while predicting Recruitment for every validation item. Experiments with five and ten epochs on the expanded set still collapsed to one class. The successful run initialized each new category embedding using its base GPT-2 subwords and trained the resized embeddings only; 20 epochs reached a 0.065 validation loss, and constrained generation returned valid category tokens. The full actual metrics, configuration, confusion matrix, and individual predictions are in `data/evaluation.json`.

The validation set is small (five paragraphs per class) and was used for checkpoint selection, so these results are preliminary validation evidence, not an independent final test score. The result should not be generalized to operational HR decisions without a larger separately held-out dataset and human review. Ambiguous or mixed-topic passages and wording outside the examples remain failure modes.

The classifier should perform well on text that closely matches policy wording and labels represented in the dataset. It may fail on ambiguous mixed topics, highly specialized legal wording, short fragments, or categories outside the five selected labels.

## Project structure

- `app/core.py`: dataset, token extension, training, and inference
- `app/train.py`: fine-tuning and held-out evaluation entry point
- `app/main.py`: Streamlit upload interface
- `data/hr_policy_examples.csv`: labelled HR paragraphs
- `tests/test_core.py`: milestone tests

## Secrets and CI

Never commit `.env`, credentials, or private model artifacts. `.gitignore` excludes `.env`, virtual environments, caches, local data, and `models/`. GitHub Actions installs the dependencies and runs `pytest -q` without downloading or training GPT-2.
