# HR Policy Document Classifier

This project fine-tunes GPT-2 to classify internal HR policy paragraphs into five categories:

- Leave Policy
- Expense Claim
- Recruitment
- Employee Benefits
- Workplace Conduct

The Streamlit interface accepts a UTF-8 text file, separates it into paragraphs, and uses the fine-tuned model to classify every section.

## Dataset

The labelled dataset is `data/hr_policy_examples.csv`. It contains 50 original paragraphs: 40 for training and 10 held out for validation. Each category has 8 training examples and 2 validation examples. The `split` column prevents validation leakage.

## GPT-2 extension

`app/core.py` loads `gpt2` with `AutoTokenizer` and `AutoModelForCausalLM`, then adds these special tokens:

- `<HR_LEAVE_POLICY>`
- `<HR_EXPENSE_CLAIM>`
- `<HR_RECRUITMENT>`
- `<HR_EMPLOYEE_BENEFITS>`
- `<HR_WORKPLACE_CONDUCT>`

It resizes GPT-2's input and output embedding matrix after adding the tokens. Training formats each example as `HR Policy Text` followed by `Category:` and the category token. The prompt tokens are masked with `-100`, so Hugging Face Trainer optimizes next-token prediction for the HR category token.

## Training

Training uses `Trainer` and `TrainingArguments`. The default run is three epochs at a learning rate of `5e-5`, with two examples per device. The trained tokenizer and model are saved to `models/hr-gpt2/best/`.

```bash
python -m app.train
```

The script repeats the held-out validation evaluation and writes `data/evaluation.json`.

## Inference

`classify_text()` loads the saved tokenizer and causal language model, builds the category prompt, and scores each HR category token as GPT-2's next token. The highest-scoring category is converted back to a readable label. Streamlit uses the same function for each uploaded paragraph.

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

The latest three-epoch GPT-2 run was evaluated on the 10 held-out paragraphs. The actual classifier accuracy was **20% (2/10)** and the majority-class baseline was also **20%**, predicting Leave Policy. The run used 40 training examples and CPU training for three epochs. `data/evaluation.json` contains every held-out expected/predicted label.

The classifier should perform well on text that closely matches policy wording and labels represented in the dataset. It may fail on ambiguous mixed topics, highly specialized legal wording, short fragments, or categories outside the five selected labels.

## Project structure

- `app/core.py`: dataset, token extension, training, and inference
- `app/train.py`: fine-tuning and held-out evaluation entry point
- `app/main.py`: Streamlit upload interface
- `data/hr_policy_examples.csv`: labelled HR paragraphs
- `tests/test_core.py`: milestone tests

## Secrets and CI

Never commit `.env`, credentials, or private model artifacts. `.gitignore` excludes `.env`, virtual environments, caches, local data, and `models/`. GitHub Actions installs the dependencies and runs `pytest -q` without downloading or training GPT-2.
