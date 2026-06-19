# TakeMeter: Hacker News Discourse Classifier

AI201 Project 3. TakeMeter is a text classifier for Hacker News comments. It
labels a comment as `evidence_based`, `reasoned_opinion`, or `low_substance`
so we can study whether a fine-tuned model learns discourse quality distinctions
that regular Hacker News readers would recognize.

Project due date from the course page: Monday, June 22, 2026 at 2:59 PM GMT+8.

## Setup

```bash
cd /home/cheems/dev/AI201/Week3/ai201-project3-takemeter
uv sync --all-groups
```

Run checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Community Choice

The chosen community is Hacker News. It is a good fit because the comments are
public, text-heavy, and varied: some comments contain technical evidence, some
make plausible but experience-based arguments, and some are mostly reactions or
dismissals. These distinctions matter in the community because Hacker News users
often value detailed technical reasoning over unsupported hot takes.

## Label Taxonomy

The labels are mutually exclusive and ordered by the kind of support the comment
provides, not by whether I agree with the comment.

### `evidence_based`

A comment that makes a clear claim and supports it with concrete evidence,
technical detail, numbers, examples, or a specific comparison.

Examples:

- SQLite is a good fit here because the workload is mostly local reads, and the
  deployment avoids a network hop that would dominate latency.
- The article's benchmark is misleading: it compares cold-start Node against a
  warmed Rust binary and never reports p95 latency.

Decision rule: if a comment includes a statistic or example but does not use it
to support a claim, label it `reasoned_opinion` instead.

### `reasoned_opinion`

A comment that states a position and gives understandable reasoning, but relies
mainly on judgment, experience, or plausibility rather than verifiable evidence.

Examples:

- I would still choose boring server-side rendering for this because the team is
  small and the maintenance cost matters more than novelty.
- This sounds useful for prototypes, but I would not put it in a billing path
  until the failure modes are easier to inspect.

Decision rule: if the comment has a rationale but no concrete evidence that
another reader could verify, keep it here rather than `evidence_based`.

### `low_substance`

A comment that is mostly a joke, dismissal, agreement, emotional reaction, or
unsupported assertion with little reusable reasoning.

Examples:

- This is just hype. Nobody serious would use it.
- Finally someone said it. This whole thing is ridiculous.

Decision rule: if a comment is short but still gives a concrete reason, label it
`reasoned_opinion`; short length alone is not enough for `low_substance`.

## Data Collection and Labeling

The data source is public Hacker News comments collected through the public HN
Search API. No private channels, login-only content, or scraped personal data
are used.

Collect a review queue:

```bash
uv run takemeter-collect-hn --target 240 --output data/raw/hackernews_comments.csv
uv run python -m ai201_project3_takemeter.heuristic_prelabler \
  --input data/raw/hackernews_comments.csv \
  --output data/labeled/takemeter_hn_review_queue.csv
```

Then manually review every row in `data/labeled/takemeter_hn_review_queue.csv`.
Correct the labels, write notes for at least 3 genuinely hard examples, set
`review_status` to `reviewed`, and save the final file as:

```text
data/labeled/takemeter_hn_labeled.csv
```

Check the rough queue distribution before review:

```bash
uv run takemeter-validate-dataset \
  data/labeled/takemeter_hn_review_queue.csv \
  --allow-unreviewed
```

Validate before uploading to Colab:

```bash
uv run takemeter-validate-dataset data/labeled/takemeter_hn_labeled.csv
```

Target distribution: at least 200 labeled examples, with no single label above
70% of the dataset. The preferred target is roughly 30-45% per label.

## Fine-Tuning Approach

The planned base model is `distilbert-base-uncased`, trained in the course
starter Colab notebook with the labeled CSV above. The notebook handles the
70/15/15 train-validation-test split, tokenization, training loop, metrics, and
confusion matrix.

Default hyperparameters from the course notebook:

- Epochs: 3
- Learning rate: 2e-5
- Batch size: 16

Hyperparameter decision: start with the defaults because the dataset is small
and subjective. If validation performance is unstable, adjust only one variable
at a time and document the change here.

## Zero-Shot Baseline

The baseline model is Groq `llama-3.3-70b-versatile`, run on the same locked
test set created by the Colab notebook. The prompt must include the label
definitions above and instruct the model to output only one label name:
`evidence_based`, `reasoned_opinion`, or `low_substance`.

Use `baseline_prompt.md` as the prompt source for Colab Section 5.

## Evaluation Report

Fill this section after running Colab Sections 4 and 6. Commit
`reports/evaluation_results.json` and `reports/confusion_matrix.png` after
downloading them from Colab.

### Metrics

| Model | Accuracy | Notes |
| --- | ---: | --- |
| Zero-shot Groq baseline | TODO | Fill from Colab Section 5 |
| Fine-tuned DistilBERT | TODO | Fill from Colab Section 6 |

### Per-Class Metrics

| Model | Label | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| Baseline | evidence_based | TODO | TODO | TODO |
| Baseline | reasoned_opinion | TODO | TODO | TODO |
| Baseline | low_substance | TODO | TODO | TODO |
| Fine-tuned | evidence_based | TODO | TODO | TODO |
| Fine-tuned | reasoned_opinion | TODO | TODO | TODO |
| Fine-tuned | low_substance | TODO | TODO | TODO |

### Fine-Tuned Confusion Matrix

Write the final matrix as a markdown table here, not only as an image.

| True \ Predicted | evidence_based | reasoned_opinion | low_substance |
| --- | ---: | ---: | ---: |
| evidence_based | TODO | TODO | TODO |
| reasoned_opinion | TODO | TODO | TODO |
| low_substance | TODO | TODO | TODO |

### Failure Analysis

Add at least 3 wrong predictions after Colab evaluation.

1. TODO: quote or summarize the comment, true label, predicted label, and why it
   failed.
2. TODO.
3. TODO.

### Sample Classifications

Run 3-5 new comments through the fine-tuned model and record the output.

| Comment summary | Predicted label | Confidence | Reasonable? |
| --- | --- | ---: | --- |
| TODO | TODO | TODO | TODO |

## Local Interface

After Colab training, export or download the fine-tuned Hugging Face model files
into `models/takemeter/`. Then install inference dependencies and launch the UI:

```bash
uv sync --extra inference --all-groups
uv run python app.py
```

Open the Gradio URL printed in the terminal. The interface accepts a new comment
and displays the predicted label plus confidence scores.

## Reflection

TODO after evaluation: explain what the model actually learned versus what the
label taxonomy intended it to learn. Look for shortcuts such as comment length,
technical vocabulary, or emotional wording.

## Spec Reflection

The spec helped by forcing the label decision rules before training. That makes
the annotation process less subjective and gives the evaluation report a clear
standard for failure analysis.

One planned divergence is using the Hacker News Firebase API and local `uv`
helper scripts outside Colab. The starter notebook still handles training, but
the local scripts make data collection, review, and validation repeatable.

## AI Usage

1. Codex helped scaffold the `uv` project, extract the assignment requirements
   from the Week 3 HTML page, and create helper scripts. I reviewed the generated
   project structure and kept the pieces that match the assignment.
2. Codex generated a heuristic pre-labeling script. These are not final labels:
   every row must be manually reviewed and corrected before submission. Any
   pre-labeling assistance should be disclosed in the final report.

## Submission Checklist

- GitHub repository link.
- `planning.md` in the repo root.
- Final labeled dataset at `data/labeled/takemeter_hn_labeled.csv`.
- `README.md` completed with metrics, confusion matrix table, failures, sample
  classifications, reflection, spec reflection, and AI usage.
- `reports/evaluation_results.json` and `reports/confusion_matrix.png`.
- 3-5 minute demo video showing classifications, one correct prediction, one
  incorrect prediction, and the evaluation report.
