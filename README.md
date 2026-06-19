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

Primary dataset format: JSONL. Each row is one JSON object, which is safer for
long community comments containing commas, quotes, newlines, URLs, and metadata.
The course Colab notebook expects CSV, so this repo exports a temporary
`text,label` CSV from the reviewed JSONL only when needed.

Collect a review queue:

```bash
uv run takemeter-collect-hn --target 240 --output data/raw/hackernews_comments.jsonl
uv run python -m ai201_project3_takemeter.heuristic_prelabler \
  --input data/raw/hackernews_comments.jsonl \
  --output data/labeled/takemeter_hn_review_queue.jsonl
```

Then manually review every row in `data/labeled/takemeter_hn_review_queue.jsonl`.
Correct the labels, write notes for at least 3 genuinely hard examples, set
`review_status` to `reviewed`, and save the final file as:

```text
data/labeled/takemeter_hn_labeled.jsonl
```

Check the rough queue distribution before review:

```bash
uv run takemeter-validate-dataset \
  data/labeled/takemeter_hn_review_queue.jsonl \
  --allow-unreviewed
```

Validate the reviewed JSONL:

```bash
uv run takemeter-validate-dataset data/labeled/takemeter_hn_labeled.jsonl
```

Export the Colab upload CSV from the reviewed JSONL:

```bash
uv run takemeter-export-colab-csv \
  --input data/labeled/takemeter_hn_labeled.jsonl \
  --output data/colab/takemeter_hn_labeled_for_colab.csv
```

Target distribution: at least 200 labeled examples, with no single label above
70% of the dataset. The preferred target is roughly 30-45% per label.

Current reviewed JSONL distribution:

| Label | Count | Share |
| --- | ---: | ---: |
| `reasoned_opinion` | 103 | 42.9% |
| `evidence_based` | 85 | 35.4% |
| `low_substance` | 52 | 21.7% |


## Fine-Tuning Approach

The planned base model is `distilbert-base-uncased`, trained in the course
starter Colab notebook with the exported Colab CSV. The JSONL file remains the
source of truth; the CSV is only an adapter for the notebook. The notebook
handles the 70/15/15 train-validation-test split, tokenization, training loop,
metrics, and confusion matrix.

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

Colab outputs are committed in `reports/evaluation_results.json` and
`reports/confusion_matrix.png`.

### Metrics

| Model | Accuracy | Notes |
| --- | ---: | --- |
| Zero-shot Groq baseline | 0.583 | 36/36 parseable responses |
| Fine-tuned DistilBERT | 0.417 | 3 epochs, default course hyperparameters |

### Per-Class Metrics

| Model | Label | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| Baseline | evidence_based | 1.00 | 0.23 | 0.38 |
| Baseline | reasoned_opinion | 0.59 | 0.67 | 0.62 |
| Baseline | low_substance | 0.50 | 1.00 | 0.67 |
| Fine-tuned | evidence_based | 0.36 | 0.77 | 0.49 |
| Fine-tuned | reasoned_opinion | 0.62 | 0.33 | 0.43 |
| Fine-tuned | low_substance | 0.00 | 0.00 | 0.00 |

### Fine-Tuned Confusion Matrix

| True \ Predicted | evidence_based | reasoned_opinion | low_substance |
| --- | ---: | ---: | ---: |
| evidence_based | 10 | 3 | 0 |
| reasoned_opinion | 10 | 5 | 0 |
| low_substance | 8 | 0 | 0 |

### Failure Analysis

The fine-tuned model performed worse than the zero-shot baseline: 41.7% accuracy
versus 58.3%. The dominant failure mode is visible in the confusion matrix: the
fine-tuned model never predicted `low_substance` on the test set. All 8 true
`low_substance` examples were classified as `evidence_based`.

1. `low_substance -> evidence_based`: all 8 true low-substance comments were
   predicted as evidence-based. This suggests the model learned a shortcut from
   Hacker News surface features such as technical vocabulary, named tools, or
   longer phrasing, rather than learning whether the comment actually contains
   reusable reasoning.
2. `reasoned_opinion -> evidence_based`: 10 of 15 reasoned-opinion comments were
   predicted as evidence-based. This is the same boundary problem described in
   `planning.md`: a plausible explanation or anecdote can look like evidence,
   but the taxonomy requires inspectable support that directly backs the claim.
3. `evidence_based -> reasoned_opinion`: 3 of 13 evidence-based comments were
   predicted as reasoned-opinion. These likely contain evidence that is less
   obvious than numbers, benchmarks, or links, so the model under-recognized
   support expressed through mechanisms or comparisons.

### Sample Classifications

I ran four new comments through the fine-tuned model after training:

| Comment summary | Predicted label | Confidence | Reasonable? |
| --- | --- | ---: | --- |
| Benchmark critique: cold-start numbers are compared against warm runs, so latency is not measured consistently. | `evidence_based` | 0.354 | Yes. The comment gives a specific methodological reason. |
| Production caution: the idea is useful, but debugging still sounds too fragile for a billing path. | `reasoned_opinion` | 0.345 | Yes. The comment gives plausible reasoning without concrete evidence. |
| Unsupported dismissal: "This is just hype. Nobody serious is going to use this." | `evidence_based` | 0.346 | No. This should be `low_substance`; the error matches the confusion matrix pattern where the model never predicts `low_substance`. |
| Technical correction: the README's npm command is wrong because the package is published under a different name. | `evidence_based` | 0.357 | Yes. The comment gives a concrete, checkable reason. |

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

The model learned some distinction between `evidence_based` and
`reasoned_opinion`, but it did not learn the intended three-way taxonomy. It
collapsed `low_substance` into `evidence_based`, which means it likely overfit
to Hacker News style cues: technical wording, specificity, or confident tone.
The intended behavior was to separate concrete support from plausible reasoning
and unsupported reaction. The learned behavior appears closer to "technical or
specific-sounding comments are evidence-based," even when the comment lacks a
real argument.

To improve this, I would collect more clear `low_substance` examples from Hacker
News, especially low-substance comments that still mention technical topics, and
add more hard negatives where tool names, numbers, or links appear without
actually supporting a claim.

## Spec Reflection

The spec helped by forcing the label decision rules before training. That makes
the annotation process less subjective and gives the evaluation report a clear
standard for failure analysis.

One planned divergence is using JSONL as the source dataset format with local
`uv` helper scripts outside Colab. The starter notebook still handles training,
but the local scripts make data collection, review, validation, and Colab CSV
export repeatable.

## AI Usage

1. Codex helped scaffold the `uv` project, extract the assignment requirements
   from the Week 3 HTML page, and create helper scripts. I reviewed the generated
   project structure and kept the pieces that match the assignment.
2. Codex generated a heuristic pre-labeling script. These are not final labels:
   every row must be manually reviewed and corrected before submission. Any
   pre-labeling assistance should be disclosed in the final report.
3. Codex ran four parallel sub-agents to review the 240 rough labels by shard.
   I merged their suggested corrections into `takemeter_hn_labeled.jsonl` and
   kept `review_status=reviewed_ai_assisted` so the dataset history is explicit.

## Submission Checklist

- GitHub repository link.
- `planning.md` in the repo root.
- Final labeled dataset at `data/labeled/takemeter_hn_labeled.jsonl`.
- Temporary local Colab upload CSV at
  `data/colab/takemeter_hn_labeled_for_colab.csv` generated from JSONL. This is
  ignored by git unless the grader explicitly asks for it.
- `README.md` completed with metrics, confusion matrix table, failures, sample
  classifications, reflection, spec reflection, and AI usage.
- `reports/evaluation_results.json` and `reports/confusion_matrix.png`.
- 3-5 minute demo video showing classifications, one correct prediction, one
  incorrect prediction, and the evaluation report.
