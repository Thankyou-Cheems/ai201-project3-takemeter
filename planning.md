# TakeMeter Planning

## Community

I chose Hacker News comments. Hacker News is a strong fit for this classifier
because its public discussions are text-heavy and include a wide range of
discourse quality: technical analysis, experience-based judgment, quick
reactions, jokes, and unsupported dismissals. The distinction matters to regular
participants because the community tends to reward comments that explain a
technical or product claim instead of simply reacting to it.

The data will come from public comments collected through the public HN Search
API. I will not use private messages, authenticated spaces, or comments from
closed communities.

## Labels

The classifier uses three labels. The labels are mutually exclusive and focus on
the support structure of the comment rather than whether the comment is correct.

### `evidence_based`

A comment that makes a clear claim and supports it with concrete evidence,
technical detail, numbers, examples, or a specific comparison.

Clear examples:

- "SQLite is a good fit here because the workload is mostly local reads, and the
  deployment avoids a network hop that would dominate latency."
- "The article's benchmark is misleading: it compares cold-start Node against a
  warmed Rust binary and never reports p95 latency."

### `reasoned_opinion`

A comment that states a position and gives understandable reasoning, but relies
mainly on judgment, experience, or plausibility rather than verifiable evidence.

Clear examples:

- "I would still choose boring server-side rendering for this because the team is
  small and the maintenance cost matters more than novelty."
- "This sounds useful for prototypes, but I would not put it in a billing path
  until the failure modes are easier to inspect."

### `low_substance`

A comment that is mostly a joke, dismissal, agreement, emotional reaction, or
unsupported assertion with little reusable reasoning.

Clear examples:

- "This is just hype. Nobody serious would use it."
- "Finally someone said it. This whole thing is ridiculous."

## Hard Edge Cases

The hardest boundary is between `evidence_based` and `reasoned_opinion`. A
comment may mention a number, benchmark, or example but use it only as decoration
instead of reasoning. Decision rule: label it `evidence_based` only if the
evidence directly supports the claim and another reader could inspect or verify
the evidence. Otherwise, label it `reasoned_opinion`.

The second hard boundary is between `reasoned_opinion` and `low_substance`.
Short comments can still contain a real reason. Decision rule: if a short
comment gives a concrete reason that could help another reader evaluate the
claim, label it `reasoned_opinion`; if it only agrees, dismisses, jokes, or
asserts, label it `low_substance`.

Hard examples to fill during annotation:

1. TODO: paste a real ambiguous comment here and explain the final decision.
2. TODO.
3. TODO.

## Data Collection Plan

I will collect at least 240 public Hacker News comments, expecting to discard
some duplicates, off-topic fragments, or comments that are too short to classify
fairly. The final labeled dataset must contain at least 200 examples.

Commands:

```bash
uv run takemeter-collect-hn --target 240 --output data/raw/hackernews_comments.csv
uv run python -m ai201_project3_takemeter.heuristic_prelabler \
  --input data/raw/hackernews_comments.csv \
  --output data/labeled/takemeter_hn_review_queue.csv
```

I will manually review every row in the review queue, correct labels, and save
the final file as `data/labeled/takemeter_hn_labeled.csv`.

Target distribution: no label above 70% of the dataset. If one label is
underrepresented after 200 examples, I will collect more comments and search for
examples that clearly fit the underrepresented label before moving to training.

## Evaluation Metrics

Accuracy is useful but not enough because an imbalanced dataset could make a
majority-label classifier look deceptively strong. I will report:

- Overall accuracy for the zero-shot baseline and fine-tuned model.
- Per-class precision, recall, and F1 so I can see which label boundaries fail.
- A confusion matrix to identify directional mistakes, such as
  `reasoned_opinion` being predicted as `evidence_based`.
- At least 3 wrong predictions with written analysis.

F1 is the most important per-label metric because the labels are subjective and
each class needs both precision and recall to be useful.

## Definition of Success

For this classifier to be useful as a real community tool, the fine-tuned model
should beat the zero-shot baseline on overall accuracy and achieve at least 0.70
F1 on each label. For a minimal acceptable deployment, it should reach at least
0.65 overall accuracy and no class should have F1 below 0.55.

If the fine-tuned model performs worse than the baseline, I will treat that as a
failure worth analyzing rather than hiding. Likely causes would be noisy labels,
too few examples for one class, or a label boundary that is too subtle for 200
examples.

## AI Tool Plan

### Label stress-testing

I will ask an AI tool to generate boundary cases between
`evidence_based`/`reasoned_opinion` and `reasoned_opinion`/`low_substance`.
If the generated cases are hard to classify, I will refine the decision rules
before final annotation.

### Annotation assistance

The repo includes a heuristic pre-labeler to speed up review. These labels are
not final. I will review and correct every row, keep notes for hard examples,
and disclose the pre-labeling in the README AI usage section.

### Failure analysis

After Colab evaluation, I will paste the wrong predictions into an AI tool and
ask it to identify common patterns. I will verify those patterns manually by
re-reading the examples before including them in the final evaluation report.

## Implementation Notes

Local code is managed with `uv`. Colab remains the training environment because
the course starter notebook already provides the GPU setup, DistilBERT training
loop, baseline comparison, and output files.

The local Gradio interface is a stretch feature and demo helper. It expects the
fine-tuned model export to be copied into `models/takemeter/`.
