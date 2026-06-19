# Baseline Results

Milestone 4 zero-shot baseline using Groq `llama-3.3-70b-versatile`.

Test examples: 36  
Parseable responses: 36/36  
Accuracy: 0.583

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `evidence_based` | 1.00 | 0.23 | 0.38 | 13 |
| `reasoned_opinion` | 0.59 | 0.67 | 0.62 | 15 |
| `low_substance` | 0.50 | 1.00 | 0.67 | 8 |
| macro avg | 0.70 | 0.63 | 0.56 | 36 |
| weighted avg | 0.72 | 0.58 | 0.54 | 36 |

## Reflection

The baseline parsed every response successfully, so the prompt format is stable
enough for comparison. Its main weakness is `evidence_based` recall: it only
identified 23% of true `evidence_based` comments, even though the predictions it
did make for that label were correct. It also over-predicted `low_substance`,
catching every true low-substance comment but mislabeling some higher-substance
comments as low-substance.

My hypothesis is that the zero-shot model is relying on surface tone and obvious
evidence markers. Fine-tuning should help it learn the project-specific boundary
between concrete evidence, plausible reasoning, and unsupported reaction.
