# AI201 Project 3 Submission Checklist

## Before Colab

- [ ] Read `planning.md`.
- [ ] Collect public HN comments:
  `uv run takemeter-collect-hn --target 240`.
- [ ] Create the review queue:
  `uv run python -m ai201_project3_takemeter.heuristic_prelabler`.
- [ ] Manually review every row.
- [ ] Save final dataset as `data/labeled/takemeter_hn_labeled.csv`.
- [ ] Add at least 3 hard examples to `planning.md`.
- [ ] Run `uv run takemeter-validate-dataset`.

## In Colab

- [ ] Make a copy of the TakeMeter starter notebook.
- [ ] Set runtime to T4 GPU.
- [ ] Section 1: define label map and upload final CSV.
- [ ] Section 2: split/tokenize and confirm distribution.
- [ ] Section 5: add Groq key and baseline prompt from `baseline_prompt.md`.
- [ ] Section 3: fine-tune DistilBERT.
- [ ] Section 4: evaluate fine-tuned model and inspect wrong predictions.
- [ ] Section 6: export comparison results.

## After Colab

- [ ] Download `evaluation_results.json` into `reports/`.
- [ ] Download `confusion_matrix.png` into `reports/`.
- [ ] Fill README metrics and confusion matrix markdown table.
- [ ] Analyze at least 3 wrong predictions.
- [ ] Add 3-5 sample classifications with confidence.
- [ ] Fill reflection and AI usage sections.
- [ ] Run local interface for demo:
  `uv sync --extra inference --all-groups && uv run python app.py`.
- [ ] Record 3-5 minute demo video.
- [ ] Submit GitHub repo link and demo video in Course Portal.
