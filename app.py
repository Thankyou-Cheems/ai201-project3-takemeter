"""Gradio interface for a fine-tuned TakeMeter model exported from Colab."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

DEFAULT_MODEL_DIR = Path("models/takemeter")


def _load_pipeline(model_dir: Path):
    """Load a local Hugging Face text-classification pipeline lazily."""
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise RuntimeError(
            "Install inference dependencies first: uv sync --extra inference"
        ) from exc

    if not model_dir.exists() or not any(model_dir.iterdir()):
        raise RuntimeError(
            "No exported model found. After Colab training, download/copy the "
            "fine-tuned model files into models/takemeter/."
        )
    return pipeline("text-classification", model=str(model_dir), top_k=None)


def classify_post(
    post: str, model_dir: str = str(DEFAULT_MODEL_DIR)
) -> tuple[str, dict]:
    """Classify one post and return the best label plus all confidences."""
    text = post.strip()
    if not text:
        return "Paste a post or comment to classify.", {}

    classifier = _load_pipeline(Path(model_dir))
    predictions = classifier(text)
    scores = (
        predictions[0]
        if predictions and isinstance(predictions[0], list)
        else predictions
    )
    normalized = {
        str(item["label"]): round(float(item["score"]), 4)
        for item in sorted(scores, key=lambda row: row["score"], reverse=True)
    }
    best_label, best_score = next(iter(normalized.items()))
    return f"{best_label} ({best_score:.1%})", normalized


demo = gr.Interface(
    fn=classify_post,
    inputs=[
        gr.Textbox(
            label="Post or comment",
            lines=8,
            placeholder="Paste a Hacker News comment here...",
        ),
        gr.Textbox(label="Model directory", value=str(DEFAULT_MODEL_DIR)),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.JSON(label="Confidence scores"),
    ],
    title="TakeMeter",
    description=(
        "Classifies Hacker News comments into evidence_based, reasoned_opinion, "
        "or low_substance using the fine-tuned model exported from Colab."
    ),
)


if __name__ == "__main__":
    demo.launch()
