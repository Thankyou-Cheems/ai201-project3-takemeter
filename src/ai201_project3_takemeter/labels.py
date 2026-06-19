"""Label taxonomy for the default Hacker News TakeMeter project."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelDefinition:
    """A single mutually exclusive discourse-quality label."""

    name: str
    short_name: str
    definition: str
    positive_examples: tuple[str, str]
    edge_case_rule: str


LABELS: tuple[LabelDefinition, ...] = (
    LabelDefinition(
        name="evidence_based",
        short_name="Evidence-based analysis",
        definition=(
            "A comment that makes a clear claim and supports it with concrete "
            "evidence, technical detail, numbers, examples, or a specific comparison."
        ),
        positive_examples=(
            "SQLite is a good fit here because the workload is mostly local reads, "
            "and the deployment avoids a network hop that would dominate latency.",
            "The article's benchmark is misleading: it compares cold-start Node "
            "against a warmed Rust binary and never reports p95 latency.",
        ),
        edge_case_rule=(
            "If a comment includes a statistic or example but does not use it to "
            "support a claim, label it reasoned_opinion instead."
        ),
    ),
    LabelDefinition(
        name="reasoned_opinion",
        short_name="Reasoned opinion",
        definition=(
            "A comment that states a position and gives understandable reasoning, "
            "but relies mainly on judgment, experience, or plausibility rather than "
            "verifiable evidence."
        ),
        positive_examples=(
            "I would still choose boring server-side rendering for this because "
            "the team is small and the maintenance cost matters more than novelty.",
            "This sounds useful for prototypes, but I would not put it in a billing "
            "path until the failure modes are easier to inspect.",
        ),
        edge_case_rule=(
            "If the comment has a rationale but no concrete evidence that another "
            "reader could verify, keep it here rather than evidence_based."
        ),
    ),
    LabelDefinition(
        name="low_substance",
        short_name="Low-substance reaction",
        definition=(
            "A comment that is mostly a joke, dismissal, agreement, emotional "
            "reaction, or unsupported assertion with little reusable reasoning."
        ),
        positive_examples=(
            "This is just hype. Nobody serious would use it.",
            "Finally someone said it. This whole thing is ridiculous.",
        ),
        edge_case_rule=(
            "If a comment is short but still gives a concrete reason, label it "
            "reasoned_opinion; short length alone is not enough for low_substance."
        ),
    ),
)

LABEL_NAMES: tuple[str, ...] = tuple(label.name for label in LABELS)
LABEL_MAP: dict[str, int] = {name: index for index, name in enumerate(LABEL_NAMES)}
ID_TO_LABEL: dict[int, str] = {index: name for name, index in LABEL_MAP.items()}


def label_definitions_markdown() -> str:
    """Render the taxonomy for prompts, planning docs, and README sections."""
    sections = []
    for label in LABELS:
        examples = "\n".join(f"  - {example}" for example in label.positive_examples)
        sections.append(
            f"### `{label.name}`: {label.short_name}\n"
            f"{label.definition}\n\n"
            f"Examples:\n{examples}\n\n"
            f"Decision rule: {label.edge_case_rule}"
        )
    return "\n\n".join(sections)
