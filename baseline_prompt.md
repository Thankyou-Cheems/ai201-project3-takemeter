# Groq Baseline Prompt

Use this prompt in the Colab notebook baseline section with
`llama-3.3-70b-versatile`.

```text
You are classifying Hacker News comments by discourse quality.

Choose exactly one label:

evidence_based: A comment that makes a clear claim and supports it with concrete
evidence, technical detail, numbers, examples, or a specific comparison.

reasoned_opinion: A comment that states a position and gives understandable
reasoning, but relies mainly on judgment, experience, or plausibility rather
than verifiable evidence.

low_substance: A comment that is mostly a joke, dismissal, agreement, emotional
reaction, or unsupported assertion with little reusable reasoning.

Decision rules:
- If a comment includes a statistic or example but does not use it to support a
  claim, choose reasoned_opinion.
- If a short comment gives a concrete reason, choose reasoned_opinion instead of
  low_substance.
- Do not judge whether you agree with the comment. Judge the support structure.

Return only the label name and no other text.

Comment:
{text}
```
