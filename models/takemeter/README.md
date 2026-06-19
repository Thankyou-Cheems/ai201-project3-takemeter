# Fine-Tuned Model Export

After Colab training, copy the exported Hugging Face model files into this
directory if you want to run the local Gradio interface.

The app expects files such as:

- `config.json`
- `model.safetensors` or `pytorch_model.bin`
- tokenizer files such as `tokenizer.json`, `vocab.txt`, and
  `tokenizer_config.json`

Then run:

```bash
uv sync --extra inference --all-groups
uv run python app.py
```
