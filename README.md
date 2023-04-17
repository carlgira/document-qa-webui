# Document QA with GPT4All
This demo uses the GPT4All model to create text embeddings for a document and then uses these embeddings to find similar documents. The similar documents are then used to answer questions

- Download the model 
```bash
wget https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/gpt4all-lora-quantized-ggml.bin
```

- Install the dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Test

- Gradio:
Use the files state_of_the_union.txt or state_of_the_union.docx to test the gradio app.
```bash
gradio app.py
```

- Notebook
```bash
jupyter-lab
```

