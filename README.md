# Document QA
This demo uses a LLM model to create text embeddings for a document and then uses these embeddings to find similar documents. The similar documents are then used to answer questions

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
gradio gradio-app.py
```

- Notebook
```bash
jupyter-lab
```

## Run Server

```bash
/bin/bash start.sh
```

## Call Services

- Load files
```bash
curl -F 'document=@docs/state_of_the_union.txt' http://127.0.0.1:3000/load_file
```

- Query database
```bash
curl http://127.0.0.1:3000/query_docs -H 'Content-Type: application/json'  -d '{"question": "why nato was created?"}'
```
