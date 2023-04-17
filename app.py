from langchain.llms import LlamaCpp
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
import os
import gradio as gr


chatbot = None
db = None
chuck_size = 500
max_num_of_tokens = 2048
model = "gpt4all-lora-quantized-ggml.bin"


def upload_document_and_create_text_bindings(file):
    global db

    file_name = file.name.split('/')[-1]
    file_path = file.name
    persist_directory = 'db_' + file_name

    llm_embeddings = LlamaCppEmbeddings(model_path=model)

    if os.path.isdir(persist_directory):
        db = Chroma(persist_directory=persist_directory, embedding_function=llm_embeddings)
        return "state_of_the_union.txt"

    loader = TextLoader(file_path)

    #if file_name.endswith('.pdf'):
    #    loader = PyPDFLoader(file_path)

    if file_name.endswith('.docx'):
        loader = UnstructuredWordDocumentLoader(file_path)

    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=chuck_size, chunk_overlap=100, separator='\n')
    split_docs = text_splitter.split_documents(documents)

    if len(split_docs) > 20:
        raise "Document-is-to-big.txt"

    db = Chroma.from_documents(split_docs, llm_embeddings, persist_directory=persist_directory)
    db.persist()

    return "state_of_the_union.txt"


def analyze_question(question):
    global chatbot
    global db

    if db is None:
        return "Please upload a document first"

    llm = LlamaCpp(model_path=model, n_ctx=max_num_of_tokens)
    chain = load_qa_chain(llm, chain_type="stuff")

    docs = db.similarity_search(question)
    if len(docs) == 0:
        return "Sorry, I don't know the answer"

    responses = []
    for rdoc in docs:
        responses.append(Document(page_content=chain.run(input_documents=[rdoc], question=question),  metadata=rdoc.metadata))

    chain = load_qa_chain(llm, chain_type="stuff")

    while len(responses) != 1:
        c = 0
        length_context = 0
        for i in range(len(responses)):
            count_tokens = 40 + len(question.split(' ')) + length_context + len(responses[i].page_content.split(' '))
            if max_num_of_tokens > count_tokens:
                length_context += len(responses[i].page_content.split(' '))
                c += 1

        responses.append(Document(page_content=chain.run(input_documents=responses[:c], question=question)))
        responses = responses[c:]

    return responses[0].page_content


with gr.Blocks(title='Document QA with GPT4All') as demo:
    gr.Markdown("# Document QA with GPT4All")
    gr.Markdown("This demo uses the GPT4All model to create text embeddings for a document and then uses these embeddings to find similar documents. The similar documents are then used to answer questions")
    gr.Markdown("The document processing can take a while. Try with documents not bigger that 5000 words")
    gr.Markdown("## How to use it")
    gr.Markdown("Upload a document (docx or txt). Wait for the document to be processed and then ask a question you want. The answer will be displayed in the chatbot.")
    with gr.Row():
        with gr.Column():
            file_upload = gr.File()
            upload_button = gr.UploadButton("Select Document", file_types=["txt", "pdf", "docx"])
            upload_button.upload(upload_document_and_create_text_bindings, upload_button, file_upload)
        with gr.Column():
            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.Button("Clear")

    def user(user_message, history):
        answer = analyze_question(user_message)
        return "", history + [[user_message, answer]]

    def bot(history):
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
