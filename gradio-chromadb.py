import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import OnlinePDFLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import gradio as gr

chatbot = None
db = None
qa = None
chuck_size = 300
embeddings = HuggingFaceHubEmbeddings()
llm = HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-1-pythia-12b", model_kwargs={"temperature": 0.01, "max_new_tokens":300})


def upload_document_and_create_text_bindings(file):
    global db, qa

    file_name = file.name.split('/')[-1]
    file_path = file.name
    persist_directory = 'db_' + file_name

    loader = None
    
    if file_name.endswith('.txt'):
        loader = TextLoader(file_path)
    
    if file_name.endswith('.pdf'):
        loader = OnlinePDFLoader(file_path)

    if file_name.endswith('.docx'):
        loader = UnstructuredWordDocumentLoader(file_path)

    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=chuck_size, chunk_overlap=50, separator='\n')
    split_docs = text_splitter.split_documents(documents)

    if os.path.isdir(persist_directory):
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        db = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_directory)
        db.persist()
    
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    return 'file-loaded.txt'


def analyze_question(question):
    global chatbot, db, qa

    if db is None:
        return "Please upload a document first"

    result = qa({"query": question})

    docs = result['source_documents']
        
    if len(docs) == 0:
        return "Sorry, I don't know the answer"

    return result['result']


with gr.Blocks(title='Document QA with OpenAssistant') as demo:
    gr.Markdown("# Document QA with OpenAssistant")
    gr.Markdown("This demo uses the OpenAssistant model to create text embeddings for a document and then uses these embeddings to find similar documents. The similar documents are then used to answer questions")
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
