import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.docstore.document import Document
from langchain.document_loaders import OnlinePDFLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from flask import Flask, render_template, request, redirect, url_for, jsonify

flask = Flask(__name__)

chatbot = None
db = None
qa = None
chuck_size = 300
embeddings = HuggingFaceHubEmbeddings()
llm = HuggingFaceHub(repo_id="google/flan-ul2", model_kwargs={"temperature":0.01, "max_new_tokens":300})


@flask.route('/load_file', methods=['POST'])
def load_file():
    global db, qa
    try:
        file = request.files['document']
        file_name = 'docs/' + file.filename
        
        persist_directory ='docs/db_' + file.filename
        
        if not os.path.isdir(persist_directory):
            file.save(file_name)
        
        loader = None
        
        if file_name.endswith('.txt'):
            loader = TextLoader(file_name)
        
        if file_name.endswith('.pdf'):
            loader = OnlinePDFLoader(file_name)

        if file_name.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(file_name)

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
    except:
        return jsonify({"status": "file not loaded"})

    return jsonify({"status": "file loaded"})


@flask.route('/query_docs', methods=['POST'])
def query_docs():
    global chatbot, db, qa

    question = request.get_json()['question']
    
    if db is None:
        return jsonify({"response" : "No documents"})

    result = qa({"query": question})

    docs = result['source_documents']
        
    if len(docs) == 0:
        return jsonify({"response" : "Sorry, I don't know the answer"})

    return jsonify({"response" : result['result']})

if __name__ == '__main__':
    flask.run(host='0.0.0.0', port=3000)