#importing modules
import openai
import os
import yaml,json
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.llms import AzureOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from flask import Flask,request
from flask_cors import CORS

app=Flask(__name__)
CORS(app)

config=yaml.safe_load(open(r'C:\Users\KMNandee\OneDrive - Molina Healthcare\Desktop\project\vector_search\data\config.yml'))

OPENAI_API_TYPE = config['OPENAI_API_TYPE']
OPENAI_API_BASE = config['OPENAI_API_BASE']
OPENAI_API_KEY = config['OPENAI_API_KEY']
OPENAI_API_VERSION =config['OPENAI_API_VERSION']
embedding_model_name = config['embedding_model_name']
embedding_model_deployment_name = config['embedding_model_deployment_name']
text_gen_model = config['text_gen_model']
text_gen_model_deployment_name = config['text_gen_model_deployment_name']
vector_store_address=config['vector_store_address']
vector_store_password=config['vector_store_password']
index_name=config['index_name']
#synonymap=config['synonymap']

def setting_os_variables(OPENAI_API_TYPE,OPENAI_API_BASE,OPENAI_API_KEY,OPENAI_API_VERSION):
    os.environ["OPENAI_API_TYPE"] = OPENAI_API_TYPE
    os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["OPENAI_API_VERSION"] =str(OPENAI_API_VERSION)

def get_embedding_model(embedding_model_deployment_name):
    try:
        embedding: OpenAIEmbeddings = OpenAIEmbeddings(deployment=embedding_model_deployment_name, chunk_size=1)
        #embedding_function = embedding.embed_query
        return embedding
    except Exception as e:
        print("while getting embedding model: ",e)

    
def get_vector_store(vector_store_address,vector_store_password,index_name,embedding_model_deployment_name):
    index_name: str = "salesforce-vector-database1"
    #synonymap: str="synonym1"
    try:
        embedding: OpenAIEmbeddings = get_embedding_model(embedding_model_deployment_name)
        embedding_function = embedding.embed_query
        db = AzureSearch(
            azure_search_endpoint=vector_store_address,
            azure_search_key=vector_store_password,
            index_name=index_name,
            embedding_function=embedding_function
            #synonymap=synonymap
        )
        return db
    except Exception as e:
        print("exception while getting vector store: ",e)

def get_llm(text_gen_model_deployment_name,text_gen_model):
    try:
        llm = AzureOpenAI(
        deployment_name= text_gen_model_deployment_name,
        model_name= text_gen_model,
        temperature=0.1
        )
        return llm
    except Exception as e:
        print("exception while getting an llm: ",e)


def get_chain_retriever(vector_store,text_gen_llm):
    try:
        #The Given context belongs and Complies to the State New Mexico . Please provide an answer for the question using  the given context only, if the context does not provide an answer for the following question return 'The given context does not provide an answer for your question' 
        retriever=vector_store.as_retriever()   
        prompt_template="""
        {context}
        Question:{question}:"""
        PROMPT=PromptTemplate(template=prompt_template,input_variables=["context","question"])
        #PROMPT=PromptTemplate(template=prompt_template,input_variables=["question"])
        chain_type_kwargs={"prompt":PROMPT}
        qa_chain = RetrievalQA.from_chain_type(llm=text_gen_llm,chain_type="stuff",retriever=retriever,return_source_documents=True,chain_type_kwargs=chain_type_kwargs)
        return qa_chain
        
    except Exception as e:
        print("exception while getting chain retriever: ",e)


def get_result(query,qa_chain):
    try:
        result = qa_chain({"query":query})
        return result
    except Exception as e:
        print("exception while getting result: ",e)

def process_result(result):
    try:
        response = {}
        docs_list =[]
        response["answer"] =result["result"]
        for doc in result["source_documents"]:
            doc_dict ={}
            if "Possible Questions:" in doc.page_content:
                doc.page_content=doc.page_content.split("Possible Questions:")[0]
            else:
                pass
            doc_dict["content"] = doc.page_content
            doc_dict["meta"] = doc.metadata
            #doc_dict["meta"] = doc.metadata["section"]
            docs_list.append(doc_dict)
        response["document_list"]=docs_list
        
        #print(docs_list)
        return response
    except Exception as e:
        print("exception while getting on process result: ",e)



@app.route("/semantic_search/langchain/similarity_search/v1", methods = ['POST'])
def semantic_search():
    setting_os_variables(OPENAI_API_TYPE,OPENAI_API_BASE,OPENAI_API_KEY,OPENAI_API_VERSION)
    response_dict ={}
    try:
        question = request.json['question']
        #content="this is content. Possible Questions:question"
        #embedding_model=get_embedding_model(embedding_model_deployment_name)
        vector_store=get_vector_store(vector_store_address,vector_store_password,index_name,embedding_model_deployment_name)
        text_gen_llm=get_llm(text_gen_model_deployment_name,text_gen_model)
        qa_chain=get_chain_retriever(vector_store,text_gen_llm)
        result=get_result(question,qa_chain)
        response=process_result(result)
        return response
    except Exception as e:
        print(e)
        response_dict['error'] = str(e)
        return response_dict
  
app.run(host='0.0.0.0',port=9008)
