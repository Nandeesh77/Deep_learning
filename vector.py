# pip install langchain
# pip install azure-search-documents==11.4.0b6
# pip install python-dotenv



# OPENAI_API_BASE=<YOUR-AZURE-OPENAI-ENDPOINT>
# OPENAI_API_KEY=<YOUR-AZURE-OPENAI-KEY>
# OPENAI_API_VERSION=<YOUR-AZURE-OPENAI-API-VERSION>
# AZURE_COGNITIVE_SEARCH_SERVICE_NAME=<YOUR-COG-SEARCH-SERVICE-NAME>
# AZURE_COGNITIVE_SEARCH_API_KEY=<YOUR-COG-SEARCH-KEY>
# AZURE_COGNITIVE_SEARCH_INDEX_NAME=<YOUR-COG-SEARCH-INDEX-NAME>
# AZURE_COGNITIVE_SEARCH_ENDPOINT=<YOUR-COG-SEARCH-ENDPOINT>
# OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME=<YOUR-AZURE-OPENAI-ADA-EMBEDDING-DEPLOYMENT>
# OPENAI_ADA_EMBEDDING_MODEL_NAME=<YOUR-AZURE-OPENAI-ADA-EMBEDDING-MODEL>



import os

from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AzureSearch

load_dotenv('.env')  # take environment variables from .env.

root_dir = ".\\azure-openai-samples"

# Loop through the folders
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass


# Split into chunk of texts
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)


# Initialize our embedding model
embeddings=OpenAIEmbeddings(deployment=os.getenv('OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME'),
                                model=os.getenv('OPENAI_ADA_EMBEDDING_MODEL_NAME'),
                                openai_api_base=os.getenv('OPENAI_API_BASE'),
                                openai_api_type="azure",
                                chunk_size=1)

index_name = 'index-azure-openai-samples'

# Set our Azure Search
acs = AzureSearch(azure_search_endpoint=os.getenv('AZURE_COGNITIVE_SEARCH_ENDPOINT'),
                 azure_search_key=os.getenv('AZURE_COGNITIVE_SEARCH_API_KEY'),
                 index_name=index_name,
                 embedding_function=embeddings.embed_query)

# Add documents to Azure Search
acs.add_documents(documents=texts)


import json
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain.prompts import PromptTemplate

# Define Azure Cognitive Search as our retriever
index_name = 'index-azure-openai-samples'
retriever = AzureCognitiveSearchRetriever(content_key="content", top_k=10, index_name=index_name)



# Set chatGPT 3.5 as our LLM
llm = AzureChatOpenAI(deployment_name="gpt-35-turbo-16k", temperature=0)


# Define a template message
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Set the Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True
)


questions = ['Explain the notebook 01_create_resource.ipynb',
             'Explain the notebook 02_OpenAI_getting_started.ipynb',             
             'Give me a step by step process to create an OpenAI Demo'
             ]


chat_history = []

for question in questions:
    result = qa_chain({"query": question, "chat_history": chat_history})
    #chat_history.append((question, result))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['result']} \n")
    print(f"**Source**:{json.loads(result['source_documents'][0].metadata['metadata'])['source']} \n")
