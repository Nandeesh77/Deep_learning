from langchain import OpenAIEmbeddings
from azure.search.documents import SearchClient, IndexClient
from azure.search.documents.models import (
    SearchIndex,
    InputField,
    Skill,
    SkillCatalog,
)

# Replace with your Azure configurations
OPENAI_API_KEY = "<YOUR_OPENAI_API_KEY>"
OPENAI_API_ENDPOINT = "<YOUR_OPENAI_API_ENDPOINT>"
AZURE_SEARCH_SERVICE_NAME = "<YOUR_AZURE_SEARCH_SERVICE_NAME>"
AZURE_SEARCH_API_KEY = "<YOUR_AZURE_SEARCH_API_KEY>"
AZURE_SEARCH_INDEX_NAME = "<YOUR_AZURE_SEARCH_INDEX_NAME>"

# Initialize OpenAI embeddings
openai_embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    api_endpoint=OPENAI_API_ENDPOINT,
    model_version="1.3.7",
)

# Initialize Azure Search clients
index_client = IndexClient.from_connection_string(
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_SERVICE_NAME,
)
search_client = SearchClient.from_connection_string(
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_SERVICE_NAME,
    index_name=AZURE_SEARCH_INDEX_NAME,
)

# Define the text document
text_document = "This is the text document to be indexed."

# Create the vector embedding
vector_embedding = openai_embeddings.embed(text_document)

# Define the document to index
document = {
    "id": "1",
    "content": text_document,
    "embedding": vector_embedding,
}

# Index the document
index_client.index_documents(documents=[document])

# Search for similar documents
search_results = search_client.search(
    query="*",
    filter="distance(embedding, @value) ge 0.7",
    parameters={"@value": vector_embedding},
)

# Print the search results
for result in search_results.results:
    print(f"Document ID: {result.document.id}")
    print(f"Similarity score: {result.score}")

