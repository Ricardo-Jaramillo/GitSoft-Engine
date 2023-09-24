import os
from dotenv import load_dotenv
from fastapi import FastAPI

# Create an instance of the FastAPI class
app = FastAPI()

load_dotenv()

from softtek_llm.chatbot import Chatbot
from softtek_llm.models import OpenAI
from softtek_llm.schemas import Filter
from softtek_llm.cache import Cache
from softtek_llm.vectorStores import PineconeVectorStore
from softtek_llm.embeddings import OpenAIEmbeddings

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL_NAME = os.getenv("OPENAI_CHAT_MODEL_NAME")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_EMBEDDINGS_MODEL_NAME = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

index_name = "cosmosdb-index"
# Get the service endpoint and API key from the environment
SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")

# Create a client
credential = AzureKeyCredential(SEARCH_API_KEY)
client = SearchClient(
    endpoint=SEARCH_ENDPOINT, index_name=index_name, credential=credential
)

# usr_input = input("Prompt: ")

# for result in results:
#     print("{}: {}".format(result["repository"], result["language"]))

model = OpenAI(
    api_key=OPENAI_API_KEY,
    model_name=OPENAI_CHAT_MODEL_NAME,
    api_type="azure",
    api_base=OPENAI_API_BASE,
)
filters = [
    Filter(
        type="DENY",
        case="ANYTHING related to the Titanic, no matter the question. Seriously, NO TITANIC, it's a sensitive topic.",
    ),
]
vector_store = PineconeVectorStore(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT,
    index_name=PINECONE_INDEX_NAME,
)
embeddings_model = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model_name=OPENAI_EMBEDDINGS_MODEL_NAME,
    api_type="azure",
    api_base=OPENAI_API_BASE,
)

cache = Cache(
    vector_store=vector_store,
    embeddings_model=embeddings_model,
)

chatbot = Chatbot(
    model=model,
    # filters=filters,
    # cache=cache,
    description="You are a polite and very helpful assistant.",
)
# response = chatbot.chat(
#     "Hi, my name is Jeff",
#     # cache_kwargs={"namespace": "chatbot-cache-test"},
#     print_cache_score=True,
# )

# print(response)


@app.get("/")
async def root():
    return {"message": "Hello World"}


# Define an endpoint that takes a query parameter and returns a response
@app.get("/process_string/")
async def process_string(input_string: str):
    # Replace this with your own logic to process the input_string
    # For now, let's simply return it as is
    results = client.search(search_text=input_string)

    temp = ""
    watchers = []
    reponames = []
    for i, result in enumerate(results):
        # res = result
        watchers.append(result["watchers"])
        reponames.append(result["repository"])
        temp += str(result)
        if i == 20:
            break

    try:
        response = chatbot.chat(
            f"Please write a human-readable answer, provide the best repository or repositories that matches user interests/skills, based on the following JSON output. {temp}",
            # cache_kwargs={"namespace": "chatbot-cache-test"},
            print_cache_score=True,
        )
        return {
            "output_string": response,
            "watchers": watchers,
            "reponame": temp,
        }
    except Exception as e:
        response = chatbot.chat(
            f"Please inform the user there is not a match with any repository",
            # cache_kwargs={"namespace": "chatbot-cache-test"},
            print_cache_score=True,
        )
        return {"output_string": response}
