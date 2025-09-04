import os
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import clip
from PIL import Image
import base64
import hashlib
import asyncio

from chromadb import PersistentClient
from chromadb.config import Settings

# ------------------- FastAPI App ------------------- #
app = FastAPI(title="Unlimited Multi-Image Search API")

# ------------------- Config ------------------- #
DEVICE = "cpu"  # change to "cuda" if GPU available
IMAGE_DIR = "./uploaded_images"
os.makedirs(IMAGE_DIR, exist_ok=True)
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "image_collection"

# ------------------- Load CLIP ------------------- #
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# ------------------- Initialize ChromaDB ------------------- #
client = PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)

if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    collection = client.get_collection(name=COLLECTION_NAME)
else:
    collection = client.create_collection(name=COLLECTION_NAME)

# ------------------- In-memory hash map ------------------- #
hash_to_file_path = {}

# ------------------- Helper functions ------------------- #
def encode_image(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def get_image_features(image_paths: List[str]):
    batch_size = 32  # adjust for memory
    all_features = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        image_input = torch.stack(images).to(DEVICE)
        with torch.no_grad():
            features = model.encode_image(image_input).float()
        features /= features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu())
    
    return torch.cat(all_features, dim=0).numpy()

def get_text_features(texts: List[str]):
    with torch.no_grad():
        features = model.encode_text(clip.tokenize(texts).to(DEVICE)).float()
    features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()

def compute_file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()

def image_exists_in_collection(file_hash: str) -> bool:
    return file_hash in collection.get()['ids']

# ------------------- Upload endpoint ------------------- #
@app.post("/upload-images")
async def upload_images(files: List[UploadFile] = File(...)):
    added_files = []
    image_paths_to_embed = []
    image_hashes = []

    for file in files:
        # Stream file content to avoid memory issues
        content = await file.read()
        file_hash = compute_file_hash(content)
        file_path = os.path.join(IMAGE_DIR, file.filename)

        if image_exists_in_collection(file_hash):
            continue

        # Save file asynchronously
        await asyncio.to_thread(lambda: open(file_path, 'wb').write(content))

        # Update hash map
        hash_to_file_path[file_hash] = file_path

        image_paths_to_embed.append(file_path)
        image_hashes.append(file_hash)
        added_files.append(file.filename)

    # Generate embeddings in batches
    if image_paths_to_embed:
        embeddings = get_image_features(image_paths_to_embed)
        collection.add(
            ids=image_hashes,
            embeddings=embeddings.tolist()
        )

    return JSONResponse({"uploaded": added_files, "total_added": len(added_files)})

# ------------------- Search endpoint ------------------- #
@app.get("/search-images")
async def search_images(query: str, top_k: int = 3):
    query_embedding = get_text_features([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    output = []
    for idx, score in zip(results['ids'][0], results['distances'][0]):
        file_path = hash_to_file_path.get(idx)
        if file_path and os.path.exists(file_path):
            output.append({
                "filename": os.path.basename(file_path),
                "distance": float(score),
                "base64": encode_image(file_path)
            })

    return JSONResponse({"results": output})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8100, reload=True)



# import os
# import hashlib
# import base64
# from typing import List

# import torch
# import clip
# from PIL import Image

# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse

# import chromadb
# from chromadb.config import Settings

# from langchain_openai import ChatOpenAI
# from langchain.agents import initialize_agent, Tool
# from langchain.prompts import MessagesPlaceholder
# from langchain.memory import ConversationBufferMemory

# import requests



# DEVICE = "cpu"  
# IMAGE_DIR = "./uploaded_images"
# os.makedirs(IMAGE_DIR, exist_ok=True)
# CHROMA_DB_PATH = "./chroma_db"
# COLLECTION_NAME = "image_collection"


# model, preprocess = clip.load("ViT-B/32", device=DEVICE)


# client = chromadb.PersistentClient(path=CHROMA_DB_PATH, settings=Settings(anonymized_telemetry=False))
# if COLLECTION_NAME in [c.name for c in client.list_collections()]:
#     collection = client.get_collection(name=COLLECTION_NAME)
# else:
#     collection = client.create_collection(name=COLLECTION_NAME)


# def encode_image(image_path: str) -> str:
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# def get_image_features(image_paths: List[str]):
#     images = [preprocess(Image.open(p).convert("RGB")) for p in image_paths]
#     image_input = torch.stack(images).to(DEVICE)
#     with torch.no_grad():
#         features = model.encode_image(image_input).float()
#     features /= features.norm(dim=-1, keepdim=True)
#     return features.cpu().numpy()

# def get_text_features(texts: List[str]):
#     with torch.no_grad():
#         features = model.encode_text(clip.tokenize(texts).to(DEVICE)).float()
#     features /= features.norm(dim=-1, keepdim=True)
#     return features.cpu().numpy()

# def compute_file_hash(file_bytes: bytes) -> str:
#     return hashlib.md5(file_bytes).hexdigest()



# app = FastAPI(title="Chat + Image Retrieval API")


# @app.post("/upload-images")
# async def upload_images(files: List[UploadFile] = File(...)):
#     added_files, image_paths_to_embed, image_ids = [], [], []

#     for file in files:
#         content = await file.read()
#         file_hash = compute_file_hash(content)
#         file_path = os.path.join(IMAGE_DIR, file.filename)

#         if file_hash in collection.get()["ids"]:
#             continue

#         with open(file_path, "wb") as f:
#             f.write(content)

#         image_paths_to_embed.append(file_path)
#         image_ids.append(file_hash)
#         added_files.append(file.filename)

#     if image_paths_to_embed:
#         embeddings = get_image_features(image_paths_to_embed)
#         collection.add(ids=image_ids, embeddings=embeddings.tolist(), metadatas=[{"path": p} for p in image_paths_to_embed])

#     return JSONResponse({"uploaded": added_files, "total_added": len(added_files)})



# @app.get("/search-images")
# async def search_images(query: str, top_k: int = 3):
#     query_emb = get_text_features([query])[0].tolist()
#     results = collection.query(query_embeddings=[query_emb], n_results=top_k)

#     output = []
#     for idx, meta, score in zip(results["ids"][0], results["metadatas"][0], results["distances"][0]):
#         file_path = meta["path"]
#         if os.path.exists(file_path):
#             output.append({
#                 "filename": os.path.basename(file_path),
#                 "distance": float(score),
#                 "base64": encode_image(file_path)
#             })

#     return JSONResponse({"results": output})


# llm = ChatOpenAI(
#     model="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8",
#     openai_api_base="https://kharadechetan--example-vllm-inference-serve.modal.run/v1",
#     openai_api_key="token_1234",
#     temperature=0.7,
#     streaming=True,
# )


# def search_images_tool_fn(query: str):
#     """Call FastAPI search endpoint"""
#     url = f"http://localhost:8000/search-images?query={query}&top_k=2"
#     resp = requests.get(url)
#     if resp.status_code == 200:
#         return resp.json()
#     return {"error": "search failed"}

# tools = [
#     Tool(
#         name="SearchImages",
#         func=search_images_tool_fn,
#         description="Use this tool when the user asks to find images related to a query"
#     )
# ]

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent="chat-conversational-react-description",
#     verbose=True,
#     memory=memory,
# )



# @app.post("/chat")
# async def chat(user_query: str):
#     response = agent.run(user_query)
#     return {"response": response}
