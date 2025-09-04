import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image

API_URL = "http://127.0.0.1:8000"  # FastAPI server

st.title("Multi-Image Search")

# ------------------- Upload Images ------------------- #
st.header("Upload Images")
uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

if uploaded_files:
    files = [("files", (file.name, file, file.type)) for file in uploaded_files]
    response = requests.post(f"{API_URL}/upload-images", files=files)
    if response.ok:
        st.success(f"Uploaded: {response.json()['uploaded']}")

# ------------------- Search Images ------------------- #
st.header("Search Images")
query = st.text_input("Enter a description to search images")

if query:
    top_k = st.slider("Number of results", 1, 10, 3)
    response = requests.get(f"{API_URL}/search-images", params={"query": query, "top_k": top_k})
    
    if response.ok:
        results = response.json().get("results", [])
        st.write(f"Found {len(results)} result(s):")
        for res in results:
            st.subheader(res['filename'])
            st.write(f"Distance: {res['distance']:.4f}")
            image_data = base64.b64decode(res['base64'])
            image = Image.open(BytesIO(image_data))
            st.image(image, use_container_width=True)
            
            
            
            
            
            

# import streamlit as st
# import requests
# import base64
# from io import BytesIO
# from PIL import Image

# API_URL = "http://127.0.0.1:8000"  # FastAPI server

# st.set_page_config(page_title="Chat + Multi-Image Search", layout="wide")
# st.title("Chat + Image Search System")

# # ------------------- Upload Images ------------------- #
# st.sidebar.header("Upload Images")
# uploaded_files = st.sidebar.file_uploader(
#     "Choose images", accept_multiple_files=True, type=["png", "jpg", "jpeg"]
# )

# if uploaded_files and st.sidebar.button("Upload"):
#     files = [("files", (file.name, file, file.type)) for file in uploaded_files]
#     response = requests.post(f"{API_URL}/upload-images", files=files)
#     if response.ok:
#         st.sidebar.success(f"Uploaded: {response.json()['uploaded']}")

# # ------------------- Tabs ------------------- #
# tab1, tab2 = st.tabs([" Search Images", " Chat with LLM"])

# # ------------------- Manual Search Tab ------------------- #
# with tab1:
#     st.header("Search Images Manually")
#     query = st.text_input("Enter a description to search images")

#     if query:
#         top_k = st.slider("Number of results", 1, 10, 3)
#         response = requests.get(
#             f"{API_URL}/search-images", params={"query": query, "top_k": top_k}
#         )

#         if response.ok:
#             results = response.json().get("results", [])
#             st.write(f"Found {len(results)} result(s):")
#             for res in results:
#                 st.subheader(res["filename"])
#                 st.write(f"Distance: {res['distance']:.4f}")
#                 image_data = base64.b64decode(res["base64"])
#                 image = Image.open(BytesIO(image_data))
#                 st.image(image, use_container_width=True)

# # ------------------- Chat Tab ------------------- #
# with tab2:
#     st.header("Chat with Llama 3.1 + Image Retrieval")

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     user_input = st.text_input("You:", key="chat_input")

#     if st.button("Send") and user_input:
#         response = requests.post(f"{API_URL}/chat", params={"user_query": user_input})
#         if response.ok:
#             reply = response.json()["response"]
#             st.session_state.chat_history.append(("You", user_input))
#             st.session_state.chat_history.append(("Bot", reply))

#     # Display chat history
#     for role, text in st.session_state.chat_history:
#         if role == "You":
#             st.markdown(f"** You:** {text}")
#         else:
#             st.markdown(f"** Bot:** {text}")
