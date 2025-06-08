import os
os.environ["HUGGINGFACE_HUB_CACHE"] = "./data/models"
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"   # disable buggy file-watcher
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
load_dotenv()

import streamlit as st, torch, faiss, pickle, json, time, sys, tqdm, pathlib, requests
from sentence_transformers import SentenceTransformer
from PIL import Image
from openai import OpenAI
from tiktoken import get_encoding
ENC = get_encoding("cl100k_base")

MAX_TOKENS = 128_000                     # GPT-4o window
INDEX_DIR   = "data/index"
IMAGES_DIR  = "data/raw/images"
ARTICLES_DIR = "data/raw/articles"
ARTICLES_PER_TOKEN_BUDGET = 180     # ≈130–150 words

def tokens(text: str) -> int:
    return len(ENC.encode(text))

def max_tokens_for(n_articles: int) -> int:
    # 500 for the first article, +180 for each additional
    return 500 + max(0, n_articles - 1) * ARTICLES_PER_TOKEN_BUDGET

def open_json(slug: str) -> str:
    """Return full article text from the per-article JSON file."""
    path = pathlib.Path(ARTICLES_DIR) / f"{slug}.json"
    if not path.exists():
        return ""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("text", "")


def build_context(articles, max_prompt_tokens=60_000):
    """Return (context_str, used_articles) trimmed to fit GPT-4o input."""
    blocks, used, total = [], [], 0
    for art in articles:                 # already sorted by FAISS score
        body = art["text"].strip()
        header = f"TITLE: {art['title']}\nURL: {art['url']}\n"
        snippet = header + body
        t = tokens(snippet)
        if total + t > max_prompt_tokens:
            break
        blocks.append(snippet)
        used.append(art)
        total += t
    context = "\n\n---\n\n".join(blocks)
    return context


# ---- cached resources ----
@st.cache_resource
def load_resources():
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    clip_model = SentenceTransformer("clip-ViT-B-32")

    txt_idx = faiss.read_index(f"{INDEX_DIR}/txt.index")
    img_idx = faiss.read_index(f"{INDEX_DIR}/img.index")
    txt_meta = pickle.load(open(f"{INDEX_DIR}/txt.meta.pkl","rb"))
    img_meta = pickle.load(open(f"{INDEX_DIR}/img.meta.pkl","rb"))

    # normalize already stored vectors
    return text_model, clip_model, txt_idx, img_idx, txt_meta, img_meta

text_model, clip_model, txt_idx, img_idx, txt_meta, img_meta = load_resources()


def build_multimodal_msg(user_txt: str, uploaded_img: bytes | None):
    """
    Returns a list ready for the OpenAI ChatCompletion 'messages' parameter.
    If an image is supplied, it's Base64-encoded and added as a vision part.
    """
    if uploaded_img is None:
        return [{"role": "user", "content": user_txt}]

    import base64, mimetypes
    mime = mimetypes.guess_type("file")[0] or "image/png"
    b64  = base64.b64encode(uploaded_img).decode()
    return [{
        "role": "user",
        "content": [
            {"type": "text",  "text": user_txt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        ],
    }]

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
def generate_answer_openai(articles, q, uploaded_img=None, model="gpt-4o-mini"):
    ctx_str = build_context(articles)
    n       = len(articles)
    want_tok= max_tokens_for(n)

    sys_msg = ("You are an expert assistant.  Using ALL the sources below "
                "and the user-supplied image (if any), write a comprehensive "
                "answer.  Cite each article at least once with a markdown link.")
    user_block = f"{ctx_str}\n\nQUESTION: {q}"

    messages = [{"role": "system", "content": sys_msg}] \
            + build_multimodal_msg(user_block, uploaded_img)

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=want_tok,
    )
    return resp.choices[0].message.content.strip()




# ---- UI ----
st.title("📰 Multimodal RAG for *The Batch*")

st.markdown("You are using GPT-4o")
query = st.text_input("Ask me anything about AI news...")
uploaded   = st.file_uploader("Optional image", type=["png","jpg","jpeg","webp","gif"])
max_k = st.slider("Max articles/images", 1, 10, 5)
if st.button("Search") and query:
    with st.spinner("Embedding & searching…"):
        qvec_txt  = text_model.encode(query, normalize_embeddings=True)
        qvec_clip = clip_model.encode([query], normalize_embeddings=True)

        # text NN
        D, I = txt_idx.search(qvec_txt.reshape(1,-1), max_k)
        art_hits = [txt_meta[i] for i in I[0]]

        # image NN
        D2, I2 = img_idx.search(qvec_clip, max_k)
        img_hits = [img_meta[i] for i in I2[0]]

    # ----- tabs -----
    tabs = st.tabs(["Articles","Images"])
    with tabs[0]:
        for h in art_hits:
            col1, col2 = st.columns([1,4])
            with col1:
                if h["img"]:
                    st.image(os.path.join(IMAGES_DIR, h["img"]), use_container_width=True)
            with col2:
                st.markdown(f"**[{h['title']}]({h['url']})**")

    with tabs[1]:
        for h in img_hits:
            img_path = os.path.join(IMAGES_DIR, h["img"])
            st.image(img_path, caption=f"[{h['title']}]({h['url']})", use_container_width=True)

    # ----- answer -----
    with st.spinner("Generating answer…"):
        ctx_docs = [
            art | {"text": open_json(art["slug"])}
            for art in art_hits[: max_k]          # limit to top-k articles
        ]

        answer = generate_answer_openai(ctx_docs, query, uploaded.read() if uploaded else None)

    st.markdown("# Answer")
    st.write(answer, unsafe_allow_html=True)
