import os
os.environ["HUGGINGFACE_HUB_CACHE"] = "./data/models"

import numpy as np, json, pickle, faiss, torch
from PIL import Image, ImageSequence
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATA_JSONL = "data/raw/batch_articles.jsonl"
IMAGES_DIR = "data/raw/images"

OUT_DIR    = "data/index"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- models ----------------
text_model  = SentenceTransformer("all-MiniLM-L6-v2")
clip_model  = SentenceTransformer("clip-ViT-B-32")

text_dim = int(text_model.get_sentence_embedding_dimension())

def clip_dim(model):
    dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    vec   = model.encode(dummy, convert_to_numpy=True)
    return int(vec.shape[0])           # 512 for ViT-B/32

img_dim = clip_dim(clip_model)

text_index  = faiss.IndexFlatIP(text_dim)
img_index   = faiss.IndexFlatIP(img_dim)

text_meta, img_meta = [], []

def embed_image(path):
    img = Image.open(path)
    if getattr(img, "is_animated", False):
        img = ImageSequence.Iterator(img).__next__()   # first frame of GIF
    img = img.convert("RGB").resize((224,224))
    return clip_model.encode(img, convert_to_numpy=True, normalize_embeddings=True)

# ---------------- build -----------------
with open(DATA_JSONL, encoding="utf-8") as f:
    for line in tqdm(f, desc="Embedding articles"):
        obj = json.loads(line)
        slug = obj["url"].rstrip("/").split("/")[-1]

        # --- text ---
        vec = text_model.encode(obj["text"],
                                convert_to_numpy=True,
                                normalize_embeddings=True)
        text_index.add(vec.reshape(1,-1))
        text_meta.append({
            "slug": slug,
            "title": obj["title"],
            "url"  : obj["url"],
            "img"  : obj["image_filename"],
        })

        # --- image ---
        if obj["image_filename"]:
            img_path = os.path.join(IMAGES_DIR, obj["image_filename"])
            if os.path.exists(img_path):
                ivec = embed_image(img_path)
                img_index.add(ivec.reshape(1,-1))
                img_meta.append({
                    "slug": slug,
                    "img" : obj["image_filename"],
                    "title": obj["title"],
                    "url": obj["url"],
                })

# ---------------- save ------------------
faiss.write_index(text_index, f"{OUT_DIR}/txt.index")
faiss.write_index(img_index,  f"{OUT_DIR}/img.index")
pickle.dump(text_meta, open(f"{OUT_DIR}/txt.meta.pkl","wb"))
pickle.dump(img_meta,  open(f"{OUT_DIR}/img.meta.pkl","wb"))

print("🏁 Index build complete:",
        len(text_meta),"articles |",len(img_meta),"images")
