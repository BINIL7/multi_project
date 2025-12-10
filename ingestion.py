import os
import json
import argparse
from pathlib import Path
import pdfplumber
from PIL import Image
import pytesseract
import io
import base64
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def serialize_table(table):
    df = pd.DataFrame(table)
    return df.to_csv(index=False)

def extract_from_pdf(path, ocr=True):
    docs = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):

            try:
                text = page.extract_text() or ""
            except:
                text = ""

            if text.strip():
                docs.append({
                    "type": "text",
                    "text": text,
                    "page": i,
                    "source": os.path.basename(path)
                })

            try:
                tables = page.extract_tables()
                for t in tables:
                    txt = serialize_table(t)
                    docs.append({
                        "type": "table",
                        "text": txt,
                        "page": i,
                        "source": os.path.basename(path)
                    })
            except:
                pass

            try:
                for obj in page.images:
                    x0,y0,x1,y1 = obj.get('x0'), obj.get('top'), obj.get('x1'), obj.get('bottom')
                    try:
                        im = page.crop((x0,y0,x1,y1)).to_image(resolution=150)
                        pil = im.original
                        txt = pytesseract.image_to_string(pil) if ocr else ""

                        buf = io.BytesIO()
                        pil.save(buf, format="PNG")
                        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                        docs.append({
                            "type": "image",
                            "image_b64": img_b64,
                            "text": txt,
                            "page": i,
                            "source": os.path.basename(path)
                        })
                    except:
                        continue
            except:
                pass

    return docs

def chunk_text(text, max_chars=1500):
    sentences = text.replace('\n',' ').split('. ')
    chunks = []
    curr = ''
    for s in sentences:
        if len(curr) + len(s) + 2 < max_chars:
            curr += s + '. '
        else:
            chunks.append(curr.strip())
            curr = s + '. '
    if curr.strip():
        chunks.append(curr.strip())
    return chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--index_path", required=True)
    args = parser.parse_args()

    model_text = SentenceTransformer("all-MiniLM-L6-v2")

    index_dir = Path(args.index_path).parent
    index_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    metadata = []

    for p in Path(args.input).glob("*.pdf"):
        print("Parsing", p)
        docs = extract_from_pdf(str(p))

        for d in docs:
            if d["type"] in ["text", "table"]:
                chunks = chunk_text(d["text"])
                for c in chunks:
                    all_chunks.append(c)
                    metadata.append({
                        "type": d["type"],
                        "page": d["page"],
                        "source": d["source"]
                    })
            else:
                all_chunks.append(d["text"] or "[image]")
                metadata.append({
                    "type": "image",
                    "page": d["page"],
                    "source": d["source"],
                    "has_image": True,
                    "image_b64": d["image_b64"]
                })

    print("Computing embeddings...")
    batch = 64
    embeds = []

    for i in range(0, len(all_chunks), batch):
        b = all_chunks[i:i+batch]
        e = model_text.encode(b, show_progress_bar=False)
        embeds.append(e)

    embeds = np.vstack(embeds).astype("float32")
    dim = embeds.shape[1]

    print("Embedding dim =", dim)

    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeds)
    index.add(embeds)

    faiss.write_index(index, args.index_path)

    with open(str(index_dir / "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print("Index saved.")
