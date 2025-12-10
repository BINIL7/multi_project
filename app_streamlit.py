import streamlit as st
from retriever import SimpleRetriever
import os

st.set_page_config(page_title="Multi-Modal RAG Demo")
st.title("Multi-Modal RAG — Demo")

INDEX = "./indexes/faiss.index"
META = "./indexes/metadata.json"

if not os.path.exists(INDEX):
    st.error("Index not found. Run ingestion.py first.")
else:
    r = SimpleRetriever(INDEX, META)
    q = st.text_input("Ask something")
    if q:
        res = r.retrieve(q, k=5)
        st.write("Results:")
        for item in res:
            st.write(item)
        st.info("LLM answer part not included in this prototype.")
