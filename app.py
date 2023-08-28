from typing import Iterable
import streamlit as st
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from config import DB_CONFIG


@st.cache_resource
def load_embeddings():
    model_name = "intfloat/multilingual-e5-large"
    model_kwargs = {"device": "cuda:0" if torch.cuda.is_available() else "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embeddings


EMBEDDINGS = load_embeddings()


def make_filter_obj(options: list[dict[str]]):
    must = []
    for option in options:
        must.append(
            FieldCondition(key=option["key"], match=MatchValue(value=option["value"]))
        )
    filter = Filter(must=must)
    return filter


def get_similay(query: str, filter: Filter):
    db_url, db_api_key, db_collection_name = DB_CONFIG
    client = QdrantClient(url=db_url, api_key=db_api_key)
    db = Qdrant(
        client=client, collection_name=db_collection_name, embeddings=EMBEDDINGS
    )
    docs = db.similarity_search_with_score(
        query,
        k=20,
        filter=filter,
    )
    return docs


def main(
    query: str,
    repo_name: str,
) -> Iterable[tuple[str, tuple[str, str]]]:
    options = [{"key": "metadata.repo_name", "value": repo_name}]
    filter = make_filter_obj(options=options)
    docs = get_similay(query, filter)
    for doc, score in docs:
        text = doc.page_content
        metadata = doc.metadata
        # print(metadata)
        title = metadata.get("title")
        url = metadata.get("url")
        id_ = metadata.get("id")
        is_comment = metadata.get("type_") == "comment"
        yield title, url, id_, text, score, is_comment


with st.form("my_form"):
    st.title("GitHub Issue Search")
    query = st.text_input(label="query")
    repo_name = st.radio(
        options=["cocoa", "plone", "volto", "plone.restapi"], label="Repo name"
    )

    submitted = st.form_submit_button("Submit")
    if submitted:
        st.divider()
        st.header("Search Results")
        st.divider()
        with st.spinner("Searching..."):
            results = main(query, repo_name)
            for title, url, id_, text, score, is_comment in results:
                with st.container():
                    if not is_comment:
                        st.subheader(f"#{id_} - {title}")
                    else:
                        st.subheader(f"comment with {title}")
                    st.write(url)
                    st.write(text)
                    st.write(score)
                    # st.markdown(html, unsafe_allow_html=True)
                    st.divider()
