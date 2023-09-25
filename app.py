from datetime import datetime
from time import time
from typing import Iterable

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range
from langchain.chains import RetrievalQA
from openai.error import InvalidRequestError
from langchain.chat_models import ChatOpenAI

from config import DB_CONFIG, INDEX_KEYS
from models import BaseModel


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


@st.cache_resource
def llm_model(model="gpt-3.5-turbo", temperature=0.2):
    llm = ChatOpenAI(model=model, temperature=temperature)
    return llm


@st.cache_resource
def load_vicuna_model():
    if torch.cuda.is_available():
        model_name = "lmsys/vicuna-13b-v1.5"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        return tokenizer, model
    else:
        return None, None


EMBEDDINGS = load_embeddings()
LLM = llm_model()
VICUNA_TOKENIZER, VICUNA_MODEL = load_vicuna_model()


@st.cache_resource
def _get_vicuna_llm(temperature=0.2) -> HuggingFacePipeline | None:
    if VICUNA_MODEL is not None:
        pipe = pipeline(
            "text-generation",
            model=VICUNA_MODEL,
            tokenizer=VICUNA_TOKENIZER,
            max_new_tokens=1024,
            temperature=temperature,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
    else:
        llm = None
    return llm


VICUNA_LLM = _get_vicuna_llm()


def make_filter_obj(options: list[dict[str]]):
    # print(options)
    must = []
    for option in options:
        if "value" in option:
            must.append(
                FieldCondition(
                    key=option["key"], match=MatchValue(value=option["value"])
                )
            )
        elif "range" in option:
            range_ = option["range"]
            must.append(
                FieldCondition(
                    key=option["key"],
                    range=Range(
                        gt=range_.get("gt"),
                        gte=range_.get("gte"),
                        lt=range_.get("lt"),
                        lte=range_.get("lte"),
                    ),
                )
            )
    filter = Filter(must=must)
    return filter


def get_similay(query: str, filter: Filter):
    db_url, db_api_key, db_collection_name = DB_CONFIG
    client = QdrantClient(url=db_url, api_key=db_api_key)
    db = Qdrant(
        client=client, collection_name=db_collection_name, embeddings=EMBEDDINGS
    )
    qdocs = db.similarity_search_with_score(
        query,
        k=20,
        filter=filter,
    )
    return qdocs


def get_retrieval_qa(filter: Filter, llm):
    db_url, db_api_key, db_collection_name = DB_CONFIG
    client = QdrantClient(url=db_url, api_key=db_api_key)
    db = Qdrant(
        client=client, collection_name=db_collection_name, embeddings=EMBEDDINGS
    )
    retriever = db.as_retriever(
        search_kwargs={
            "filter": filter,
        }
    )
    result = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return result


def _get_related_url(metadata) -> Iterable[str]:
    urls = set()
    for m in metadata:
        url = m["url"]
        if url in urls:
            continue
        urls.add(url)
        ctime = datetime.fromtimestamp(m["ctime"])
        # print(m)
        yield f'<p>URL: <a href="{url}">{url}</a> (created: {ctime:%Y-%m-%d})</p>'


def _get_query_str_filter(
    query: str,
    index: str,
) -> tuple[str, Filter]:
    options = [{"key": "metadata.index", "value": index}]
    filter = make_filter_obj(options=options)
    return query, filter


def run_qa(
    llm,
    query: str,
    index: str,
) -> tuple[str, str]:
    now = time()
    query_str, filter = _get_query_str_filter(query, index)
    qa = get_retrieval_qa(filter, llm)
    try:
        result = qa(query_str)
    except InvalidRequestError as e:
        return "回答が見つかりませんでした。別な質問をしてみてください", str(e)
    else:
        metadata = [s.metadata for s in result["source_documents"]]
        sec_html = f"<p>実行時間: {(time() - now):.2f}秒</p>"
        html = "<div>" + sec_html + "\n".join(_get_related_url(metadata)) + "</div>"
    return result["result"], html


def run_search(
    query: str,
    index: str,
) -> Iterable[tuple[BaseModel, float, str]]:
    query_str, filter = _get_query_str_filter(query, index)
    qdocs = get_similay(query_str, filter)
    for qdoc, score in qdocs:
        text = qdoc.page_content
        metadata = qdoc.metadata
        # print(metadata)
        data = BaseModel(
            index=index,
            id=metadata.get("id"),
            title=metadata.get("title"),
            ctime=metadata.get("ctime"),
            user=metadata.get("user"),
            url=metadata.get("url"),
            type=metadata.get("type"),
        )
        yield data, score, text


with st.form("my_form"):
    st.title("Document Search")
    query = st.text_area(label="query")
    index = st.selectbox(label="index", options=INDEX_KEYS)

    submit_col1, submit_col2 = st.columns(2)
    searched = submit_col1.form_submit_button("Search")
    if searched:
        st.divider()
        st.header("Search Results")
        st.divider()
        with st.spinner("Searching..."):
            results = run_search(query, index)
            for doc, score, text in results:
                title = doc.title
                url = doc.url
                id_ = doc.id
                score = round(score, 3)
                ctime = datetime.fromtimestamp(doc.ctime)
                user = doc.user
                with st.container():
                    st.subheader(f"#{id_} - {title}")
                    st.write(url)
                    st.write(text)
                    st.write("score:", score, "Date:", ctime.date(), "User:", user)
                    st.divider()
    qa_searched = submit_col2.form_submit_button("Q&A by OpenAI")
    if qa_searched:
        st.divider()
        st.header("Answer by OpenAI GPT-3")
        st.divider()
        with st.spinner("Thinking..."):
            results = run_qa(
                LLM,
                query,
                index,
            )
            answer, html = results
            with st.container():
                st.write(answer)
                st.markdown(html, unsafe_allow_html=True)
                st.divider()
    if torch.cuda.is_available():
        qa_searched_vicuna = submit_col2.form_submit_button("Answer by Vicuna")
        if qa_searched_vicuna:
            st.divider()
            st.header("Answer by Vicuna-13b-v1.5")
            st.divider()
            with st.spinner("Thinking..."):
                results = run_qa(
                    VICUNA_LLM,
                    query,
                    index,
                )
                answer, html = results
                with st.container():
                    st.write(answer)
                    st.markdown(html, unsafe_allow_html=True)
                    st.divider()
