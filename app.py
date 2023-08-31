from datetime import datetime, date, timedelta
from typing import Iterable
import streamlit as st
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range
from config import DB_CONFIG
from model import Issue


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
    docs = db.similarity_search_with_score(
        query,
        k=20,
        filter=filter,
    )
    return docs


def main(
    query: str,
    repo_name: str,
    query_options: str,
    start_date: date,
    end_date: date,
    include_comments: bool,
) -> Iterable[tuple[Issue, float, str]]:
    options = [{"key": "metadata.repo_name", "value": repo_name}]
    if start_date is not None and end_date is not None:
        options.append(
            {
                "key": "metadata.created_at",
                "range": {
                    "gte": int(datetime.fromisoformat(str(start_date)).timestamp()),
                    "lte": int(
                        datetime.fromisoformat(
                            str(end_date + timedelta(days=1))
                        ).timestamp()
                    ),
                },
            }
        )
    if not include_comments:
        options.append({"key": "metadata.type_", "value": "issue"})
    filter = make_filter_obj(options=options)
    if query_options == "Empty":
        query_options = ""
    query_str = f"{query_options}{query}"
    docs = get_similay(query_str, filter)
    for doc, score in docs:
        text = doc.page_content
        metadata = doc.metadata
        # print(metadata)
        issue = Issue(
            repo_name=repo_name,
            id=metadata.get("id"),
            title=metadata.get("title"),
            created_at=metadata.get("created_at"),
            user=metadata.get("user"),
            url=metadata.get("url"),
            labels=metadata.get("labels"),
            type_=metadata.get("type_"),
        )
        yield issue, score, text


with st.form("my_form"):
    st.title("GitHub Issue Search")
    query = st.text_input(label="query")
    repo_name = st.radio(
        options=[
            "cpython",
            "pyvista",
            "plone",
            "volto",
            "plone.restapi",
            "nvda",
            "nvdajp",
            "cocoa",
        ],
        label="Repo name",
    )
    query_options = st.radio(
        options=[
            "query: ",
            "query: passage: ",
            "Empty",
        ],
        label="Query options",
    )
    date_min = date(2022, 1, 1)
    date_max = date.today()
    date_col1, date_col2 = st.columns(2)
    start_date = date_col1.date_input(
        label="Select a start date",
        value=date_min,
        format="YYYY-MM-DD",
    )
    end_date = date_col2.date_input(
        label="Select a end date",
        value=date_max,
        format="YYYY-MM-DD",
    )
    include_comments = st.checkbox(label="Include Issue comments", value=True)

    submitted = st.form_submit_button("Submit")
    if submitted:
        st.divider()
        st.header("Search Results")
        st.divider()
        with st.spinner("Searching..."):
            results = main(
                query, repo_name, query_options, start_date, end_date, include_comments
            )
            for issue, score, text in results:
                title = issue.title
                url = issue.url
                id_ = issue.id
                score = round(score, 3)
                created_at = datetime.fromtimestamp(issue.created_at)
                user = issue.user
                labels = issue.labels
                is_comment = issue.type_ == "comment"
                with st.container():
                    if not is_comment:
                        st.subheader(f"#{id_} - {title}")
                    else:
                        st.subheader(f"comment with {title}")
                    st.write(url)
                    st.write(text)
                    st.write("score:", score, "Date:", created_at.date(), "User:", user)
                    st.write(f"{labels=}")
                    # st.markdown(html, unsafe_allow_html=True)
                    st.divider()
