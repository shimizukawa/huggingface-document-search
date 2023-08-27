from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant

from gh_issue_loader import GHLoader
from config import DB_CONFIG


CHUNK_SIZE = 500


def get_text_chunk(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    texts = text_splitter.split_documents(docs)
    return texts


def store(texts):
    model_name = "intfloat/multilingual-e5-large"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    db_url, db_api_key, db_collection_name = DB_CONFIG
    _ = Qdrant.from_documents(
        texts,
        embeddings,
        url=db_url,
        api_key=db_api_key,
        collection_name=db_collection_name,
    )


def main(repo_name: str, path: str) -> None:
    loader = GHLoader(repo_name, path)
    docs = loader.load()
    texts = get_text_chunk(docs)
    store(texts)


if __name__ == "__main__":
    """
    $ python store.py "REPO_NAME" "FILE_PATH"
    $ python store.py cocoa data/cocoa-issues.json
    """
    import sys

    args = sys.argv
    if len(args) != 3:
        print("No args, you need two args for repo_name, json_file_path")
    else:
        repo_name = args[1]
        path = args[2]
        main(repo_name, path)
