from tqdm import tqdm
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant

from doc_loader import DocLoader
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
    model_kwargs = {"device": "cuda:0" if torch.cuda.is_available() else "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    db_url, db_api_key, db_collection_name = DB_CONFIG
    for text in tqdm(texts):
        _ = Qdrant.from_documents(
            [text],
            embeddings,
            url=db_url,
            api_key=db_api_key,
            collection_name=db_collection_name,
        )


def main(project_name: str, path: str) -> None:
    loader = DocLoader(project_name, path)
    docs = loader.load()
    texts = get_text_chunk(docs)
    store(texts)


if __name__ == "__main__":
    """
    $ python store.py "PROJECT_NAME" "FILE_PATH"
    $ python store.py hoge data/hoge-docs.json
    """
    import sys

    args = sys.argv
    if len(args) != 3:
        print("No args, you need two args for project_name, json_file_path")
    else:
        project_name = args[1]
        path = args[2]
        main(project_name, path)
