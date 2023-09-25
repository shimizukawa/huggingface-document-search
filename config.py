import os


SAAS = True


def get_db_config():
    url = os.environ["QDRANT_URL"]
    api_key = os.environ["QDRANT_API_KEY"]
    collection_name = "document-search"
    return url, api_key, collection_name


def get_local_db_congin():
    url = "localhost"
    # api_key = os.environ["QDRANT_API_KEY"]
    collection_name = "document-search"
    return url, None, collection_name


def get_index_keys():
    keys = [
        k for k in [
            k.strip().lower()
            for k in os.environ["INDEX_KEYS"].split(",")
        ]
        if k
    ]
    if not keys:
        keys = ["INDEX_KEYS is empty"]
    return keys


DB_CONFIG = get_db_config() if SAAS else get_local_db_congin()
INDEX_KEYS = get_index_keys()