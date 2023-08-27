import os


SAAS = False


def get_db_config():
    url = os.environ["QDRANT_URL"]
    api_key = os.environ["QDRANT_API_KEY"]
    collection_name = "gh-issues"
    return url, api_key, collection_name


def get_local_db_congin():
    url = "localhost"
    # api_key = os.environ["QDRANT_API_KEY"]
    collection_name = "gh-issues"
    return url, None, collection_name


DB_CONFIG = get_db_config() if SAAS else get_local_db_congin()
