from dataclasses import asdict
import json
from typing import Iterator
from dateutil.parser import parse
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from model import Doc


def date_to_int(dt_str: str) -> int:
    dt = parse(dt_str)
    return int(dt.timestamp())


def get_contents(project_name: str, filename: str) -> Iterator[tuple[Doc, str]]:
    """filename for file with ndjson

        {"id": <page_id>, "title": <page title>, "content": <page body>, "ctime": ..., "user": <name>, "url": "https:..."}
        {"title": ...}
    """
    with open(filename, "r") as f:
        obj = [json.loads(line) for line in f]
    for data in obj:
        title = data["title"]
        body = data["content"]
        ctime = date_to_int(data["ctime"]) if isinstance(data["ctime"], str) else data["ctime"]
        doc = Doc(
            project_name=project_name,
            id=data["id"],
            title=title,
            ctime=ctime,
            user=data["user"],
            url=data["url"],
        )
        text = title
        if body:
            text += "\n\n" + body
        yield doc, text


class DocLoader(BaseLoader):
    def __init__(self, project_name: str, filename: str):
        self.project_name = project_name
        self.filename = filename

    def lazy_load(self) -> Iterator[Document]:
        for doc, text in get_contents(self.project_name, self.filename):
            metadata = asdict(doc)
            yield Document(page_content=text, metadata=metadata)

    def load(self) -> list[Document]:
        return list(self.lazy_load())
