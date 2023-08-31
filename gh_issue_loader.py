from dataclasses import asdict
import json
from typing import Iterator
from dateutil.parser import parse
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from gh_issue_loader import Issue


def date_to_int(dt_str: str) -> int:
    dt = parse(dt_str)
    return int(dt.timestamp())


def get_contents(repo_name: str, filename: str) -> Iterator[tuple[Issue, str]]:
    with open(filename, "r") as f:
        obj = [json.loads(line) for line in f]
    for data in obj:
        title = data["title"]
        body = data["body"]
        issue = Issue(
            repo_name=repo_name,
            id=data["number"],
            title=title,
            created_at=date_to_int(data["created_at"]),
            user=data["user.login"],
            url=data["html_url"],
            labels=data["labels_"],
            type_="issue",
        )
        text = title
        if body:
            text += "\n\n" + body
        yield issue, text
        comments = data["comments_"]
        for comment in comments:
            issue = Issue(
                repo_name=repo_name,
                id=comment["id"],
                title=data["title"],
                created_at=date_to_int(comment["created_at"]),
                user=comment["user.login"],
                url=comment["html_url"],
                labels=data["labels_"],
                type_="comment",
            )
            yield issue, comment["body"]


class GHLoader(BaseLoader):
    def __init__(self, repo_name: str, filename: str):
        self.repo_name = repo_name
        self.filename = filename

    def lazy_load(self) -> Iterator[Document]:
        for issue, text in get_contents(self.repo_name, self.filename):
            metadata = asdict(issue)
            yield Document(page_content=text, metadata=metadata)

    def load(self) -> list[Document]:
        return list(self.lazy_load())
