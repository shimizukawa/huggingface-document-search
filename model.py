from dataclasses import dataclass


@dataclass(frozen=True)
class Doc:
    project_name: str
    id: int
    title: str
    ctime: int
    user: str
    url: str
