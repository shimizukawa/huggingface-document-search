from dataclasses import dataclass


@dataclass(frozen=True)
class Issue:
    repo_name: str
    id: int
    title: str
    created_at: int
    user: str
    url: str
    labels: list[str]
    type_: str
