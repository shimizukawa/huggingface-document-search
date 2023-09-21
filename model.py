from dataclasses import dataclass


@dataclass(frozen=True)
class Doc:
    project_name: str
    id: int
    title: str
    created_at: int
    user: str
    url: str
