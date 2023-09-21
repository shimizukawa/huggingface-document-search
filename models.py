import dataclasses


@dataclasses.dataclass()
class BaseModel:
    index: str
    id: int
    title: str
    ctime: int
    user: str
    url: str
    type: str


@dataclasses.dataclass(frozen=True)
class GithubIssue(BaseModel):
    labels: list[str]
    type: str = "issue"


@dataclasses.dataclass(frozen=True)
class WikiPage:
    type: str = "wiki"
