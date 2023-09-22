import dataclasses


@dataclasses.dataclass(frozen=True)
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
    labels: list[str] = dataclasses.field(default_factory=list)
    type: str = "issue"


@dataclasses.dataclass(frozen=True)
class WikiPage:
    type: str = "wiki"
