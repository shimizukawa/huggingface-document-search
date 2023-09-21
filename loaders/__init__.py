from .wikipage import WikiPageLoader
from .github_issue import GithubIssueLoader

LOADERS = {
    "wikipage": WikiPageLoader,
    "github_issue": GithubIssueLoader
}
LOADER_NAMES = tuple(LOADERS.keys())

def get_loader(loader_name, **kwargs):
    return LOADERS.get(loader_name)(**kwargs)
