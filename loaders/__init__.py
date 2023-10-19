from .wikipage import WikiPageLoader
from .github_issue import GithubIssueLoader
from .rtdhtmlpage import RTDHtmlPageLoader

LOADERS = {
    "wikipage": WikiPageLoader,
    "github_issue": GithubIssueLoader,
    "rtdhtmlpage": RTDHtmlPageLoader,
}
LOADER_NAMES = tuple(LOADERS.keys())

def get_loader(loader_name, **kwargs):
    return LOADERS.get(loader_name)(**kwargs)
