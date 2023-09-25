---
title: Document Search
emoji: 🐠
colorFrom: green
colorTo: purple
sdk: streamlit
sdk_version: 1.25.0
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Required Environment variables

- `INDEX_NAMES`: comma separated index names
- `QDRANT_URL`: Qdrant API endpoint
- `QDRANT_API_KEY`: Qdrant API Key
- `OPENAI_API_KEY`: OpenAI API Key

# import GitHub issues

## export from github
first, generate token on: https://github.com/settings/tokens

```
$ git clone https://github.com/kazamori/github-api-tools 
$ pip install -e ./github-api-tools 
$ export GITHUB_API_TOKEN="********"
$ gh-cli-issues --repository <org/repo>
$ ls <repo>-issues.json
```

## import from json

```
$ python store.py -l github_issue <index> ../<repo>-issues.json 
```

# import Wiki Pages

## export from somewhere

create `pages.json` like:
```json
{"id": <page_id>, "title": <page title>, "content": <page body>, "ctime": ..., "user": <name>, "url": "https:..."}
{"title": ...}
```

## import from json

```
$ python store.py -l wikipage <index> ../pages.json 
```
