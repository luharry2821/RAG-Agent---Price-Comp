# Retell MCP server

`.mcp.json` registers Retell's hosted MCP endpoint (`https://mcp.retellai.com`)
for this project. The `Authorization` header is read from `RETELL_API_KEY` via
`${...}` expansion, so no credential is stored in the repository.

## Local setup

```bash
export RETELL_API_KEY=key_...   # your Retell API key
claude                          # approve the server at the prompt on first run
```

Claude Code asks for approval once per project before it will contact a server
declared in `.mcp.json`; the choice is remembered afterwards.

Verify without starting a session:

```bash
claude mcp list
```

A healthy result names the server and reports no config warnings. If it prints
`Missing environment variables: RETELL_API_KEY`, the variable is not exported in
the shell you ran it from.

## Cloud sessions (claude.ai/code)

Two settings live on the environment, not in this repo:

1. **The key** — add `RETELL_API_KEY` under the environment's API credentials or
   environment variables. A new session picks it up.
2. **Network access** — the default network policy denies `mcp.retellai.com`.
   Add that host to the allowed domains, or select a broader access level.

Both are under the cloud environment menu in the session title bar, then *Edit*.

## Auth header format

The header is sent verbatim as `Authorization: <RETELL_API_KEY>`. If Retell
rejects it, the endpoint likely wants a bearer prefix — set
`RETELL_API_KEY="Bearer key_..."` rather than editing `.mcp.json`, so the
credential stays out of version control.
