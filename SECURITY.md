# Security policy

## Supported versions

The latest minor release receives security fixes.

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's private
[security advisory](https://github.com/fragompul/ragforge/security/advisories/new) form, or email
frangomezpulido2002@gmail.com with a description and reproduction steps. You will get an acknowledgement
within a few days.

## Scope and design notes

- `ragforge.guardrails` is a **defense-in-depth** layer, not a guarantee: pattern rules catch common
  indirect prompt-injection and PII cases but a determined attacker can evade them. Treat retrieved text as
  untrusted and keep tools least-privilege.
- The MCP server exposes read-only tools by default; `--allow-ingest` is an explicit opt-in.
- The HTTP server is minimal (`http.server`): put it behind a reverse proxy with authentication and rate
  limiting; it bounds request size and `k` but has no auth.
- Indexes are JSON; do not load index files from untrusted sources without validating their size.

## Deployment notes

- `ragforge serve --asgi --api-key ...` requires a bearer token; the stdlib server has no authentication and should
  stay on localhost or behind a proxy.
- `MultiTenantRag` isolates tenants by giving each its own pipeline; do not share a pipeline across tenants and
  filter by metadata instead.
- The MCP server and the Claude Desktop extension are read-only unless ingestion is explicitly enabled.
- Guardrails are pattern-based and measured in `docs/guardrails.md`; recall on public injection datasets is
  partial, so treat them as one layer, not a complete defense.
