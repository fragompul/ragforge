# Security Policy

## Supported Versions

`ragforge` is currently pre-1.0 (`0.x`). Security fixes are made against the
latest release on the `main` branch; there is no long-term support branch
yet.

| Version | Supported |
| ------- | --------- |
| 0.x     | ✅ |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for a suspected security
vulnerability. Instead, report it privately via
[GitHub's private vulnerability reporting](https://github.com/fragompul/ragforge/security/advisories/new)
on this repository, or by emailing the maintainer listed on the
[GitHub profile](https://github.com/fragompul).

Include, where possible:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, or a minimal proof-of-concept.
- The affected version/commit.

You should expect an initial response within a few days. Once a fix is
confirmed, a new release will be published and the reporter credited
(unless anonymity is requested) in the release notes.

## Scope and Known Considerations

`ragforge` core has zero required third-party dependencies, which
eliminates most supply-chain attack surface for typical usage. Two areas
are worth flagging explicitly for anyone deploying it:

- **`src/ragforge/server.py`** implements a minimal HTTP JSON API with no
  built-in authentication, TLS termination, or rate limiting -- it is meant
  to sit behind a reverse proxy or API gateway in any real deployment, the
  same way a bare Flask/FastAPI app would.
- **`src/ragforge/embeddings_providers.py`** and **`src/ragforge/telemetry.py`**
  lazily import optional third-party packages (`openai`, `cohere`,
  `sentence-transformers`, `opentelemetry-api`) only when their respective
  factory functions are called. Pin versions of whichever extras you
  install, as with any dependency.
