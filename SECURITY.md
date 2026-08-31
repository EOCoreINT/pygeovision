# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.0.x   | ✅ Active |
| 1.0.x   | ⚠️ Critical fixes only |
| < 1.0   | ❌ No support |

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities via public GitHub Issues.**

Instead, email: **security@pygeovision.org**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested mitigation

You will receive an acknowledgment within **48 hours** and a detailed response within **7 days**.

## Security Practices

### Credential Handling
- All satellite provider credentials are stored in the **system keyring** (macOS Keychain, Windows Credential Manager, Linux Secret Service) — never in plain text on disk
- Encrypted file fallback (`~/.pygeofetch/credentials.enc`) uses Fernet AES-128-CBC
- Logs automatically redact passwords, tokens, and API keys
- `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` must be set in headless environments — credentials then come from environment variables only

### Network Security
- TLS 1.2+ enforced on all provider connections
- No `verify=False` anywhere in the codebase
- Connection timeouts configured per provider
- **PyGeoVision never phones home** — no telemetry collected

### Data Integrity
- SHA256 / MD5 checksum verification on all downloads
- Atomic file writes (`.tmp` then rename) — no partial files on disk
- ONNX model hash verification on load (when enabled)

### API Security
- `InferenceServer` uses JWT Bearer token authentication
- Rate limiting per API key
- CORS disabled by default
- No user data stored server-side after inference

### Docker Security
- Production image runs as non-root user (`pgvuser`, uid=1000)
- Base images pinned to specific digests in CI
- Secrets passed via environment variables, never baked into images

### Dependency Security
- Dependencies pinned with minimum versions in `pyproject.toml`
- `pip audit` runs in CI on every pull request
- Dependabot configured for automated security updates

## Scope

The following are **in scope** for security reports:
- Credential exposure or leakage
- Remote code execution via malformed GeoTIFF/STAC inputs
- Authentication bypass in `InferenceServer`
- Path traversal in file operations
- Dependency vulnerabilities with CVE scores ≥ 7.0

The following are **out of scope**:
- Vulnerabilities in optional dependencies (PyGeoFetch) — report to those projects
- Social engineering
- Physical security
- Issues requiring physical access to user machines
