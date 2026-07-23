"""
pygeovision.enterprise.auth
============================
Role-Based Access Control (RBAC), API key management, and SSO integration.
"""
from __future__ import annotations

import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field

logger = logging.getLogger("pygeovision.enterprise.auth")


ROLES = {
    "viewer":      {"search", "download", "view"},
    "analyst":     {"search", "download", "view", "inference", "export"},
    "admin":       {"search", "download", "view", "inference", "export", "train", "admin"},
    "superadmin":  {"*"},
}


@dataclass
class User:
    user_id:    str
    email:      str
    roles:      list[str] = field(default_factory=lambda: ["viewer"])
    api_keys:   list[str] = field(default_factory=list)
    created_at: float     = field(default_factory=time.time)
    active:     bool      = True

    def has_permission(self, permission: str) -> bool:
        for role in self.roles:
            perms = ROLES.get(role, set())
            if "*" in perms or permission in perms:
                return True
        return False


class RBACManager:
    """Role-based access control for PyGeoVision Enterprise."""

    def __init__(self) -> None:
        self._users: dict[str, User] = {}

    def create_user(self, user_id: str, email: str, roles: list[str] = None) -> User:
        user = User(user_id=user_id, email=email, roles=roles or ["viewer"])
        self._users[user_id] = user
        logger.info("User created: %s  roles=%s", email, roles)
        return user

    def check_permission(self, user_id: str, permission: str) -> bool:
        user = self._users.get(user_id)
        if not user or not user.active:
            return False
        return user.has_permission(permission)

    def assign_role(self, user_id: str, role: str) -> None:
        if role not in ROLES:
            raise ValueError(f"Unknown role '{role}'. Valid: {list(ROLES)}")
        user = self._users[user_id]
        if role not in user.roles:
            user.roles.append(role)
        logger.info("Role '%s' assigned to %s", role, user.email)

    def get_user(self, user_id: str) -> User | None:
        return self._users.get(user_id)


class APIKeyManager:
    """Manage API keys for programmatic access."""

    def __init__(self) -> None:
        self._keys: dict[str, dict] = {}  # key_hash → metadata

    def generate(self, user_id: str, description: str = "", ttl_days: int = 365) -> str:
        raw_key   = f"pgv_{secrets.token_urlsafe(32)}"
        key_hash  = hashlib.sha256(raw_key.encode()).hexdigest()
        self._keys[key_hash] = {
            "user_id":    user_id,
            "description": description,
            "created_at": time.time(),
            "expires_at": time.time() + ttl_days * 86400,
            "active":     True,
        }
        logger.info("API key generated for user %s (TTL=%d days)", user_id, ttl_days)
        return raw_key  # Return once — not stored in plaintext

    def validate(self, raw_key: str) -> str | None:
        """Returns user_id if key is valid, None otherwise."""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        meta     = self._keys.get(key_hash)
        if not meta or not meta["active"]:
            return None
        if time.time() > meta["expires_at"]:
            meta["active"] = False
            return None
        return meta["user_id"]

    def revoke(self, raw_key: str) -> None:
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        if key_hash in self._keys:
            self._keys[key_hash]["active"] = False


class SSOProvider:
    """Generic OIDC (OpenID Connect) SSO integration — Authorization Code flow.

    Works with any spec-compliant OIDC identity provider (Okta, Azure AD,
    Auth0, Google Workspace, Keycloak, etc.) by either:
      1. Passing `issuer` and letting the provider discover its endpoints
         via the standard `.well-known/openid-configuration` document, or
      2. Passing `authorization_endpoint` / `token_endpoint` explicitly.

    This does NOT implement SAML 2.0 — only OIDC. SAML integration is
    provider-specific enough (XML metadata, signed assertions) that a
    generic implementation isn't meaningful; use a dedicated SAML library
    (e.g. python3-saml) configured with your IdP's metadata instead.

    Example::

        sso = SSOProvider(
            issuer="https://your-org.okta.com",
            client_id="...", client_secret="...",
        )
        auth_url = sso.get_auth_url(redirect_uri="https://app/callback", state="xyz")
        # ... user authenticates at auth_url, IdP redirects back with `code` ...
        tokens = sso.exchange_code(code, redirect_uri="https://app/callback")
        claims = sso.verify_id_token(tokens["id_token"])
    """

    def __init__(
        self,
        provider: str = "oidc",
        client_id: str = "",
        client_secret: str = "",
        issuer: str | None = None,
        authorization_endpoint: str | None = None,
        token_endpoint: str | None = None,
        jwks_uri: str | None = None,
        scope: str = "openid email profile",
    ) -> None:
        self.provider       = provider
        self._client_id     = client_id
        self._client_secret = client_secret
        self.issuer         = issuer
        self.scope          = scope
        self._authorization_endpoint = authorization_endpoint
        self._token_endpoint         = token_endpoint
        self._jwks_uri                = jwks_uri
        self._discovered = False

    def _discover(self) -> None:
        """Fetch the OIDC discovery document if endpoints weren't given explicitly."""
        if self._discovered or (self._authorization_endpoint and self._token_endpoint):
            return
        if not self.issuer:
            raise ValueError(
                "SSOProvider needs either `issuer` (for OIDC discovery) or "
                "explicit `authorization_endpoint` + `token_endpoint`."
            )
        import requests
        url = self.issuer.rstrip("/") + "/.well-known/openid-configuration"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        doc = resp.json()
        self._authorization_endpoint = self._authorization_endpoint or doc["authorization_endpoint"]
        self._token_endpoint         = self._token_endpoint or doc["token_endpoint"]
        self._jwks_uri                = self._jwks_uri or doc.get("jwks_uri")
        self._discovered = True

    def get_auth_url(self, redirect_uri: str, state: str = "") -> str:
        """Build the OIDC authorization-code-flow login URL to redirect the user to."""
        self._discover()
        from urllib.parse import urlencode
        params = {
            "response_type": "code",
            "client_id":     self._client_id,
            "redirect_uri":  redirect_uri,
            "scope":         self.scope,
            "state":         state or secrets.token_urlsafe(16),
        }
        logger.info("SSO auth URL built (%s, issuer=%s)", self.provider, self.issuer)
        return f"{self._authorization_endpoint}?{urlencode(params)}"

    def exchange_code(self, code: str, redirect_uri: str) -> dict:
        """Exchange an authorization code for access/refresh/ID tokens (RFC 6749 §4.1.3)."""
        self._discover()
        import requests
        resp = requests.post(
            self._token_endpoint,
            data={
                "grant_type":    "authorization_code",
                "code":          code,
                "redirect_uri":  redirect_uri,
                "client_id":     self._client_id,
                "client_secret": self._client_secret,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def verify_id_token(self, id_token: str) -> dict:
        """Verify and decode an OIDC ID token's claims against the provider's JWKS.

        Requires `pip install pyjwt cryptography`.
        """
        self._discover()
        try:
            import jwt
            from jwt import PyJWKClient
        except ImportError:
            raise ImportError("pip install pyjwt cryptography")
        if not self._jwks_uri:
            raise ValueError("No jwks_uri available — provide it explicitly or an `issuer` for discovery.")
        jwk_client = PyJWKClient(self._jwks_uri)
        signing_key = jwk_client.get_signing_key_from_jwt(id_token)
        return jwt.decode(
            id_token, signing_key.key, algorithms=["RS256"],
            audience=self._client_id, issuer=self.issuer,
        )
