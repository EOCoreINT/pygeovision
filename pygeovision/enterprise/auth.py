"""
pygeovision.enterprise.auth
============================
Role-Based Access Control (RBAC), API key management, and SSO integration.
"""
from __future__ import annotations
import hashlib, logging, secrets, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

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
    roles:      List[str] = field(default_factory=lambda: ["viewer"])
    api_keys:   List[str] = field(default_factory=list)
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
        self._users: Dict[str, User] = {}

    def create_user(self, user_id: str, email: str, roles: List[str] = None) -> User:
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

    def get_user(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)


class APIKeyManager:
    """Manage API keys for programmatic access."""

    def __init__(self) -> None:
        self._keys: Dict[str, Dict] = {}  # key_hash → metadata

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

    def validate(self, raw_key: str) -> Optional[str]:
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
    """Stub for SSO (SAML 2.0 / OIDC) integration."""

    def __init__(self, provider: str = "oidc", client_id: str = "", client_secret: str = "") -> None:
        self.provider      = provider
        self._client_id    = client_id
        self._client_secret= client_secret

    def get_auth_url(self, redirect_uri: str, state: str = "") -> str:
        logger.info("SSO auth URL requested (%s)", self.provider)
        return f"https://sso.example.com/authorize?client_id={self._client_id}&redirect_uri={redirect_uri}"

    def exchange_code(self, code: str, redirect_uri: str) -> Dict:
        """Exchange authorization code for access/refresh tokens."""
        raise NotImplementedError("Configure your SSO provider credentials.")
