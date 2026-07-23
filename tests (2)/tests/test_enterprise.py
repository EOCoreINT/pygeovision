"""
tests/test_enterprise.py
==========================
Tests for pygeovision.enterprise — RBACManager, APIKeyManager, SSOProvider,
AuditLogger, ComplianceChecker, GDPRHandler.

No external services required; all tests use in-memory state.
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


# ══════════════════════════════════════════════════════════════════════════════
# 1. RBACManager
# ══════════════════════════════════════════════════════════════════════════════

class TestRBACManager:

    def test_create_user_returns_user(self):
        from pygeovision.enterprise.auth import RBACManager
        rbac = RBACManager()
        user = rbac.create_user("alice", "alice@example.com", roles=["viewer"])
        assert user.user_id == "alice"
        assert user.email   == "alice@example.com"
        assert "viewer" in user.roles

    def test_default_role_is_viewer(self):
        from pygeovision.enterprise.auth import RBACManager
        rbac = RBACManager()
        user = rbac.create_user("bob", "bob@example.com")
        assert "viewer" in user.roles

    def test_viewer_can_search(self):
        from pygeovision.enterprise.auth import RBACManager
        rbac = RBACManager()
        rbac.create_user("u1", "u1@example.com", roles=["viewer"])
        assert rbac.check_permission("u1", "search") is True

    def test_viewer_cannot_admin(self):
        from pygeovision.enterprise.auth import RBACManager
        rbac = RBACManager()
        rbac.create_user("u2", "u2@example.com", roles=["viewer"])
        assert rbac.check_permission("u2", "admin") is False

    def test_analyst_can_inference(self):
        from pygeovision.enterprise.auth import RBACManager
        rbac = RBACManager()
        rbac.create_user("a1", "a1@example.com", roles=["analyst"])
        assert rbac.check_permission("a1", "inference") is True

    def test_analyst_cannot_admin(self):
        from pygeovision.enterprise.auth import RBACManager
        rbac = RBACManager()
        rbac.create_user("a2", "a2@example.com", roles=["analyst"])
        assert rbac.check_permission("a2", "admin") is False

    def test_admin_can_train(self):
        from pygeovision.enterprise.auth import RBACManager
        rbac = RBACManager()
        rbac.create_user("ad1", "ad1@example.com", roles=["admin"])
        assert rbac.check_permission("ad1", "train") is True

    def test_superadmin_has_all_permissions(self):
        from pygeovision.enterprise.auth import RBACManager
        rbac = RBACManager()
        rbac.create_user("sa", "sa@example.com", roles=["superadmin"])
        for perm in ["search", "download", "inference", "train", "admin", "export"]:
            assert rbac.check_permission("sa", perm) is True, f"superadmin missing {perm}"

    def test_unknown_user_returns_false(self):
        from pygeovision.enterprise.auth import RBACManager
        rbac = RBACManager()
        assert rbac.check_permission("nonexistent_user", "search") is False

    def test_assign_role(self):
        from pygeovision.enterprise.auth import RBACManager
        rbac = RBACManager()
        rbac.create_user("u3", "u3@example.com", roles=["viewer"])
        assert rbac.check_permission("u3", "inference") is False
        rbac.assign_role("u3", "analyst")
        assert rbac.check_permission("u3", "inference") is True

    def test_assign_unknown_role_raises(self):
        from pygeovision.enterprise.auth import RBACManager
        rbac = RBACManager()
        rbac.create_user("u4", "u4@example.com")
        with pytest.raises(ValueError, match="Unknown role"):
            rbac.assign_role("u4", "nonexistent_role")

    def test_get_user_returns_user(self):
        from pygeovision.enterprise.auth import RBACManager
        rbac = RBACManager()
        rbac.create_user("u5", "u5@example.com")
        user = rbac.get_user("u5")
        assert user is not None
        assert user.email == "u5@example.com"

    def test_get_unknown_user_returns_none(self):
        from pygeovision.enterprise.auth import RBACManager
        rbac = RBACManager()
        assert rbac.get_user("nobody") is None

    def test_inactive_user_denied(self):
        from pygeovision.enterprise.auth import RBACManager
        rbac = RBACManager()
        user = rbac.create_user("u6", "u6@example.com", roles=["admin"])
        user.active = False
        assert rbac.check_permission("u6", "search") is False


# ══════════════════════════════════════════════════════════════════════════════
# 2. APIKeyManager
# ══════════════════════════════════════════════════════════════════════════════

class TestAPIKeyManager:

    def test_generate_returns_string(self):
        from pygeovision.enterprise.auth import APIKeyManager
        mgr = APIKeyManager()
        key = mgr.generate("alice", description="test key")
        assert isinstance(key, str)
        assert len(key) > 20

    def test_key_starts_with_pgv(self):
        from pygeovision.enterprise.auth import APIKeyManager
        mgr = APIKeyManager()
        key = mgr.generate("alice")
        assert key.startswith("pgv_")

    def test_valid_key_returns_user_id(self):
        from pygeovision.enterprise.auth import APIKeyManager
        mgr = APIKeyManager()
        key = mgr.generate("alice")
        user_id = mgr.validate(key)
        assert user_id == "alice"

    def test_invalid_key_returns_none(self):
        from pygeovision.enterprise.auth import APIKeyManager
        mgr = APIKeyManager()
        assert mgr.validate("pgv_invalidkeyxyz123") is None

    def test_revoked_key_returns_none(self):
        from pygeovision.enterprise.auth import APIKeyManager
        mgr = APIKeyManager()
        key = mgr.generate("alice")
        assert mgr.validate(key) == "alice"
        mgr.revoke(key)
        assert mgr.validate(key) is None

    def test_expired_key_returns_none(self):
        from pygeovision.enterprise.auth import APIKeyManager
        mgr = APIKeyManager()
        key = mgr.generate("alice", ttl_days=0)  # expires immediately
        # Manually backdate the expiry
        import hashlib
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        mgr._keys[key_hash]["expires_at"] = time.time() - 1
        assert mgr.validate(key) is None

    def test_two_keys_are_unique(self):
        from pygeovision.enterprise.auth import APIKeyManager
        mgr  = APIKeyManager()
        key1 = mgr.generate("alice")
        key2 = mgr.generate("alice")
        assert key1 != key2

    def test_key_not_stored_in_plaintext(self):
        from pygeovision.enterprise.auth import APIKeyManager
        mgr = APIKeyManager()
        key = mgr.generate("alice")
        # The raw key should NOT appear in the internal store
        for meta in mgr._keys.values():
            assert key not in str(meta)


# ══════════════════════════════════════════════════════════════════════════════
# 2b. SSOProvider (OIDC)
# ══════════════════════════════════════════════════════════════════════════════

class TestSSOProvider:

    def test_get_auth_url_with_explicit_endpoints(self):
        """No network call needed when endpoints are given explicitly."""
        from pygeovision.enterprise.auth import SSOProvider
        sso = SSOProvider(
            client_id="my-client-id",
            authorization_endpoint="https://idp.example.org/authorize",
            token_endpoint="https://idp.example.org/token",
        )
        url = sso.get_auth_url(redirect_uri="https://app.example.com/callback", state="xyz123")
        assert url.startswith("https://idp.example.org/authorize?")
        assert "client_id=my-client-id" in url
        assert "state=xyz123" in url
        assert "response_type=code" in url

    def test_get_auth_url_generates_state_if_absent(self):
        from pygeovision.enterprise.auth import SSOProvider
        sso = SSOProvider(
            client_id="c",
            authorization_endpoint="https://idp.example.org/authorize",
            token_endpoint="https://idp.example.org/token",
        )
        url = sso.get_auth_url(redirect_uri="https://app/callback")
        assert "state=" in url

    def test_discover_requires_issuer_or_explicit_endpoints(self):
        from pygeovision.enterprise.auth import SSOProvider
        sso = SSOProvider(client_id="c")  # no issuer, no explicit endpoints
        with pytest.raises(ValueError):
            sso.get_auth_url(redirect_uri="https://app/callback")

    def test_exchange_code_posts_to_token_endpoint(self, monkeypatch):
        from pygeovision.enterprise.auth import SSOProvider

        class _FakeResponse:
            def raise_for_status(self): pass
            def json(self): return {"access_token": "abc", "id_token": "def"}

        captured = {}
        def fake_post(url, data=None, timeout=None):
            captured["url"] = url
            captured["data"] = data
            return _FakeResponse()

        import requests
        monkeypatch.setattr(requests, "post", fake_post)

        sso = SSOProvider(
            client_id="c", client_secret="s",
            authorization_endpoint="https://idp.example.org/authorize",
            token_endpoint="https://idp.example.org/token",
        )
        tokens = sso.exchange_code("auth-code-123", redirect_uri="https://app/callback")
        assert tokens["access_token"] == "abc"
        assert captured["url"] == "https://idp.example.org/token"
        assert captured["data"]["grant_type"] == "authorization_code"
        assert captured["data"]["code"] == "auth-code-123"


# ══════════════════════════════════════════════════════════════════════════════
# 3. AuditLogger
# ══════════════════════════════════════════════════════════════════════════════

class TestAuditLogger:

    def test_log_action_appends_to_memory(self):
        from pygeovision.enterprise.audit import AuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(f"{tmp}/audit.jsonl")
            logger.log_action("alice", "inference", "prithvi", outcome="success")
            assert len(logger._events) == 1

    def test_multiple_events_appended(self):
        from pygeovision.enterprise.audit import AuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(f"{tmp}/audit.jsonl")
            for i in range(5):
                logger.log_action(f"user{i}", "search", "sentinel2")
            assert len(logger._events) == 5

    def test_jsonl_file_created(self):
        from pygeovision.enterprise.audit import AuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/audit.jsonl"
            logger = AuditLogger(path)
            logger.log_action("alice", "download", "scene_001")
            assert pathlib.Path(path).exists()

    def test_jsonl_file_valid_json_per_line(self):
        from pygeovision.enterprise.audit import AuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/audit.jsonl"
            logger = AuditLogger(path)
            logger.log_action("alice", "inference", "model", outcome="success")
            logger.log_action("bob",   "search",    "sentinel2")
            lines = pathlib.Path(path).read_text().strip().split("\n")
            for line in lines:
                data = json.loads(line)   # must not raise
                assert "user_id" in data
                assert "action"  in data

    def test_query_by_user(self):
        from pygeovision.enterprise.audit import AuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(f"{tmp}/a.jsonl")
            logger.log_action("alice", "search", "s2")
            logger.log_action("bob",   "search", "s2")
            logger.log_action("alice", "inference", "prithvi")
            alice_events = logger.query(user_id="alice")
            assert len(alice_events) == 2
            assert all(e.user_id == "alice" for e in alice_events)

    def test_query_by_action(self):
        from pygeovision.enterprise.audit import AuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(f"{tmp}/a.jsonl")
            logger.log_action("alice", "search",    "s2")
            logger.log_action("alice", "inference", "prithvi")
            logger.log_action("bob",   "search",    "landsat")
            search_events = logger.query(action="search")
            assert len(search_events) == 2

    def test_query_by_since(self):
        from pygeovision.enterprise.audit import AuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(f"{tmp}/a.jsonl")
            time.time() - 10
            logger.log_action("alice", "search", "s2")
            t_after  = time.time()
            logger.log_action("bob",   "inference", "prithvi")
            recent = logger.query(since=t_after - 1)
            assert len(recent) >= 1
            assert all(e.timestamp >= t_after - 1 for e in recent)

    def test_export_jsonl(self):
        from pygeovision.enterprise.audit import AuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(f"{tmp}/a.jsonl")
            logger.log_action("alice", "search", "s2")
            out = f"{tmp}/export.jsonl"
            logger.export(out, format="jsonl")
            assert pathlib.Path(out).exists()

    def test_export_json(self):
        from pygeovision.enterprise.audit import AuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(f"{tmp}/a.jsonl")
            logger.log_action("alice", "search", "s2")
            out = f"{tmp}/export.json"
            logger.export(out, format="json")
            data = json.load(open(out))
            assert isinstance(data, list)
            assert len(data) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 4. ComplianceChecker
# ══════════════════════════════════════════════════════════════════════════════

class TestComplianceChecker:

    def _full_config(self) -> dict:
        return {
            "encryption_at_rest":    True,
            "tls_enabled":           True,
            "rbac_enabled":          True,
            "mfa_required":          True,
            "audit_logging":         True,
            "data_retention_days":   90,
        }

    def test_all_checks_pass_with_good_config(self):
        from pygeovision.enterprise.compliance import ComplianceChecker
        checker = ComplianceChecker()
        checks  = checker.check_all(self._full_config())
        assert all(c.passed for c in checks), \
            [c.check_name for c in checks if not c.passed]

    def test_encryption_at_rest_check(self):
        from pygeovision.enterprise.compliance import ComplianceChecker
        checker = ComplianceChecker()
        cfg = self._full_config()
        cfg["encryption_at_rest"] = False
        checks  = checker.check_all(cfg)
        enc_check = next(c for c in checks if c.check_name == "encryption_at_rest")
        assert enc_check.passed is False
        assert enc_check.severity == "critical"

    def test_rbac_check(self):
        from pygeovision.enterprise.compliance import ComplianceChecker
        checker = ComplianceChecker()
        cfg = self._full_config()
        cfg["rbac_enabled"] = False
        checks = checker.check_all(cfg)
        rbac_check = next(c for c in checks if c.check_name == "rbac_enabled")
        assert rbac_check.passed is False

    def test_no_retention_fails(self):
        from pygeovision.enterprise.compliance import ComplianceChecker
        checker = ComplianceChecker()
        cfg = self._full_config()
        cfg["data_retention_days"] = 0
        checks = checker.check_all(cfg)
        ret_check = next(c for c in checks if c.check_name == "data_retention_policy")
        assert ret_check.passed is False

    def test_report_is_string(self):
        from pygeovision.enterprise.compliance import ComplianceChecker
        checker = ComplianceChecker()
        report  = checker.report(self._full_config())
        assert isinstance(report, str)
        assert "passed" in report.lower()

    def test_checks_have_remediation(self):
        from pygeovision.enterprise.compliance import ComplianceChecker
        checker = ComplianceChecker()
        cfg = dict.fromkeys(self._full_config(), False)
        cfg["data_retention_days"] = 0
        checks = checker.check_all(cfg)
        failed = [c for c in checks if not c.passed]
        assert len(failed) > 0
        for c in failed:
            if c.severity in ("critical", "high"):
                assert c.remediation, f"{c.check_name} has no remediation text"

    def test_total_check_count(self):
        from pygeovision.enterprise.compliance import ComplianceChecker
        checker = ComplianceChecker()
        checks  = checker.check_all(self._full_config())
        # Should have at least 5 checks
        assert len(checks) >= 5


# ══════════════════════════════════════════════════════════════════════════════
# 5. GDPRHandler
# ══════════════════════════════════════════════════════════════════════════════

class TestGDPRHandler:

    def test_detect_email(self):
        from pygeovision.enterprise.compliance import GDPRHandler
        gdpr = GDPRHandler()
        found = gdpr.detect_pii("Contact admin@pygeovision.com for access")
        assert any(item["type"] == "email" for item in found)

    def test_detect_no_pii(self):
        from pygeovision.enterprise.compliance import GDPRHandler
        gdpr = GDPRHandler()
        found = gdpr.detect_pii("Satellite data from Sentinel-2 downloaded successfully.")
        assert len(found) == 0

    def test_anonymise_email(self):
        from pygeovision.enterprise.compliance import GDPRHandler
        gdpr = GDPRHandler()
        text = gdpr.anonymise("Contact admin@example.com for access")
        assert "admin@example.com" not in text
        assert "REDACTED" in text

    def test_anonymise_preserves_non_pii(self):
        from pygeovision.enterprise.compliance import GDPRHandler
        gdpr = GDPRHandler()
        text = gdpr.anonymise("Sentinel-2 data acquired over Accra")
        assert "Sentinel-2" in text
        assert "Accra" in text

    def test_pseudonymise_is_deterministic(self):
        from pygeovision.enterprise.compliance import GDPRHandler
        gdpr = GDPRHandler()
        h1 = gdpr.pseudonymise("alice@example.com")
        h2 = gdpr.pseudonymise("alice@example.com")
        assert h1 == h2

    def test_pseudonymise_different_values(self):
        from pygeovision.enterprise.compliance import GDPRHandler
        gdpr = GDPRHandler()
        h1 = gdpr.pseudonymise("alice@example.com")
        h2 = gdpr.pseudonymise("bob@example.com")
        assert h1 != h2

    def test_pseudonymise_no_plaintext(self):
        from pygeovision.enterprise.compliance import GDPRHandler
        gdpr = GDPRHandler()
        result = gdpr.pseudonymise("alice@example.com")
        assert "alice" not in result
        assert "@" not in result

    def test_pseudonymise_returns_fixed_length(self):
        from pygeovision.enterprise.compliance import GDPRHandler
        gdpr = GDPRHandler()
        r1 = gdpr.pseudonymise("short")
        r2 = gdpr.pseudonymise("a-very-long-email-address@subdomain.example.com")
        assert len(r1) == len(r2) == 16

    def test_pii_match_is_partially_redacted(self):
        from pygeovision.enterprise.compliance import GDPRHandler
        gdpr = GDPRHandler()
        found = gdpr.detect_pii("user@example.com")
        assert found[0]["match"].endswith("***")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
