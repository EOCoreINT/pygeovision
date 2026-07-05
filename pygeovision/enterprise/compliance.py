"""
pygeovision.enterprise.compliance
===================================
GDPR, SOC 2, and HIPAA compliance utilities.
"""
from __future__ import annotations
import hashlib, logging, re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pygeovision.enterprise.compliance")


@dataclass
class ComplianceCheck:
    check_name:  str
    passed:      bool
    description: str
    severity:    str   # "critical" | "high" | "medium" | "low"
    remediation: str   = ""


class ComplianceChecker:
    """Run compliance checks against a PyGeoVision deployment configuration."""

    def check_all(self, config: Dict[str, Any]) -> List[ComplianceCheck]:
        checks = []
        checks += self._check_data_encryption(config)
        checks += self._check_access_control(config)
        checks += self._check_audit_logging(config)
        checks += self._check_data_retention(config)
        return checks

    def _check_data_encryption(self, cfg: Dict) -> List[ComplianceCheck]:
        return [
            ComplianceCheck(
                "encryption_at_rest",
                passed=cfg.get("encryption_at_rest", False),
                description="All satellite data and results must be encrypted at rest.",
                severity="critical",
                remediation="Enable disk encryption (LUKS/BitLocker) or use encrypted cloud storage.",
            ),
            ComplianceCheck(
                "encryption_in_transit",
                passed=cfg.get("tls_enabled", True),
                description="All data transfers must use TLS 1.2+.",
                severity="critical",
                remediation="Configure HTTPS with TLS 1.2+ on all endpoints; redirect HTTP to HTTPS.",
            ),
        ]

    def _check_access_control(self, cfg: Dict) -> List[ComplianceCheck]:
        return [
            ComplianceCheck(
                "rbac_enabled",
                passed=cfg.get("rbac_enabled", False),
                description="Role-based access control must be configured.",
                severity="high",
                remediation="Enable RBACManager and assign roles to all users.",
            ),
            ComplianceCheck(
                "mfa_required",
                passed=cfg.get("mfa_required", False),
                description="Multi-factor authentication recommended for all admin users.",
                severity="medium",
                remediation="Configure SSO with MFA or use time-based one-time passwords.",
            ),
        ]

    def _check_audit_logging(self, cfg: Dict) -> List[ComplianceCheck]:
        return [
            ComplianceCheck(
                "audit_logging",
                passed=cfg.get("audit_logging", False),
                description="All data access and model inference must be audit-logged.",
                severity="high",
                remediation="Enable AuditLogger and point to immutable log storage.",
            ),
        ]

    def _check_data_retention(self, cfg: Dict) -> List[ComplianceCheck]:
        retention = cfg.get("data_retention_days", 0)
        return [
            ComplianceCheck(
                "data_retention_policy",
                passed=retention > 0,
                description=f"Data retention policy must be defined (current: {retention} days).",
                severity="medium",
                remediation="Set data_retention_days in config; automate deletion with cleanup job.",
            ),
        ]

    def report(self, config: Dict) -> str:
        checks  = self.check_all(config)
        passed  = sum(1 for c in checks if c.passed)
        lines   = [f"Compliance Report: {passed}/{len(checks)} checks passed", ""]
        for c in checks:
            icon = "✓" if c.passed else "✗"
            lines.append(f"  [{c.severity.upper():8}] {icon} {c.check_name}")
            if not c.passed and c.remediation:
                lines.append(f"             → {c.remediation}")
        return "\n".join(lines)


class GDPRHandler:
    """Utilities for GDPR compliance: data subject requests, PII detection."""

    PII_PATTERNS = [
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
        (r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b", "ssn"),
        (r"\b(?:\d[ -]*?){13,16}\b", "credit_card"),
    ]

    def detect_pii(self, text: str) -> List[Dict[str, str]]:
        found = []
        for pattern, pii_type in self.PII_PATTERNS:
            matches = re.findall(pattern, text)
            for m in matches:
                found.append({"type": pii_type, "match": m[:4] + "***"})
        return found

    def anonymise(self, text: str) -> str:
        for pattern, pii_type in self.PII_PATTERNS:
            text = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", text)
        return text

    def pseudonymise(self, value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()[:16]
