"""
pygeovision.enterprise
=======================
PyGeoVision Enterprise Edition — authentication, audit, compliance.

Adds role-based access control, audit logging, usage quotas, and
compliance features on top of the open-source core.

    pip install "pygeovision[enterprise]"
"""
from pygeovision.enterprise.audit import AuditEvent, AuditLogger
from pygeovision.enterprise.auth import APIKeyManager, RBACManager, SSOProvider
from pygeovision.enterprise.compliance import ComplianceChecker, GDPRHandler

__all__ = [
    "RBACManager", "APIKeyManager", "SSOProvider",
    "AuditLogger", "AuditEvent",
    "ComplianceChecker", "GDPRHandler",
]
