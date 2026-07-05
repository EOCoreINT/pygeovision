"""
pygeovision.enterprise.audit
==============================
Audit logging for compliance (GDPR, SOC 2, HIPAA).
"""
from __future__ import annotations
import json, logging, time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pygeovision.enterprise.audit")


@dataclass
class AuditEvent:
    event_type:  str
    user_id:     str
    resource:    str
    action:      str
    outcome:     str   # "success" | "failure" | "denied"
    timestamp:   float = field(default_factory=time.time)
    ip_address:  Optional[str] = None
    metadata:    Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class AuditLogger:
    """Append-only structured audit log."""

    def __init__(self, log_path: str = "./audit.jsonl") -> None:
        self._path   = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._events: List[AuditEvent] = []

    def log(self, event: AuditEvent) -> None:
        self._events.append(event)
        with open(self._path, "a") as f:
            f.write(json.dumps(event.to_dict(), default=str) + "\n")
        logger.debug("Audit: %s %s %s → %s", event.user_id, event.action,
                     event.resource, event.outcome)

    def log_action(self, user_id: str, action: str, resource: str,
                    outcome: str = "success", **metadata) -> None:
        self.log(AuditEvent(
            event_type="api_action",
            user_id=user_id, resource=resource,
            action=action, outcome=outcome, metadata=metadata,
        ))

    def query(self, user_id: Optional[str] = None,
               action: Optional[str] = None,
               since: Optional[float] = None) -> List[AuditEvent]:
        result = self._events
        if user_id: result = [e for e in result if e.user_id == user_id]
        if action:  result = [e for e in result if e.action  == action]
        if since:   result = [e for e in result if e.timestamp >= since]
        return result

    def export(self, path: str, format: str = "jsonl") -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if format == "json":
            with open(path, "w") as f:
                json.dump([e.to_dict() for e in self._events], f, indent=2, default=str)
        else:
            with open(path, "w") as f:
                for e in self._events:
                    f.write(json.dumps(e.to_dict(), default=str) + "\n")
        return path
