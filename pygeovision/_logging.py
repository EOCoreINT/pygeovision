"""Process-wide logging setup shared by the CLI and the PyGeoVision client.

pygeofetch installs its own handler on the ``"pygeofetch"`` logger
(``HH:MM:SS LEVEL [module] message``). Records from that logger also
propagate to the root logger, where ``logging.basicConfig`` installs a
second handler, so every pygeofetch message was printed twice in two
different formats. The filter below drops the root copy of a pygeofetch
record whenever pygeofetch is already printing it itself, so each message
appears once. It is checked when a record is emitted, so it does not matter
whether pygeofetch's handler is installed before or after this runs.
"""
from __future__ import annotations

import logging

_ROOT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class _SkipRecordsPygeofetchPrintsItself(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "pygeofetch" or record.name.startswith("pygeofetch."):
            return not logging.getLogger("pygeofetch").handlers
        return True


def configure_logging(level: int = logging.INFO, datefmt: str | None = None) -> None:
    """Install the root handler once and de-duplicate pygeofetch's output."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=_ROOT_FORMAT, datefmt=datefmt)
    elif level < root.level:
        root.setLevel(level)          # honour --verbose even if a handler already exists
    for handler in root.handlers:
        if not any(isinstance(f, _SkipRecordsPygeofetchPrintsItself) for f in handler.filters):
            handler.addFilter(_SkipRecordsPygeofetchPrintsItself())
    # pygeofetch's own handler has no level of its own; follow the requested one.
    logging.getLogger("pygeofetch").setLevel(level)
