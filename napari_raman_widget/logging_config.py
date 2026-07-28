"""Process-safe logging configuration used before importing pymmcore-plus."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import MutableMapping


def configure_pymmcore_process_log(
    environ: MutableMapping[str, str] | None = None,
    *,
    pid: int | None = None,
) -> Path | None:
    """Give this application process its own pymmcore-plus rotating log.

    pymmcore-plus normally uses one shared rotating file. On Windows, two
    running processes cannot both rename that file during rollover. An
    explicitly configured ``PYMM_LOG_FILE`` is left untouched.
    """
    environ = os.environ if environ is None else environ
    configured = environ.get("PYMM_LOG_FILE")
    if configured is not None:
        disabled_values = {"0", "false", "no", "none"}
        return (
            None
            if configured.casefold() in disabled_values
            else Path(configured)
        )

    local_app_data = environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path(tempfile.gettempdir())
    log_dir = base / "pymmcore-plus" / "pymmcore-plus" / "logs"
    process_id = os.getpid() if pid is None else int(pid)
    log_file = log_dir / f"cns-raman-{process_id}.log"
    environ["PYMM_LOG_FILE"] = str(log_file)
    return log_file
