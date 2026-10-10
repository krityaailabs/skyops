
import sys as _sys

# Windows consoles and redirected output default to a legacy code page, and the labs print
# arrows and box-drawing characters: force UTF-8 there. No effect on macOS and Linux.
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure") and (getattr(_stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
        _stream.reconfigure(encoding="utf-8")
