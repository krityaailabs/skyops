"""Set the deployed handler version in config/handler.toml, portably (sed -i differs between GNU and BSD/macOS).

    python tools/set_handler.py v3
"""
import re
import sys
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "config/handler.toml"


def main(version: str) -> None:
    text = CONFIG.read_text()
    new, n = re.subn(r'^prompt = "[^"]*"', f'prompt = "{version}"', text, flags=re.M)
    if n != 1:
        raise SystemExit("config/handler.toml: expected exactly one `prompt = ...` line")
    CONFIG.write_text(new)
    print(f"config/handler.toml: prompt = \"{version}\"")


if __name__ == "__main__":
    main(sys.argv[1])
