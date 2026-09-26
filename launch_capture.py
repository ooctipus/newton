"""Reuse the pinned checked Nsight launcher with our admission-only entrypoint."""
import os
from pathlib import Path
import shlex
import sys
from capture_adapter import TOOLS

source = (TOOLS / 'nsys_checked.sh').read_text()
old = 'here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"'
entry = 'uv run python "$here/checked_capture.py"'
assert source.count(old) == source.count(entry) == 1
source = source.replace(old, 'here=' + shlex.quote(str(TOOLS)))
source = source.replace(entry, 'uv run --no-project --python ' + shlex.quote(sys.executable)
                        + ' python ' + shlex.quote(str(Path(__file__).with_name('capture_adapter.py'))))
os.execv('/bin/bash', ['bash', '-c', source, str(TOOLS / 'nsys_checked.sh'), *sys.argv[1:]])
