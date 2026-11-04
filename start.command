#!/bin/bash
cd -- "$(dirname -- "$0")" || exit 1
if ! command -v python3 >/dev/null 2>&1; then
    echo "Install Python 3.11+ and run again."
    read -r -p "Press Enter to close..."
    exit 1
fi
python3 start.py "$@"
result=$?
if [ "$result" -ne 0 ]; then
    read -r -p "Startup failed. Press Enter to close..."
fi
exit "$result"
