#!/bin/sh
# Double-clickable macOS wrapper for ./install.sh; Terminal shows its messages.
status=0
/bin/sh "$(dirname -- "$0")/install.sh" "$@" || status=$?
if [ -t 0 ]; then
    printf 'Press Return to close this window.'
    read -r _
fi
exit "$status"
