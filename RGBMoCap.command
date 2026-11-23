#!/bin/sh
# Double-clickable macOS wrapper for ./RGBMoCap.sh; Terminal shows its messages.
status=0
/bin/sh "$(dirname -- "$0")/RGBMoCap.sh" "$@" || status=$?
if [ "$status" -ne 0 ] && [ -t 0 ]; then
    printf 'Press Return to close this window.'
    read -r _
fi
exit "$status"
