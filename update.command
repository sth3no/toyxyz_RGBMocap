#!/bin/sh
# Double-clickable macOS wrapper for ./update.sh; Terminal shows its messages.
# The braces make sh read the whole body before Git can replace this file.
{
status=0
/bin/sh "$(dirname -- "$0")/update.sh" "$@" || status=$?
if [ -t 0 ]; then
    printf 'Press Return to close this window.'
    read -r _
fi
exit "$status"
}
