#!/bin/sh
# Linux and macOS counterpart of RGBMoCap.cmd.
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python="$root/runtime/bin/python3.10"
if [ ! -x "$python" ]; then
    echo "RGB MoCap is not installed. Run ./install.sh first."
    exit 1
fi
cd -- "$root" || exit 1
if [ "$(uname -s)" = Darwin ]; then
    # Qt uses Cocoa (XQuartz's DISPLAY must not select xcb). A Finder-opened
    # .command may lack Homebrew's PATH, where FFmpeg usually lives.
    for dir in /opt/homebrew/bin /usr/local/bin; do
        case ":$PATH:" in
            *":$dir:"*) ;;
            *) [ -d "$dir" ] && PATH="$PATH:$dir" ;;
        esac
    done
    export PATH
# X11/XWayland avoids the Qt OpenGL context failures seen on some Wayland and WSLg desktops.
elif [ -n "${DISPLAY:-}" ] && [ -z "${QT_QPA_PLATFORM:-}" ]; then
    export QT_QPA_PLATFORM=xcb
fi
status=0
"$python" -I -X utf8 -u -B "$root/launch_source.py" "$@" || status=$?
[ "$status" -eq 0 ] || echo "RGB MoCap exited with code $status. See the messages above."
exit "$status"
