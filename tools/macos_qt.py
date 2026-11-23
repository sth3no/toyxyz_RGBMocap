"""Keep Qt able to find its plugins on macOS.

Qt's plugin scan skips files that carry the macOS "hidden" flag (UF_HIDDEN), so
QApplication then aborts with 'Could not find the Qt platform plugin "cocoa"'.
Some sync/backup tools set that flag on everything inside dot-folders (for
example a checkout below ~/Documents/.../.name/). Only the flag is cleared;
file contents, permissions and other flags stay as they are.
"""
import importlib.util
import os
import stat
from pathlib import Path


def reveal_qt_plugins(plugins=None):
    """Clear UF_HIDDEN below PySide6's plugin folder; return how many entries changed."""
    if plugins is None:
        spec = importlib.util.find_spec('PySide6')
        if spec is None or not spec.origin:
            return 0
        plugins = Path(spec.origin).parent / 'Qt/plugins'
    changed = 0
    for path in (Path(plugins), *Path(plugins).rglob('*')):
        try:
            flags = path.lstat().st_flags
            if flags & stat.UF_HIDDEN:
                os.chflags(path, flags & ~stat.UF_HIDDEN, follow_symlinks=False)
                changed += 1
        except (OSError, AttributeError, NotImplementedError):
            continue  # Unwritable or unsupported: Qt reports a missing plugin itself.
    return changed
