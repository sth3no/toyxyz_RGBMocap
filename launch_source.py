"""Launch editable application code using the private common runtime."""
import os
from pathlib import Path
import sys

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from tools.source_launch import configure, verify
configure(root)
from tools.session_lock import app_lock
with app_lock(root):
    try:
        verify(root)
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
    from mocap.runtime_registry import activate
    activate(root,'gui')
    from app import main
    if '--mcp-stdio' in sys.argv[1:]:
        with os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0) as protocol:
            os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
            raise SystemExit(main(mcp_output=protocol))
    raise SystemExit(main())
