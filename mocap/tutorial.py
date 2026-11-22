"""In-window tutorial: highlights a control, explains it and waits for the user's step.

The chapters are data in tutorial_steps. MainWindow never starts the tutorial by itself; app.py and the Help menu do.
Its state is an installation-local convenience in the Resource Safety config/local.json, never project data.
"""

STATE = 'tutorial'


def load_state():
    """The saved state; {} when nothing usable is saved, None when the settings cannot be read."""
    from .resource_safety import path, read_settings
    try:value = read_settings(path()).get(STATE)
    except (OSError, ValueError):return None
    return value if isinstance(value, dict) else {}


def save_state(**changes):
    from .resource_safety import update
    state = load_state()
    if state is None:return False  # an unreadable file is kept, not replaced
    try:update({STATE: {**state, **changes}})
    except (OSError, ValueError):return False
    return True


def mark_done(key):
    done = (load_state() or {}).get('done')
    done = [k for k in done if isinstance(k, str)] if isinstance(done, list) else []
    return save_state(done=done if key in done else done + [key])
