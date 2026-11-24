"""One sentence for a low-memory stop: the PC's RAM, what is free and which programs hold the most."""
import os

GIB = 1024 ** 3
SHOWN = 4
SMALLEST = .3 * GIB
APP_NAME = 'RGB MoCap'


def largest(processes, app_pids=()):
    """[(name, bytes)] of the biggest programs from (pid, process name, resident bytes) rows, grouped by name."""
    totals, names = {}, {}
    for pid, name, size in processes:
        if not name or not size: continue
        name = APP_NAME if pid in app_pids else (name[:-4] if name.lower().endswith('.exe') else name)
        key = name.lower()
        names.setdefault(key, name)
        totals[key] = totals.get(key, 0) + size
    ranked = sorted(totals.items(), key=lambda item: -item[1])
    return [(names[key], size) for key, size in ranked[:SHOWN] if size >= SMALLEST]


def sentence(total, available, users):
    text = f'This PC has {total / GIB:.1f} GiB of RAM; {available / GIB:.1f} GiB is free now.'
    if users: text += ' Largest users: ' + ', '.join(f'{name} {size / GIB:.1f} GiB' for name, size in users) + '.'
    return text


def collect():
    """The sentence for this moment, or '' when the system cannot be read: the stop itself must not fail."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        own = psutil.Process(os.getpid())
        app_pids = {os.getpid(), *(child.pid for child in own.children(recursive=True))}
        rows = []
        for process in psutil.process_iter(['name', 'memory_info']):
            info = process.info
            rows.append((process.pid, info.get('name'), getattr(info.get('memory_info'), 'rss', None)))
        return sentence(memory.total, memory.available, largest(rows, app_pids))
    except Exception:
        return ''
