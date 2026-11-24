"""Typed start and end times of the trimmed range, in seconds of the loaded clip."""
import re

SECONDS = re.compile(r'\d+(\.\d+)?')
WHOLE = re.compile(r'\d+')
# A shown time is rounded to a tenth, so an end shown up to half of that past the clip still means its end.
ROUNDING = .05


def parse(text):
    """Seconds from "754.5", "12:34.5" or "1:02:03.25"; ValueError with a plain reason otherwise."""
    text = text.strip()
    if not text:raise ValueError('Type a time, for example 12:34.5.')
    if '-' in text:raise ValueError('Times cannot be negative.')
    parts = text.split(':')
    if len(parts) > 3 or not SECONDS.fullmatch(parts[-1]) or not all(WHOLE.fullmatch(part) for part in parts[:-1]):
        raise ValueError(f'"{text}" is not a time. Type it as 12:34.5, 1:02:03 or 754.5 seconds.')
    seconds = float(parts[-1])
    if len(parts) > 1 and seconds >= 60:raise ValueError(f'"{text}" has {parts[-1]} seconds; use 0 to 59.9.')
    if len(parts) == 3 and int(parts[1]) >= 60:raise ValueError(f'"{text}" has {parts[1]} minutes; use 0 to 59.')
    for place, part in enumerate(reversed(parts[:-1]), 1):seconds += int(part) * 60 ** place
    return seconds


def format_time(seconds):
    """m:ss.s below an hour, h:mm:ss.s from an hour on."""
    tenths = round(max(0., seconds) * 10)
    minutes, tenths = divmod(tenths, 600)
    hours, minutes = divmod(minutes, 60)
    second = f'{tenths // 10:02d}.{tenths % 10}'
    return f'{hours}:{minutes:02d}:{second}' if hours else f'{minutes}:{second}'


def frames(start, end, fps, total, minimum):
    """(first, end) frames of a clip of `total` frames for times in seconds; ValueError with a plain reason."""
    if end <= start:raise ValueError('The end must come after the start.')
    first, last = round(start * fps), round(end * fps)
    if last > total + ROUNDING * fps:raise ValueError(f'The clip is only {format_time(total / fps)} long.')
    last = min(total, last)
    if last - first < minimum:raise ValueError(f'A range needs at least {minimum} frames ({format_time(minimum / fps)}).')
    return first, last
