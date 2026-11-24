"""What the 3D view says about checked people it cannot draw: plain text, so it tests without Qt or a scene."""
import re

RECONSTRUCT = 'Click Reconstruct 3D.'
LINES = 3


FILL_ALL = 'Click Fill all hidden ranges.'


def visible_line(count, hidden=0):
    return f"{count} {'person' if count == 1 else 'people'} visible" + (f', {hidden} hidden' if hidden else '') + ' | Shared camera / world'


def hidden_line(entries, offered):
    """Checked people shown as ghosts now, as (name, first, end) of the hidden range each is in. `offered`: the button
    that fills the ranges can be clicked."""
    if not entries:return ''
    if len(entries) == 1:
        name, first, end = entries[0];line = f'{name}: hidden in frames {first + 1}–{end}.'
    else:line = f"{', '.join(name for name, _, _ in entries)}: hidden at this frame."
    return line + (' ' + FILL_ALL if offered else '')


def take_frames(project):
    """(first, end) of a take folder named `{first:09d}_{end:09d}_...`, or None for any other folder."""
    match = re.match(r'(\d{9})_(\d{9})_', project.name)
    frames = (int(match[1]), int(match[2])) if match else None
    return frames if frames and frames[0] < frames[1] else None


def missing_result(ident, kind, base_current, frames=None):
    """A take without the selected result. `base_current`: its uncorrected result exists, so only correction is missing.
    `frames` names the take the way the take selector does."""
    who = f'Person {ident}' + (f', frames {frames[0] + 1}–{frames[1]}' if frames else '')
    if kind.startswith('corrected_') and base_current:
        return f'{who}: no corrected result for the current settings. Click Apply Correction.'
    return f'{who}: no result for the selected options. {RECONSTRUCT}'


def waiting_people(people, checked):
    """Names of checked people who have no take at all: found, never reconstructed."""
    return [person['name'] for person in people if person['id'] in checked and not person.get('takes')]


def text(warnings, waiting, hidden=''):
    """The line about hidden people, then each note once on its own line, the first few and a count of the rest, then
    the people still to reconstruct."""
    notes = list(dict.fromkeys(warnings))
    lines = ([hidden] if hidden else []) + notes[:LINES] + ([f'… and {len(notes) - LINES} more.'] if len(notes) > LINES else [])
    if waiting:lines.append(f"{', '.join(waiting)}: not reconstructed yet. {RECONSTRUCT}")
    return '\n'.join(lines)
