"""From / To fields beside the timeline's Clip label: type the trimmed range of the loaded clip."""
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton
from .clip_range import format_time, frames, parse
from .theme import ui_px
from .tooltips import tooltip_text

WIDEST = '0:00:00.0'


class TimeField(QLineEdit):
    """As wide as the longest time it shows, at any UI scale."""
    def sizeHint(self):
        hint = super().sizeHint()
        return QSize(self.fontMetrics().horizontalAdvance(WIDEST) + ui_px(16), hint.height())

    def minimumSizeHint(self):return self.sizeHint()


def build(w, bar):
    """Add the fields and Set to the timeline's action bar, right after the Clip label."""
    w.clip_from, w.clip_to = TimeField(), TimeField()
    for label, field, which in (('From', w.clip_from, 'Start'), ('To', w.clip_to, 'End')):
        field.setAccessibleName(f'Clip {label}')
        field.setToolTip(tooltip_text(f'{which} of the trimmed range, in minutes and seconds of the loaded clip, '
                                      'for example 12:34.5 or 1:02:03. Press Enter or Set to apply.'))
        field.returnPressed.connect(lambda: apply(w))
        bar.addWidget(QLabel(label));bar.addWidget(field)
    w.clip_set = QPushButton('Set')
    w.clip_set.setToolTip(tooltip_text('Trim the clip to the typed range and show that range in the timeline.'))
    w.clip_set.clicked.connect(lambda: apply(w))
    bar.addWidget(w.clip_set)
    w._clip_shown = ('', '')
    w.slider.trim_enabled_changed.connect(lambda on: enable(w, on))
    enable(w, w.slider.trim_enabled)


def enable(w, on):
    for control in (w.clip_from, w.clip_to, w.clip_set):control.setEnabled(on)


def show(w):
    """Show the current trim; called whenever the trim or the loaded clip changes."""
    if not w.info:
        texts = ('', '')
    else:
        fps = w.info['fps']
        texts = (format_time(w.slider.clip_in / fps), format_time(w.slider.clip_out / fps))
    w.clip_from.setText(texts[0]);w.clip_to.setText(texts[1])
    w._clip_shown = texts


def apply(w):
    """Trim to the typed range; a field left as shown keeps its exact frame. An invalid entry changes nothing."""
    if not w.info or not w.slider.trim_enabled:return
    from .hidden_ranges import typed
    slider, fps = w.slider, w.info['fps']
    try:
        start = slider.clip_in / fps if w.clip_from.text() == w._clip_shown[0] else parse(w.clip_from.text())
        end = slider.clip_out / fps if w.clip_to.text() == w._clip_shown[1] else parse(w.clip_to.text())
        first, last = frames(start, end, fps, slider.maximum() + 1, slider.minimum_clip_frames())
    except ValueError as error:
        w.status.setText(typed('Clip range not changed: ' + str(error)))
        return
    if w.clip_from.text() == w._clip_shown[0]:first = slider.clip_in
    if w.clip_to.text() == w._clip_shown[1]:last = slider.clip_out
    slider.set_clip(first, last)
    slider.show_frames(slider.clip_in, slider.clip_out)
    slider.setValue(slider.clip_in)  # as dragging the start handle does
    show(w)
