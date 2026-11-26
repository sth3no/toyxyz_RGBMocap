"""The menus between File and Help: View, Playback, Sequence, People and Motion.

Every entry but two is a control that is already in the window, under the name it has there. Choosing the entry
presses that button or box, and a menu reads its controls' state each time it opens, so nothing here keeps a state of
its own. The two others, Theme and UI Scale, are the settings of File > Settings... and are saved the same way.
"""
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

from .theme import THEMES, UI_SCALES, apply_appearance, current_appearance
from .tooltips import tooltip_text

FRAMES = (('First Frame', 'Home'), ('Previous Frame', 'Left'), ('Next Frame', 'Right'), ('Last Frame', 'End'))
TOOL_KEYS = {'select': 'V', 'razor': 'C', 'hand': 'H', 'zoom': 'Z'}   # the keys of sequence_view.KEYS
VIEWPORT = ('show_grid', 'show_correction_floor', 'show_trajectory', 'show_foot_locks', 'show_names', 'show_cameras')


def usable(control):
    """Whether the control can be used now: enabled, and not on a page or in a group that is hidden."""
    return control.isEnabled() and control.isVisibleTo(control.window())


class Section:
    """The entries of one menu that follow controls of the window. The menu itself is made by the menu bar, as File
    and Help are."""

    def __init__(self, menu):
        self.menu = menu
        # PySide marks a menu's Python object as deleted when a short-lived Python object of the menu's action goes
        # away after someone asked it for its menu (QAction.menu(), as mcp_desktop.py does). Holding the action's
        # object here keeps it the one object there is, so that never happens to these menus.
        self.held = [menu.menuAction()]
        menu.setToolTipsVisible(True)
        self.follow = []   # (action, what brings it up to date)
        menu.aboutToShow.connect(self.refresh)

    def separator(self):
        return self.menu.addSeparator()

    def refresh(self):
        for action, update in self.follow:
            update(action)

    def press(self, control, text=None, key=None):
        """An entry that presses the button `control`; `key` is shown beside it and is the key the window already has."""
        if control is None:return None
        name = text or control.text()
        action = self.menu.addAction(f'{name}\t{key}' if key else name, control.click)
        action.setToolTip(control.toolTip())
        self.follow.append((action, lambda action: action.setEnabled(usable(control))))
        return action

    def box(self, control, text=None):
        """A checked entry for the check box `control`."""
        if control is None:return None
        action = self.menu.addAction(text or control.text(), control.click)
        action.setCheckable(True);action.setToolTip(control.toolTip())

        def update(action):
            action.setEnabled(usable(control));action.setChecked(control.isChecked())
        self.follow.append((action, update))
        return action

    def choice(self, title, combo, tip=''):
        """A submenu with the entries of the list `combo`, the current one marked."""
        if combo is None:return None
        menu = self.menu.addMenu(title);menu.setToolTipsVisible(True);self.held.append(menu.menuAction())
        menu.menuAction().setToolTip(tooltip_text(tip) if tip else combo.toolTip())

        def fill():
            menu.clear();group = QActionGroup(menu)
            for index in range(combo.count()):
                action = menu.addAction(combo.itemText(index), lambda index=index: combo.setCurrentIndex(index))
                action.setCheckable(True);action.setChecked(index == combo.currentIndex());group.addAction(action)
        menu.aboutToShow.connect(fill)
        self.follow.append((menu.menuAction(), lambda action: action.setEnabled(usable(combo))))
        return menu

    def picks(self, title, entries, current, choose, tip=''):
        """A submenu of (label, value) entries, the one equal to current() marked; choosing calls choose(value)."""
        menu = self.menu.addMenu(title);menu.setToolTipsVisible(True);self.held.append(menu.menuAction())
        if tip:menu.menuAction().setToolTip(tooltip_text(tip))
        group, actions = QActionGroup(menu), []
        for label, value in entries:
            action = menu.addAction(label, lambda value=value: choose(value))
            action.setCheckable(True);group.addAction(action);actions.append((action, value))

        def mark():
            now = current()
            for action, value in actions:action.setChecked(value == now)
        menu.aboutToShow.connect(mark)
        return menu


def set_appearance(window, theme=None, scale=None):
    """Save and apply a theme or a UI scale, as OK in File > Settings... does."""
    from .resource_safety import update
    from .workspace_ui import resize_appearance_controls
    old = current_appearance()
    choices = (theme or old[0], scale or old[1])
    if choices == old:return True
    try:
        update({'ui_theme': choices[0], 'ui_scale': choices[1]})
    except (OSError, ValueError) as error:
        QMessageBox.warning(window, 'Settings', f'Could not save settings.\n{error}')
        return False
    apply_appearance(QApplication.instance(), *choices)
    resize_appearance_controls(window)
    return True


def transport(window, name):
    """The transport button whose tooltip starts with `name` (First Frame, Previous Frame, ...)."""
    return next((button for button in window.findChildren(QPushButton, 'transport') if button.toolTip().startswith(name)), None)


def build(w):
    """Add the menus to the window's menu bar, between File and Help."""
    bar = w.menuBar()
    before = next((action for action in bar.actions() if action.text().replace('&', '') == 'Help'), None)
    get = lambda name: getattr(w, name, None)
    w.menus, w.menu_sections = {}, {}

    def section(title):
        made = Section(bar.addMenu(title));w.menus[title], w.menu_sections[title] = made.menu, made
        return made

    view = section('View')
    view.picks('Theme', THEMES, lambda: current_appearance()[0], lambda value: set_appearance(w, theme=value),
               'Choose the app theme. Saved like OK in File > Settings...')
    view.picks('UI Scale', [(f'{value}%', value) for value in UI_SCALES], lambda: current_appearance()[1],
               lambda value: set_appearance(w, scale=value), 'Scales text, buttons and panels relative to the display scale.')
    view.separator()
    view.box(get('video_overlay'))
    view.separator()
    for control in (get('bone_overlay'), get('face_overlay'), *(getattr(w, 'viewport_checks', {}).get(name) for name in VIEWPORT),
                    get('follow_actor')):
        view.box(control)
    view.choice('View Direction', get('view_choice'))
    view.separator()
    console = view.menu.addAction('Console', w.details.click)
    console.setCheckable(True);console.setToolTip(w.details.toolTip())
    view.follow.append((console, lambda action: action.setChecked(w.log.isVisible())))

    playback = section('Playback')
    playback.press(get('play_button'), 'Play / Pause', 'Space')
    for name, key in FRAMES:
        playback.press(transport(w, name), name, key)
    playback.choice('Speed', get('play_speed'))
    playback.separator()
    playback.press(get('timeline_frame_all_button'))
    playback.press(get('reset_trim_button'))
    playback.separator()
    playback.press(get('timeline_help_button'), 'Timeline Help')

    sequence = getattr(w, 'sequence_ui', None)
    if hasattr(sequence, 'view'):   # the app can start without the sequence
        menu = section('Sequence')
        for name in ('add_button', 'cut_button', 'delete_button', 'join_button', 'frame_button', 'review_button'):
            menu.press(getattr(sequence, name, None))
        menu.box(getattr(sequence, 'auto_advance', None))
        tools = getattr(sequence, 'tools', None)
        if tools is not None:
            from .sequence_tools import NAMES
            menu.separator()
            group = QActionGroup(menu.menu)
            for tool, button in tools.buttons.items():
                action = menu.menu.addAction(f'{NAMES[tool]}\t{TOOL_KEYS[tool]}', button.click)
                action.setCheckable(True);action.setToolTip(button.toolTip());group.addAction(action)

                def update(action, tool=tool, button=button):
                    action.setEnabled(usable(button));action.setChecked(tools.tool == tool)
                menu.follow.append((action, update))

    people = section('People')
    people.press(get('find_person_button'))
    people.choice('Person Tracking Model', get('people_tracker'))
    people.box(get('rife_interpolation'))
    people.separator()
    people.press(get('fill_all_button'))
    people.separator()
    people.press(get('people_help_button'), 'People Help')

    motion = section('Motion')
    for name in ('run_button', 'parts_button', 'mesh_button', 'cancel_button'):
        motion.press(get(name))
    motion.separator()
    for name in ('motion_correction', 'track_face', 'resource_safety'):
        motion.box(get(name))
    motion.separator()
    motion.press(get('models_button'))
    motion.press(get('model_help'), 'Model Guide')
    if before is not None:   # Help stays the last menu
        bar.removeAction(before);bar.addAction(before)
    return w.menus
