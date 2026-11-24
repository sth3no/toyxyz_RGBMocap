"""Run MCP requests in the managed desktop process's Qt thread."""
import base64
import json
from threading import Event, Thread

from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QObject, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QAbstractButton, QAbstractItemView, QAbstractSlider, QApplication,
                               QComboBox, QDoubleSpinBox, QLabel, QLineEdit, QMenu, QPlainTextEdit,
                               QSpinBox, QTabWidget, QTextEdit)


REQUEST_BYTES = 1024 * 1024
EDITORS = (QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit, QTextEdit,
           QAbstractSlider, QTabWidget, QAbstractItemView)


def display(value):
    return value.split('\t', 1)[0].replace('&&', '\0').replace('&', '').replace('\0', '&')


class DesktopBridge(QObject):
    request = Signal(object)
    disconnected = Signal()

    def __init__(self, window, input_stream, output_stream, *, watch=True):
        super().__init__(window)
        self.window = window
        self.output = output_stream
        self.handled = Event()
        self.controls_by_id = {}
        self.generation = 0
        self.request.connect(self.dispatch)
        self.disconnected.connect(window.close)
        if watch:
            Thread(target=self.read, args=(input_stream,), daemon=True).start()

    def read(self, stream):
        while line := stream.readline(REQUEST_BYTES + 1):
            if len(line) > REQUEST_BYTES:
                break
            try:
                item = json.loads(line)
            except (UnicodeError, ValueError):
                break
            if not isinstance(item, dict):
                break
            self.handled.clear()
            self.request.emit(item)
            self.handled.wait()
        self.disconnected.emit()

    def dispatch(self, request):
        response = {'id': request.get('id')}
        try:
            operations = {'state': self.state, 'controls': self.controls, 'invoke': self.invoke,
                          'seek': self.seek, 'capture': self.capture, 'stop': self.stop,
                          'pointer': self.pointer, 'drag': self.drag, 'key': self.key}
            response['result'] = operations[request['action']](**request['arguments'])
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            response['error'] = str(error)
        try:
            self.output.write(json.dumps(response, ensure_ascii=False).encode('utf-8') + b'\n')
            self.output.flush()
        except OSError:
            self.disconnected.emit()
        finally:
            self.handled.set()

    def surface(self):
        return QApplication.activePopupWidget() or QApplication.activeModalWidget() or self.window

    def state(self):
        w = self.window
        project, video = w.project, w.video_path
        return dict(project=str(project) if project else None, video=str(video) if video else None,
                    frame=w.slider.value() + 1 if video else None, result=w.active_result,
                    model=w.model_choice.currentData(), busy=w.busy, status=w.status.text(),
                    modal=self.surface().windowTitle() if self.surface() is not w else None)

    def controls(self):
        root = self.surface()
        entries = []
        self.generation += 1
        self.controls_by_id = {}

        def remember(obj, label, kind, **fields):
            control_id = f'{self.generation}.{len(entries)}'
            self.controls_by_id[control_id] = (root, obj, kind)
            entries.append(dict(id=control_id, label=display(label), kind=kind,
                                enabled=obj.isEnabled(), **fields))

        def menu(actions, path):
            for action in actions:
                if not action.isVisible() or action.isSeparator():
                    continue
                label = ' > '.join((*path, action.text()))
                if action.menu() is not None:
                    menu(action.menu().actions(), (*path, action.text()))
                else:
                    remember(action, label, 'menu', checked=action.isChecked() if action.isCheckable() else None)

        if root is self.window:
            menu(root.menuBar().actions(), ())
        elif isinstance(root, QMenu):
            menu(root.actions(), ())
        for obj in root.findChildren(QObject):
            if not isinstance(obj, (QAbstractButton, QLabel, *EDITORS)) or not obj.isVisibleTo(root):
                continue
            if not obj.isVisible() or obj.window() is not root:
                continue
            label = obj.accessibleName() or obj.objectName()
            if isinstance(obj, (QAbstractButton, QLabel)):
                label = obj.text() or label
            if isinstance(obj, QLineEdit):
                label = label or obj.placeholderText()
            label = (label or type(obj).__name__)[:300]
            if isinstance(obj, QAbstractButton):
                remember(obj, label, 'button', checked=obj.isChecked() if obj.isCheckable() else None)
            elif isinstance(obj, QLabel):
                if obj.text():
                    entries.append(dict(label=label, kind='label'))
            elif isinstance(obj, QComboBox):
                remember(obj, label, 'choice', value=obj.currentIndex(),
                         options=[obj.itemText(i) for i in range(obj.count())])
            elif isinstance(obj, QTabWidget):
                remember(obj, label, 'tab', value=obj.currentIndex(),
                         options=[obj.tabText(i) for i in range(obj.count())])
            elif isinstance(obj, QAbstractItemView):
                model = obj.model()
                if model is not None:
                    remember(obj, label, 'items', options=[str(model.index(i, 0).data())[:300]
                             for i in range(min(100, model.rowCount()))])
            elif isinstance(obj, (QSpinBox, QDoubleSpinBox, QAbstractSlider)):
                remember(obj, label, 'number', value=obj.value(), minimum=obj.minimum(), maximum=obj.maximum())
            elif not obj.isReadOnly() and not (isinstance(obj, QLineEdit) and obj.echoMode() != QLineEdit.Normal):
                remember(obj, label, 'text')
        return dict(window=root.windowTitle(), controls=entries, status=self.state())

    def invoke(self, control_id, value=None):
        try:
            root, obj, kind = self.controls_by_id[control_id]
        except KeyError:
            raise ValueError('Refresh desktop_controls before selecting this control.') from None
        if self.surface() is not root or not obj.isEnabled() or not obj.isVisible():
            raise ValueError('The control is no longer available. Refresh desktop_controls.')
        if kind in {'button', 'menu'}:
            if value is not None:
                raise ValueError('This control takes no value.')
            if kind == 'menu' and isinstance(root, QMenu) and obj in root.actions():
                def activate_menu_item():
                    root.setActiveAction(obj)
                    QTest.keyClick(root, Qt.Key_Return)

                QTimer.singleShot(0, root, activate_menu_item)
            else:
                QTimer.singleShot(0, obj, obj.click if kind == 'button' else obj.trigger)
        elif kind in {'choice', 'tab', 'items'}:
            if type(value) is not int:
                raise ValueError('Use the 0-based index from options.')
            if kind == 'items':
                index = obj.model().index(value, 0)
                if not index.isValid() or not index.flags() & Qt.ItemIsEnabled:
                    raise ValueError('The item is unavailable.')
                obj.scrollTo(index)
                obj.setCurrentIndex(index)
            else:
                available = obj.model().index(value, 0).flags() & Qt.ItemIsEnabled if kind == 'choice' else (
                    0 <= value < obj.count() and obj.isTabEnabled(value))
                if not available:
                    raise ValueError('The option is unavailable.')
                obj.setCurrentIndex(value)
        elif kind == 'number':
            if type(value) not in (int, float) or not obj.minimum() <= value <= obj.maximum():
                raise ValueError('Enter a value within the control range.')
            obj.setValue(value if isinstance(obj, QDoubleSpinBox) else int(value))
            if isinstance(obj, (QSpinBox, QDoubleSpinBox)):
                obj.editingFinished.emit()
        else:
            if not isinstance(value, str):
                raise ValueError('Enter text.')
            if isinstance(obj, QLineEdit):
                obj.setText(value)
                obj.editingFinished.emit()
            else:
                obj.setPlainText(value)
        return {'accepted': True}

    def seek(self, frame):
        if self.surface() is not self.window:
            raise ValueError('Close the current dialog before seeking.')
        if type(frame) is not int or not 1 <= frame <= self.window.slider.maximum() + 1:
            raise ValueError('Frame is outside the current timeline.')
        self.window.slider.setValue(frame - 1)
        return self.state()

    def point(self, x, y):
        root = self.surface()
        if type(x) is not int or type(y) is not int or not root.rect().contains(QPoint(x, y)):
            raise ValueError('The point is outside the captured window.')
        return root, QPoint(x, y)

    def pointer(self, x, y, button='left', clicks=1):
        root, point = self.point(x, y)
        buttons = {'left': Qt.LeftButton, 'right': Qt.RightButton}
        if button not in buttons or clicks not in (1, 2):
            raise ValueError('Choose the left or right button and one or two clicks.')
        target = root.childAt(point) or root
        point = target.mapFrom(root, point) if target is not root else point
        send = QTest.mouseClick if clicks == 1 else QTest.mouseDClick
        QTimer.singleShot(0, target, lambda: send(target, buttons[button], Qt.NoModifier, point))
        return {'accepted': True}

    def drag(self, x, y, to_x, to_y):
        root, first = self.point(x, y)
        _, last = self.point(to_x, to_y)
        target = root.childAt(first) or root
        if target is not root:
            first, last = target.mapFrom(root, first), target.mapFrom(root, last)

        def deliver():
            QTest.mousePress(target, Qt.LeftButton, Qt.NoModifier, first)
            QTest.mouseMove(target, last)
            QTest.mouseRelease(target, Qt.LeftButton, Qt.NoModifier, last)

        QTimer.singleShot(0, target, deliver)
        return {'accepted': True}

    def key(self, shortcut):
        root = self.surface()
        sequence = QKeySequence.fromString(shortcut, QKeySequence.PortableText)
        if sequence.count() != 1 or sequence[0].key() == Qt.Key_unknown:
            raise ValueError('Enter one Qt shortcut such as Space or Ctrl+S.')
        target = root.focusWidget() or root
        combo = sequence[0]
        QTimer.singleShot(0, target, lambda: QTest.keyClick(target, combo.key(), combo.keyboardModifiers()))
        return {'accepted': True}

    def capture(self):
        from .capture import capture_window
        root = self.surface()
        image = capture_window(root) if root is self.window else root.grab()
        data = QByteArray()
        buffer = QBuffer(data)
        buffer.open(QIODevice.WriteOnly)
        if not image.save(buffer, 'PNG'):
            raise RuntimeError('The desktop capture failed.')
        return dict(png=base64.b64encode(bytes(data)).decode('ascii'))

    def stop(self):
        if self.surface() is not self.window:
            raise ValueError('Close the current dialog before closing the application.')
        QTimer.singleShot(0, self.window, self.window.close)
        return {'accepted': True}
