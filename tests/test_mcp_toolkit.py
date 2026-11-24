import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QMainWindow, QMenu, QPushButton, QSlider, QWidget

from mocap.cli import parser
from mocap.mcp_cli import commands
from mocap.mcp_desktop import DesktopBridge
from tools.rgbmocap_mcp import reply, serve


class ToolsTest(unittest.TestCase):
    def test_cli_translation_retains_repeated_values_and_explicit_false(self):
        tools = {tool.name: tool for tool in commands(parser())}
        self.assertIn('contacts_self_set', tools)
        self.assertIn('models_remove', tools)
        self.assertIn('stream', tools)
        options = dict(video='scene a.mp4', hamer=False, person=[1, 2],
                       sam31_point=[[.1, .2], [.4, .6]])
        args = parser().parse_args(tools['reconstruct'].argv(options))
        self.assertEqual(args.video, Path('scene a.mp4'))
        self.assertEqual(args.person, [1, 2])
        self.assertEqual(args.sam31_point, [[.1, .2], [.4, .6]])
        self.assertIs(args.hamer, False)
        self.assertTrue(args.json)
        with self.assertRaises(ValueError):
            tools['project_inspect'].argv({'project': '--model'})

    def test_line_protocol_keeps_notifications_out_of_stdout(self):
        kit = SimpleNamespace(tools=lambda: [{'name': 'mock'}])
        requests = [
            {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-06-18'}},
            {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
            {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
        ]
        incoming = io.BytesIO(b''.join(json.dumps(row).encode() + b'\n' for row in requests))
        outgoing = io.BytesIO()
        serve(incoming, outgoing, kit)
        head, tools = [json.loads(row) for row in outgoing.getvalue().splitlines()]
        self.assertEqual(head['result']['protocolVersion'], '2025-06-18')
        self.assertEqual(tools['result']['tools'], [{'name': 'mock'}])
        self.assertEqual(reply({'jsonrpc': '2.0', 'method': 'ping', 'id': 3}, kit)['result'], {})

    def test_desktop_dispatch_controls_and_capture_in_qt_thread(self):
        app = QApplication.instance() or QApplication([])
        window = QMainWindow()
        window.project = window.video_path = None
        window.active_result = 'initial'
        window.busy = False
        window.motion_view = None
        root = QWidget(window)
        window.setCentralWidget(root)
        window.slider = QSlider(root)
        window.model_choice = QComboBox(root)
        window.model_choice.addItem('ExMR', 'exmr')
        window.status = QLabel('Ready', root)
        button = QPushButton('User command', root)
        calls = []
        button.clicked.connect(lambda: calls.append('clicked'))
        window.show()
        app.processEvents()
        try:
            bridge = DesktopBridge(window, None, io.BytesIO(), watch=False)
            self.assertEqual(bridge.state()['model'], 'exmr')
            control = next(item for item in bridge.controls()['controls'] if item['label'] == 'User command')
            bridge.invoke(control['id'])
            app.processEvents()
            self.assertEqual(calls, ['clicked'])
            self.assertTrue(bridge.capture()['png'].startswith('iVBORw0KGgo'))
            menu = QMenu('Context', window)
            menu.addAction('Context command', lambda: calls.append('menu'))
            menu.popup(window.mapToGlobal(QPoint(0, 0)))
            app.processEvents()
            action = next(item for item in bridge.controls()['controls'] if item['label'] == 'Context command')
            bridge.invoke(action['id'])
            app.processEvents()
            self.assertEqual(calls, ['clicked', 'menu'])
        finally:
            window.close()
