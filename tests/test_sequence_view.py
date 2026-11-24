"""The sequence timeline at levels 1 and 2 (videos as blocks, rows of people under them), offscreen, through QTest."""
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from mocap.sequence import MIN_SECONDS, Sequence
from mocap.sequence_state import LABELS, Person, State, Take

APP = QApplication.instance() or QApplication([])


def application():
    """The app's own font and style, set once as app.py does."""
    if not getattr(application, 'configured', False):
        from mocap.theme import configure_application
        configure_application(APP);application.configured = True


def state(kind, people=(), frames=300):
    people = list(people)
    return State(kind, 'The reason.' if kind == 'missing' else '', people, frames, 30.,
                 LABELS.get(kind) or ('1 person' if len(people) == 1 else f'{len(people)} people'))


def person(ident, *takes, ready=True):
    return Person(ident, f'Person {ident}', '#d06a5c' if ident == 1 else '#5c8fd0', True,
                  [Take(a, b, ready, False) for a, b in takes])


def wheel(view, point, notches=1, modifiers=Qt.NoModifier):
    event = QWheelEvent(QPointF(point), QPointF(view.mapToGlobal(point)), QPoint(0, 0), QPoint(0, 120 * notches),
                        Qt.NoButton, modifiers, Qt.NoScrollPhase, False)
    APP.sendEvent(view, event)


def drag(view, start, end, steps=4):
    QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, start)
    for i in range(1, steps + 1):
        QTest.mouseMove(view, start + (end - start) * i / steps)
    QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, end)


class ViewCase(unittest.TestCase):
    def setUp(self):
        application()
        from mocap.sequence_view import SequenceView
        self.sequence = Sequence()
        self.b1 = self.sequence.add('C:/videos/zongler.mp4', 20., 0., 10.)
        self.b2 = self.sequence.add('C:/videos/tanec.mp4', 30., 5., 25.)
        self.b3 = self.sequence.add('C:/videos/park.mp4', 8.)
        self.states = {self.b1.id: state('tracked', [person(1, (0, 300))]),
                       self.b2.id: state('partly', [person(1, (0, 600)), person(2, (0, 200), (400, 600), ready=False)], 600),
                       self.b3.id: state('untracked', frames=240)}
        self.view = SequenceView()
        self.view.resize(1000, 200)
        self.view.show_sequence(self.sequence, self.states)
        self.view.show();APP.processEvents()
        self.addCleanup(self.view.deleteLater)
        self.signals = {name: [] for name in ('selected', 'opened', 'scrubbed', 'moved', 'resized', 'removed')}
        for name, seen in self.signals.items():
            getattr(self.view, name).connect(lambda *args, seen=seen: seen.append(args))


class SequenceViewTest(ViewCase):
    def test_levels_follow_zoom(self):
        self.view.frame_all()
        self.assertEqual(self.view.level(), 1)
        self.assertTrue(self.view.person_rect(self.b2.id, 1).isEmpty())
        self.view.show_block(self.b2.id)
        self.assertEqual(self.view.level(), 2)
        self.assertFalse(self.view.person_rect(self.b2.id, 1).isEmpty())
        self.assertFalse(self.view.person_rect(self.b2.id, 2).isEmpty())
        self.assertAlmostEqual(self.view.span(), min(20 / .8, .8 * self.sequence.total()))

    def test_show_block_of_one_block_is_level_2(self):
        one = Sequence();block = one.add('C:/videos/zongler.mp4', 20.)
        self.view.show_sequence(one, {block.id: state('tracked', [person(1, (0, 600))], 600)})
        self.view.show_block(block.id)
        self.assertEqual(self.view.level(), 2)
        self.assertAlmostEqual(self.view.span(), 16.)
        self.assertFalse(self.view.block_rect(block.id).isEmpty())

    def test_click_selects_and_scrubs(self):
        self.view.frame_all()
        QTest.mouseClick(self.view, Qt.LeftButton, Qt.NoModifier, self.view.block_rect(self.b2.id).center())
        self.assertEqual(self.signals['selected'], [(self.b2.id,)])
        t, final = self.signals['scrubbed'][-1]
        self.assertTrue(final)
        self.assertTrue(10 <= t <= 30, t)

    def test_ruler_scrubs(self):
        self.view.frame_all()
        drag(self.view, QPoint(300, 10), QPoint(500, 10))
        scrubbed = self.signals['scrubbed']
        self.assertGreater(len(scrubbed), 2)
        self.assertTrue(all(not final for _, final in scrubbed[:-1]))
        self.assertTrue(scrubbed[-1][1])
        self.assertGreater(scrubbed[-1][0], scrubbed[0][0])
        self.assertEqual(self.signals['selected'], [])

    def test_double_click_opens(self):
        self.view.frame_all()
        QTest.mouseDClick(self.view, Qt.LeftButton, Qt.NoModifier, self.view.block_rect(self.b1.id).center())
        self.assertEqual(self.signals['opened'], [(self.b1.id, None)])
        self.view.show_block(self.b2.id)
        QTest.mouseDClick(self.view, Qt.LeftButton, Qt.NoModifier, self.view.person_rect(self.b2.id, 2).center())
        self.assertEqual(self.signals['opened'][-1], (self.b2.id, 2))

    def test_wheel_in_past_the_limit_opens(self):
        self.view.show_block(self.b2.id)
        at = self.view.block_rect(self.b2.id).center()
        for _ in range(40):
            if self.view.span() <= .5 * 20:break
            wheel(self.view, at)
            self.assertEqual(self.signals['opened'], [])
        self.assertLessEqual(self.view.span(), 10)
        span = self.view.span()
        wheel(self.view, self.view.block_rect(self.b2.id).center())
        self.assertEqual(self.signals['opened'], [(self.b2.id, None)])
        self.assertEqual(self.view.span(), span)

    def test_wheel_out_reaches_level_1_and_shift_pans(self):
        self.view.show_block(self.b2.id)
        start = self.view.view_start
        wheel(self.view, QPoint(500, 40), modifiers=Qt.ShiftModifier)
        self.assertNotEqual(self.view.view_start, start)
        for _ in range(20):wheel(self.view, QPoint(500, 40), -1)
        self.assertEqual(self.view.level(), 1)

    def test_edge_drag_resizes_on_release(self):
        self.view.frame_all()
        rect = self.view.block_rect(self.b1.id)
        start = QPoint(rect.right() - 2, rect.center().y())
        QTest.mousePress(self.view, Qt.LeftButton, Qt.NoModifier, start)
        for dx in (-25, -50, -100):QTest.mouseMove(self.view, start + QPoint(dx, 0))
        self.assertEqual(self.signals['resized'], [])
        QTest.mouseRelease(self.view, Qt.LeftButton, Qt.NoModifier, start + QPoint(-100, 0))
        self.assertEqual(len(self.signals['resized']), 1)
        ident, first, end = self.signals['resized'][0]
        self.assertEqual((ident, first), (self.b1.id, 0.))
        self.assertLess(end, 10)
        self.assertGreaterEqual(end - first, MIN_SECONDS)
        self.assertEqual(self.signals['selected'], [])

    def test_a_small_move_at_an_edge_is_a_click(self):
        from mocap.theme import ui_px
        self.view.frame_all()
        rect = self.view.block_rect(self.b1.id)
        start = QPoint(rect.right() - 2, rect.center().y())
        drag(self.view, start, start + QPoint(-(ui_px(3) - 1), 0), steps=2)
        self.assertEqual(self.signals['resized'], [])
        self.assertEqual(self.signals['selected'], [(self.b1.id,)])
        self.assertTrue(self.signals['scrubbed'][-1][1])
        drag(self.view, start, start + QPoint(-ui_px(3), 0), steps=2)
        self.assertEqual(len(self.signals['resized']), 1)

    def test_edge_drag_stays_inside_the_source(self):
        self.view.frame_all()
        rect = self.view.block_rect(self.b2.id)
        drag(self.view, QPoint(rect.left() + 2, rect.center().y()), QPoint(rect.left() - 600, rect.center().y()))
        drag(self.view, QPoint(rect.right() - 2, rect.center().y()), QPoint(rect.right() + 600, rect.center().y()))
        drag(self.view, QPoint(rect.right() - 2, rect.center().y()), QPoint(rect.left() - 600, rect.center().y()))
        (_, s1, e1), (_, s2, e2), (_, s3, e3) = self.signals['resized']
        self.assertEqual((s1, e1), (0., 25.))
        self.assertEqual((s2, e2), (5., 30.))
        self.assertAlmostEqual(s3, 5.);self.assertAlmostEqual(e3, 5. + MIN_SECONDS)

    def test_body_drag_reorders(self):
        self.view.frame_all()
        one, two = self.view.block_rect(self.b1.id), self.view.block_rect(self.b2.id)
        drag(self.view, one.center(), two.center() + QPoint(10, 0))
        self.assertEqual(self.signals['moved'], [(self.b1.id, 1)])
        self.assertEqual(self.signals['selected'], [])

    def test_delete_key(self):
        self.view.set_current(self.b2.id)
        self.view.setFocus()
        QTest.keyClick(self.view, Qt.Key_Delete)
        self.assertEqual(self.signals['removed'], [(self.b2.id,)])
        self.view.editable = False
        QTest.keyClick(self.view, Qt.Key_Delete)
        self.assertEqual(self.signals['removed'], [(self.b2.id,)])

    def test_tiny_block_stays_clickable(self):
        from mocap.theme import ui_px
        long = Sequence();big = long.add('C:/videos/long.mp4', 3600.);small = long.add('C:/videos/short.mp4', 5.)
        self.view.show_sequence(long, {})
        self.view.frame_all()
        rect = self.view.block_rect(small.id)
        self.assertGreaterEqual(rect.width(), ui_px(6))
        QTest.mouseClick(self.view, Qt.LeftButton, Qt.NoModifier, rect.center())
        self.assertEqual(self.signals['selected'], [(small.id,)])
        QTest.mouseClick(self.view, Qt.LeftButton, Qt.NoModifier, self.view.block_rect(big.id).center())
        self.assertEqual(self.signals['selected'][-1], (big.id,))

    def test_not_editable(self):
        self.view.frame_all()
        self.view.editable = False
        one, two = self.view.block_rect(self.b1.id), self.view.block_rect(self.b2.id)
        drag(self.view, QPoint(one.right() - 2, one.center().y()), QPoint(one.right() - 100, one.center().y()))
        drag(self.view, one.center(), two.center() + QPoint(10, 0))
        self.assertEqual((self.signals['resized'], self.signals['moved']), ([], []))
        QTest.mouseClick(self.view, Qt.LeftButton, Qt.NoModifier, two.center())
        self.assertEqual(self.signals['selected'], [(self.b2.id,)])

    def test_insets_follow_the_ui_scale(self):
        from mocap.theme import apply_appearance, current_appearance, ui_px
        old = current_appearance();self.addCleanup(lambda:apply_appearance(APP, *old))
        apply_appearance(APP, old[0], 200);self.view.refresh_appearance();self.view.show_block(self.b2.id)
        self.assertNotEqual(ui_px(2), 2)
        rect = self.view._layout()[1][2]
        self.assertEqual((rect.top(), rect.height()), (self.view.TRACK_TOP + ui_px(2), self.view.TRACK_HEIGHT - ui_px(5)))
        cell = self.view._cell(rect, 0)
        self.assertEqual((cell.top(), cell.height()), (self.view.TRACK_TOP + self.view.TRACK_HEIGHT + ui_px(2), self.view.TRACK_HEIGHT - ui_px(5)))

    def test_paints_every_state(self):
        from mocap.theme import apply_appearance
        sequence = Sequence();states = {}
        for kind in ('missing', 'untracked', 'found', 'partly', 'tracked'):
            block = sequence.add(f'C:/videos/{kind}.mp4', 12., 1., 9.)
            people = [] if kind in ('missing', 'untracked') else [person(1, (0, 240)), person(2, (30, 120), ready=kind == 'tracked')]
            states[block.id] = state(kind, people, 240)
        self.view.show_sequence(sequence, states)
        self.view.set_current(sequence.blocks[2].id);self.view.set_busy(sequence.blocks[3].id);self.view.set_playhead(20.)
        try:
            for theme in ('dark', 'light'):
                apply_appearance(APP, theme, 100)
                self.view.frame_all();self.assertFalse(self.view.grab().isNull())
                self.view.show_block(sequence.blocks[3].id);self.assertFalse(self.view.grab().isNull())
            self.view.show_sequence(Sequence(), {});self.assertFalse(self.view.grab().isNull())
            self.assertEqual(self.view.level(), 1)
        finally:
            apply_appearance(APP, 'dark', 100)


if __name__ == '__main__':
    unittest.main()
