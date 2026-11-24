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

    def test_a_scrub_paints_without_a_drag(self):
        self.view._press = dict(mode='scrub', x=300)
        self.view._paint_drag(None, self.view._layout())   # a scrub has no 'dragged'; it returned before the painter

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


class LinkViewTest(unittest.TestCase):
    """Two blocks: a with people 1 and 2, b with people 3 and 1; rows from sequence_links.rows."""

    def setUp(self):
        application()
        from mocap.sequence_view import SequenceView
        self.sequence = Sequence()
        self.a = self.sequence.add('C:/videos/zongler.mp4', 10.)
        self.b = self.sequence.add('C:/videos/tanec.mp4', 10.)
        self.states = {self.a.id: state('tracked', [person(1, (0, 300)), person(2, (0, 300))]),
                       self.b.id: state('tracked', [person(3, (0, 300)), person(1, (0, 300))])}
        self.view = SequenceView()
        self.view.resize(1000, 200)
        self.show()
        self.view.show();APP.processEvents()
        self.addCleanup(self.view.deleteLater)
        self.signals = {name: [] for name in ('selected', 'moved', 'resized', 'linked', 'unlinked', 'transition_clicked', 'menu_requested')}
        for name, seen in self.signals.items():
            getattr(self.view, name).connect(lambda *args, seen=seen: seen.append(args))

    def show(self, row_states=None):
        from mocap.sequence_links import rows
        self.view.show_sequence(self.sequence, self.states, rows(self.sequence, self.states), row_states or {})
        self.view.view_start, self.view.view_span = 0., .95 * self.sequence.total();self.view.update()

    def link(self, source, target):
        self.sequence.link(source, target);self.show()

    def test_gap_and_piece_are_drawn(self):
        self.link((self.a.id, 2), (self.b.id, 3))
        self.assertEqual(self.sequence.gap(self.b.id), 1.)
        one, two = self.view.block_rect(self.a.id), self.view.block_rect(self.b.id)
        self.assertGreater(two.left(), one.right())
        piece = self.view.transition_rect((self.b.id, 3))
        self.assertFalse(piece.isEmpty())
        self.assertTrue(one.right() < piece.center().x() < two.left(), (one, piece, two))
        self.assertEqual(piece.top(), self.view.person_rect(self.a.id, 2).top())
        self.assertEqual(piece.top(), self.view.person_rect(self.b.id, 3).top())
        self.assertTrue(self.view.transition_rect((self.b.id, 1)).isEmpty())

    def test_rows_come_from_links(self):
        self.assertNotEqual(self.view.person_rect(self.b.id, 1).top(), self.view.person_rect(self.a.id, 1).top())
        self.link((self.a.id, 1), (self.b.id, 1))
        self.assertEqual(self.view.person_rect(self.b.id, 1).top(), self.view.person_rect(self.a.id, 1).top())
        self.assertEqual(self.view.person_rect(self.b.id, 3).top(), self.view.person_rect(self.a.id, 2).top())
        self.view.show_sequence(self.sequence, self.states)   # without rows: the person index, as stage 1
        self.assertEqual(self.view.person_rect(self.b.id, 3).top(), self.view.person_rect(self.a.id, 1).top())

    def test_drag_bar_onto_row_links(self):
        start = self.view.person_rect(self.b.id, 3).center()
        drag(self.view, start, QPoint(start.x(), self.view.person_rect(self.a.id, 2).center().y()))
        self.assertEqual(self.signals['linked'], [((self.a.id, 2), (self.b.id, 3))])
        self.assertEqual((self.signals['unlinked'], self.signals['selected']), ([], []))

    def test_drag_linked_bar_away_unlinks(self):
        self.states[self.b.id].people.append(person(4, (0, 300)))   # b's third row: no person of a
        self.link((self.a.id, 2), (self.b.id, 3))
        start = self.view.person_rect(self.b.id, 3).center()
        drag(self.view, start, start + QPoint(0, self.view.TRACK_HEIGHT))   # row 2: no person of a
        self.assertEqual(self.signals['unlinked'], [((self.b.id, 3),)])
        drag(self.view, start, QPoint(start.x(), self.view.person_rect(self.a.id, 2).center().y()))   # its own source
        self.assertEqual((self.signals['linked'], len(self.signals['unlinked'])), ([], 1))
        drag(self.view, start, QPoint(start.x(), self.view.person_rect(self.a.id, 1).center().y()))
        self.assertEqual(self.signals['linked'], [((self.a.id, 1), (self.b.id, 3))])

    def test_a_drag_released_off_the_rows_does_nothing(self):   # below the last row, in the ruler or the Video row, outside the widget, beside the block: no link, no unlink
        self.link((self.a.id, 2), (self.b.id, 3))
        start, view = self.view.person_rect(self.b.id, 3).center(), self.view
        for end in (start + QPoint(0, view.TRACK_HEIGHT), QPoint(start.x(), view.height() - 2), QPoint(start.x(), 5),
                    QPoint(start.x(), view.block_rect(self.b.id).center().y()), QPoint(start.x(), view.height() + 40),
                    QPoint(view.width() + 40, view.person_rect(self.a.id, 1).center().y()), QPoint(-40, view.person_rect(self.a.id, 1).center().y()),
                    QPoint(view.block_rect(self.a.id).center().x(), start.y() + view.TRACK_HEIGHT)):
            drag(view, start, end)
            self.assertEqual((self.signals['linked'], self.signals['unlinked']), ([], []), end)
        self.states[self.b.id].people.append(person(4, (0, 300)));self.show()
        drag(view, start, QPoint(view.block_rect(self.a.id).center().x(), start.y() + view.TRACK_HEIGHT))   # a real row, but under the block before
        self.assertEqual(self.signals['unlinked'], [])
        drag(view, start, start + QPoint(0, view.TRACK_HEIGHT));self.assertEqual(self.signals['unlinked'], [((self.b.id, 3),)])

    def test_a_refresh_during_a_drag_resets_the_cursor(self):
        start = self.view.person_rect(self.b.id, 3).center()
        QTest.mousePress(self.view, Qt.LeftButton, Qt.NoModifier, start)
        for dy in (5, 15, 25):QTest.mouseMove(self.view, start + QPoint(0, dy))
        self.assertEqual(self.view.cursor().shape(), Qt.ClosedHandCursor)
        self.show();self.assertEqual(self.view.cursor().shape(), Qt.ArrowCursor);self.assertIsNone(self.view._press)
        QTest.mouseRelease(self.view, Qt.LeftButton, Qt.NoModifier, start + QPoint(0, 25))

    def test_vertical_drag_never_moves_a_block(self):
        start = self.view.person_rect(self.b.id, 3).center()
        drag(self.view, start, start + QPoint(-300, 2 * self.view.TRACK_HEIGHT))   # an empty row, not linked: nothing
        drag(self.view, start, QPoint(start.x() - 300, self.view.block_rect(self.a.id).center().y()))   # the Video row
        drag(self.view, self.view.person_rect(self.a.id, 1).center(), self.view.person_rect(self.b.id, 1).center())
        self.assertEqual((self.signals['moved'], self.signals['resized']), ([], []))
        self.assertEqual((self.signals['linked'], self.signals['unlinked']), ([], []))   # a's people have no block before them

    def test_click_on_piece(self):
        self.link((self.a.id, 2), (self.b.id, 3))
        QTest.mouseClick(self.view, Qt.LeftButton, Qt.NoModifier, self.view.transition_rect((self.b.id, 3)).center())
        self.assertEqual(self.signals['transition_clicked'], [((self.b.id, 3),)])
        self.assertEqual(self.signals['selected'], [])

    def test_right_click_requests_menu(self):
        at = self.view.person_rect(self.b.id, 3).center()
        QTest.mouseClick(self.view, Qt.RightButton, Qt.NoModifier, at)
        self.assertEqual(self.signals['menu_requested'], [(self.b.id, 3, self.view.mapToGlobal(at))])
        QTest.mouseClick(self.view, Qt.RightButton, Qt.NoModifier, self.view.block_rect(self.b.id).center())
        self.assertEqual(len(self.signals['menu_requested']), 1)

    def test_string_person_ids_pass_as_they_are(self):
        self.states[self.b.id] = state('tracked', [person('x', (0, 300))]);self.show()
        start = self.view.person_rect(self.b.id, 'x').center()
        drag(self.view, start, QPoint(start.x(), self.view.person_rect(self.a.id, 2).center().y()))
        self.assertEqual(self.signals['linked'], [((self.a.id, 2), (self.b.id, 'x'))])

    def test_not_editable(self):
        self.link((self.a.id, 2), (self.b.id, 3))
        self.view.editable = False
        start = self.view.person_rect(self.b.id, 3).center()
        drag(self.view, start, start + QPoint(0, self.view.TRACK_HEIGHT))
        drag(self.view, start, QPoint(start.x(), self.view.person_rect(self.a.id, 1).center().y()))
        QTest.mouseClick(self.view, Qt.LeftButton, Qt.NoModifier, self.view.transition_rect((self.b.id, 3)).center())
        QTest.mouseClick(self.view, Qt.RightButton, Qt.NoModifier, start)
        for name in ('linked', 'unlinked', 'transition_clicked', 'menu_requested', 'moved', 'resized'):
            self.assertEqual(self.signals[name], [], name)

    def test_tooltip_texts(self):
        from mocap.sequence_view import piece_text, piece_tip
        self.assertEqual(piece_text(1., 'smooth'), '1.0 s Smooth join')
        self.assertEqual(piece_text(3., 'auto'), '3.0 s Automatic')
        self.assertEqual(piece_tip(3., 'auto', 'Person 2', 'Person 3'),
                         "3.0 s Automatic (Experimental)\nGenerated transition between the end of Person 2's last take and the start of "
                         "Person 3's take, wherever they sit in their blocks. Click to change it.")

    def test_paints_row_states(self):
        from mocap.theme import apply_appearance
        self.link((self.a.id, 2), (self.b.id, 3));self.link((self.a.id, 1), (self.b.id, 1))
        self.sequence.set_method((self.b.id, 1), 'auto')
        try:
            for theme in ('dark', 'light'):
                apply_appearance(APP, theme, 100)
                for row_state in ('linked', 'joined', 'out of date'):
                    self.show({(self.b.id, 3): row_state, (self.b.id, 1): 'joined'})
                    self.assertFalse(self.view.grab().isNull())
                start = self.view.person_rect(self.b.id, 3).center()   # with a link drag in progress
                QTest.mousePress(self.view, Qt.LeftButton, Qt.NoModifier, start)
                QTest.mouseMove(self.view, start + QPoint(5, self.view.TRACK_HEIGHT))
                self.assertFalse(self.view.grab().isNull())
                QTest.mouseRelease(self.view, Qt.LeftButton, Qt.NoModifier, start)
                self.view.frame_all();self.assertFalse(self.view.grab().isNull())
        finally:
            apply_appearance(APP, 'dark', 100)


if __name__ == '__main__':
    unittest.main()
