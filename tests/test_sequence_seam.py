"""The seam between two neighbouring blocks of the sequence: the timeline notices the pointer on it, and the card shown
over it offers to join the tracking across it. The view is driven offscreen; the card and the join run in the real main
window over the ExMR scenes of tests/test_sequence_join.py (synthetic finished takes; the queue starts no worker)."""
import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QHelpEvent, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QToolTip

import test_sequence_join as tj
import test_sequence_view as tv
from mocap import sequence_seam

APP, pump = tv.APP, tj.pump


def hover(view, point):
    """The pointer moves over the widget with no button held."""
    APP.sendEvent(view, QMouseEvent(QEvent.MouseMove, QPointF(point), QPointF(view.mapToGlobal(point)), Qt.NoButton, Qt.NoButton, Qt.NoModifier))


class SeamViewTest(tv.ViewCase):
    def setUp(self):
        super().setUp()
        self.seams = []
        self.view.seam_changed.connect(self.seams.append)

    def seam(self, block):return self.view.seam_rect(block.id).center()

    def test_the_pointer_between_two_blocks_is_on_their_seam(self):
        view = self.view
        self.assertIsNone(view.seam)
        hover(view, self.seam(self.b2));self.assertEqual(view.seam, self.b2.id)
        hover(view, self.seam(self.b2) + QPoint(2, 0));self.assertEqual(self.seams, [self.b2.id])   # said once
        hover(view, self.seam(self.b3));self.assertEqual(view.seam, self.b3.id)
        hover(view, view.block_rect(self.b2.id).center());self.assertIsNone(view.seam)
        self.assertEqual(self.seams, [self.b2.id, self.b3.id, None])

    def test_a_seam_reaches_a_little_into_both_blocks_and_the_first_block_has_none_before_it(self):
        view = self.view;seam, left, right = view.seam_rect(self.b2.id), view.block_rect(self.b1.id), view.block_rect(self.b2.id)
        self.assertLess(seam.left(), left.right());self.assertGreater(seam.right(), right.left())
        self.assertLess(seam.width(), 40)
        self.assertEqual((seam.top(), seam.bottom()), (left.top(), left.bottom()))
        self.assertTrue(view.seam_rect(self.b1.id).isEmpty())

    def test_the_seam_of_linked_blocks_covers_their_transition(self):
        view = self.view;self.sequence.link((self.b1.id, 1), (self.b2.id, 1));view.show_sequence(self.sequence, self.states)
        seam, left, right = view.seam_rect(self.b2.id), view.block_rect(self.b1.id), view.block_rect(self.b2.id)
        self.assertGreater(right.left() - left.right(), 10)   # the transition's second
        self.assertLess(seam.left(), left.right());self.assertGreater(seam.right(), right.left())
        hover(view, QPoint((left.right() + right.left()) // 2, seam.center().y()));self.assertEqual(view.seam, self.b2.id)

    def test_only_the_video_row_has_seams(self):
        view = self.view;view.show_block(self.b2.id);point = self.seam(self.b2)
        hover(view, point + QPoint(0, view.TRACK_HEIGHT));self.assertIsNone(view.seam)   # a row of people
        hover(view, QPoint(point.x(), 5));self.assertIsNone(view.seam)                   # the ruler
        hover(view, point);self.assertEqual(view.seam, self.b2.id)

    def test_only_the_selection_tool_on_an_editable_timeline_has_seams(self):
        view = self.view;point = self.seam(self.b2)
        hover(view, point);view.set_tool('razor');self.assertIsNone(view.seam)
        hover(view, point);self.assertIsNone(view.seam)
        view.set_tool('select');hover(view, point);self.assertEqual(view.seam, self.b2.id)
        view.editable = False;self.assertIsNone(view.seam)
        hover(view, point);self.assertIsNone(view.seam)

    def test_a_press_and_leaving_the_timeline_end_the_seam(self):
        view = self.view;point = self.seam(self.b2)
        hover(view, point);QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, point);self.assertIsNone(view.seam)
        QTest.mouseMove(view, point + QPoint(1, 0));self.assertIsNone(view.seam)   # while the button is held
        QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, point + QPoint(1, 0))
        hover(view, point);self.assertEqual(view.seam, self.b2.id)
        APP.sendEvent(view, QEvent(QEvent.Leave));self.assertIsNone(view.seam)
        self.assertEqual(self.seams, [self.b2.id, None, self.b2.id, None])

    def test_a_seam_that_left_the_sequence_ends(self):
        view = self.view;hover(view, self.seam(self.b3));self.sequence.delete(self.b3.id)
        view.show_sequence(self.sequence, self.states);self.assertIsNone(view.seam)

    def test_a_seam_shows_no_tooltip_of_its_block(self):
        view = self.view;inside, point = view.block_rect(self.b2.id).center(), self.seam(self.b2)
        tip = lambda at:APP.sendEvent(view, QHelpEvent(QEvent.ToolTip, at, view.mapToGlobal(at)))
        hover(view, inside);tip(inside);self.assertTrue(QToolTip.isVisible());self.assertIn('tanec.mp4', QToolTip.text())
        hover(view, point);tip(point);self.assertTrue(pump(lambda:not QToolTip.isVisible(), 3))   # Qt hides a tip after a moment

    def test_the_seam_under_the_pointer_is_marked(self):
        view = self.view;plain = view.grab().toImage()
        hover(view, self.seam(self.b2));marked = view.grab().toImage()
        self.assertNotEqual(plain, marked)
        x, y = self.seam(self.b2).x(), self.seam(self.b2).y()
        self.assertNotEqual(plain.pixelColor(x, y), marked.pixelColor(x, y))


@unittest.skipUnless(importlib.util.find_spec('torch') and tj.FFMPEG.is_file(), 'needs torch and the bundled ffmpeg')
class SeamCase(unittest.TestCase):
    setUp = tj.JoinTest.setUp
    scene = tj.JoinTest.scene
    blocks = tj.JoinTest.blocks
    linked = tj.JoinTest.linked
    wait_job = tj.JoinTest.wait_job
    play = tj.JoinTest.play
    play_all = tj.JoinTest.play_all
    person = tj.JoinTest.person
    status = tj.JoinTest.status

    def card(self, block):
        """The card of the seam before `block`, shown as the pointer's rest shows it."""
        offer = self.c.show_seam(block.id);self.assertIsNotNone(offer, self.w.status.text())
        self.assertFalse(self.c.seam_card.isHidden());return self.c.seam_card

    def shown(self, card):
        return [name for name in ('transition', 'open', 'join') if not getattr(card, name + '_button').isHidden()]

    def click(self, button, answer=True):
        with mock.patch.object(self.c, 'ask', return_value=answer) as asked:QTest.mouseClick(button, Qt.LeftButton)
        return asked


class SeamCardTest(SeamCase):
    def test_resting_on_a_seam_shows_its_card_after_a_moment(self):
        first, second = self.linked();c, w = self.c, self.w;card = c.seam_card
        self.assertTrue(card.isHidden());self.assertFalse(c.at_clip_level())
        point = c.view.seam_rect(second.id).center();hover(c.view, point)
        self.assertTrue(card.isHidden())   # not at once: the edge is also where a block is trimmed
        self.assertTrue(pump(lambda:not card.isHidden(), 3))
        self.assertEqual(card.title.text(), 'one.mp4 → two.mp4')
        self.assertEqual(card.text.text(), 'Ben continues Anna. Transition: 1.0 s Smooth join.')
        self.assertEqual(self.shown(card), ['transition', 'join'])
        self.assertEqual((card.join_button.text(), card.transition_button.text()), ('Join tracking', 'Transition...'))
        self.assertIs(card.window(), w)
        above = c.view.mapTo(w, QPoint(point.x(), c.view.TRACK_TOP))
        self.assertLessEqual(card.geometry().bottom(), above.y());self.assertGreater(card.geometry().bottom(), above.y() - 12)
        self.assertLess(card.geometry().left(), above.x());self.assertGreater(card.geometry().right(), above.x())
        self.assertTrue(w.rect().contains(card.geometry()))

    def test_the_card_goes_when_the_pointer_leaves_the_seam_but_not_while_it_is_on_the_card(self):
        first, second = self.linked();c = self.c;card = self.card(second)
        hover(c.view, c.view.seam_rect(second.id).center());hover(c.view, c.view.block_rect(first.id).center())
        self.assertFalse(card.isHidden());card.entered.emit()   # the pointer went from the seam onto the card
        self.assertFalse(pump(card.isHidden, 1))
        card.left.emit();self.assertTrue(pump(card.isHidden, 3))
        card = self.card(second)   # the pointer is on the card when the moment is over: it stays
        with mock.patch.object(card, 'underMouse', return_value=True):
            c._seam_changed(None);self.assertFalse(pump(card.isHidden, 1))
        c._seam_changed(None);self.assertTrue(pump(card.isHidden, 3))

    def test_a_press_or_a_turn_of_the_wheel_on_the_timeline_hides_the_card(self):
        first, second = self.linked();c = self.c;card = self.card(second)
        QTest.mousePress(c.view, Qt.LeftButton, Qt.NoModifier, c.view.seam_rect(second.id).center());self.assertTrue(card.isHidden())
        QTest.mouseRelease(c.view, Qt.LeftButton, Qt.NoModifier, c.view.seam_rect(second.id).center())
        card = self.card(second);tv.wheel(c.view, QPoint(300, 40), 1);self.assertTrue(card.isHidden())

    def test_no_card_while_a_job_runs_inside_a_block_or_with_another_model(self):
        first, second = self.linked();c, w = self.c, self.w;card = self.card(second)
        w.busy = True;c.refresh();self.assertTrue(card.isHidden());self.assertIsNone(c.show_seam(second.id))
        w.busy = False;c.refresh();self.card(second)
        c.open_block(first.id);self.assertIsNone(c.show_seam(second.id));self.assertTrue(card.isHidden())
        c.back();self.card(second)
        self.assertIsNone(c.show_seam(first.id));self.assertTrue(card.isHidden())   # nothing before the first block
        self.card(second);w.model_choice.setCurrentIndex(w.model_choice.findData('gemx'));APP.processEvents()
        self.assertTrue(card.isHidden());self.assertIsNone(c.show_seam(second.id))


class SeamJoinTest(SeamCase):
    def test_join_tracking_joins_the_row_across_the_seam_and_the_card_then_opens_it(self):
        first, second = self.linked();c, w = self.c, self.w;card = self.card(second)
        asked = self.click(card.join_button);asked.assert_called_once_with(tj.QUESTION, 'Join')
        self.assertTrue(card.isHidden())
        job = self.wait_job('take_import');self.assertEqual(Path(tj.arg(job, '--source')), self.roots[1]/tj.TAKE.format(1))
        self.assertIsNone(c.show_seam(second.id))   # busy
        self.play_all()
        self.assertEqual(w.status.text(), 'Joined Anna across 2 blocks. Double-click the row to open it.')
        card = self.card(second)
        self.assertEqual(card.text.text(), 'Ben continues Anna. Joined.');self.assertEqual(self.shown(card), ['transition', 'open'])
        self.assertEqual(card.open_button.text(), 'Open joined animation')
        c.load(second.id);QTest.mouseClick(card.open_button, Qt.LeftButton)
        self.assertTrue(pump(lambda:w.project == (self.roots[0]/'p1/takes/000000000_000000210_fill_v1').resolve(), 10), w.status.text())
        self.assertTrue(c.at_clip_level());self.assertTrue(card.isHidden())

    def test_no_to_the_question_joins_nothing(self):
        first, second = self.linked();c = self.c;card = self.card(second)
        self.click(card.join_button, False)
        self.assertEqual((self.jobs, c.joiner), ([], None));self.assertEqual(self.status((second.id, 1)), 'linked')

    def test_a_block_without_finished_motion_says_what_to_do(self):
        first, second = self.linked();c, w = self.c, self.w
        third = c.place([self.videos[2]])[0];card = self.card(third)
        text = 'three.mp4 has no finished motion for the selected model. Reconstruct it first.'
        self.assertEqual(card.title.text(), 'two.mp4 → three.mp4');self.assertEqual(card.text.text(), text)
        self.assertEqual(self.shown(card), [])
        self.assertFalse(c.join_seam(third.id));self.assertEqual(w.status.text(), text);self.assertEqual(self.jobs, [])

    def test_single_people_who_are_not_linked_are_linked_by_the_join(self):
        first, second = self.linked();c = self.c;target = (second.id, 1)
        self.assertTrue(c.unlink(target));self.assertEqual(c.sequence.links, [])
        card = self.card(second)
        self.assertEqual(card.text.text(), 'Ben will continue Anna.');self.assertEqual(self.shown(card), ['join'])
        self.click(card.join_button)
        self.assertEqual(c.sequence.link_into(*target).source, (first.id, 1))
        self.wait_job('take_import');self.play_all();self.assertEqual(self.status(target), 'joined')

    def test_several_people_are_linked_by_hand_first(self):
        first, second = self.blocks(((1, 'Anna'), (2, 'Ben')), ((1, 'Cleo'), (2, 'Dan')));c = self.c
        self.assertEqual(c.sequence.links, []);card = self.card(second)
        self.assertEqual(card.text.text(), "More than one person here. Zoom in and right-click a person's bar in two.mp4 to choose "
                                           "whom it continues, then join here.")
        self.assertEqual(self.shown(card), []);self.assertFalse(c.join_seam(second.id))
        self.assertTrue(c.link((first.id, 1), (second.id, 2)));self.assertTrue(c.link((first.id, 2), (second.id, 1)))
        card = self.card(second)
        self.assertEqual(card.text.text().splitlines(), ['Dan continues Anna. Transition: 1.0 s Smooth join.',
                                                         'Cleo continues Ben. Transition: 1.0 s Smooth join.'])
        self.assertEqual(self.shown(card), ['join'])   # the transition's method is each link's own: no one button for two
        self.click(card.join_button);self.wait_job('take_import')
        self.assertEqual(sorted(len(chain) for chain in c.joiner.chains), [2, 2])

    def test_a_first_block_reconstructed_without_find_person_gives_the_reason(self):
        first, second = self.blocks(None, ((1, 'Ben'),));c = self.c;card = self.card(second)
        self.assertEqual(card.text.text(), 'Run Find person in one.mp4 first. A block reconstructed without it has no person to continue.')
        self.assertEqual(self.shown(card), [])

    def test_a_row_of_three_blocks_is_joined_whole_from_any_of_its_seams(self):
        first, second, third = self.linked(3);c, w = self.c, self.w;card = self.card(third)
        self.assertEqual(card.text.text().splitlines(), ['Cleo continues Ben. Transition: 1.0 s Smooth join.',
                                                         'This row runs over 3 blocks; all of them are joined together.'])
        self.click(card.join_button);self.wait_job('take_import')
        self.assertEqual(w.status.text(), 'Joining Anna: importing 1 of 2.')

    def test_transition_opens_the_transition_of_the_one_link(self):
        first, second = self.linked();c = self.c;card = self.card(second)
        with mock.patch.object(c, 'edit_transition') as edit:QTest.mouseClick(card.transition_button, Qt.LeftButton)
        edit.assert_called_once_with((second.id, 1));self.assertTrue(card.isHidden())

    def test_a_join_out_of_date_is_offered_again(self):
        first, second = self.linked();c = self.c;self.click(self.card(second).join_button);self.wait_job('take_import');self.play_all()
        self.assertTrue(c.set_transition((second.id, 1), 'smooth', .5), self.w.status.text());card = self.card(second)
        self.assertEqual(card.text.text(), 'Ben continues Anna. The join is out of date. Transition: 0.5 s Smooth join.')
        self.assertEqual(self.shown(card), ['transition', 'join'])


class OfferTest(unittest.TestCase):
    def test_flowhmr_cannot_be_joined(self):
        from mocap.sequence import Sequence
        sequence = Sequence();sequence.model = 'flowhmr'
        a, b = sequence.add('C:/v/one.mp4', 5.), sequence.add('C:/v/two.mp4', 5.)
        states = {a.id:tv.state('tracked', [tv.person(1, (0, 150))]), b.id:tv.state('tracked', [tv.person(1, (0, 150))])}
        offer = sequence_seam.offer(sequence, states, b.id, 'flowhmr')
        self.assertEqual((offer.title, offer.lines, offer.join), ('one.mp4 → two.mp4', ['FlowHMR results cannot be joined.'], False))

    def test_a_missing_video_is_named(self):
        from mocap.sequence import Sequence
        sequence = Sequence();sequence.model = 'exmr'
        a, b = sequence.add('C:/v/one.mp4', 5.), sequence.add('C:/v/two.mp4', 5.)
        states = {a.id:tv.state('missing'), b.id:tv.state('tracked', [tv.person(1, (0, 150))])}
        offer = sequence_seam.offer(sequence, states, b.id, 'exmr')
        self.assertEqual(offer.lines, ['The video file of one.mp4 is missing.']);self.assertFalse(offer.join)
        with self.assertRaises(ValueError):sequence_seam.offer(sequence, states, a.id, 'exmr')


if __name__ == '__main__':
    unittest.main()
