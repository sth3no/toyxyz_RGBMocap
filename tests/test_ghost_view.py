"""The 3D view with a checked person inside an unfilled hidden range: the real MotionView offscreen over a small scene.
Offscreen has no OpenGL, so the translucent body itself is not drawn here; what the view decides to draw and say is."""
import os, tempfile, unittest
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from mocap import people_store as ps
from mocap import scene_ghosts
from mocap.scene_data import PeopleScene, SceneTake
from mocap.viewport import MotionView

APP = QApplication.instance() or QApplication([])
FPS = 30.
T = lambda who, a, b: dict(project=f'p{who}/takes/{a:09d}_{b:09d}_scene_v1', first=a, end=b)
PEOPLE = [dict(id=1, name='Anna', checked=True, takes=[T(1, 0, 100), T(1, 120, 200)]), dict(id=2, name='Ben', checked=True, takes=[T(2, 0, 200)])]
TAIL = 'Click Fill all hidden ranges.'


class Frames(np.ndarray):
    def release_windows(self):pass


def joints(frames, x):
    """A standing figure of 22 joints that walks along z; person `x` metres to the side."""
    data = np.zeros((frames, 22, 3));data[:, :, 1] = np.linspace(.9, 1.7, 22);data[:, :, 0] = x
    data[:, :, 2] = np.arange(frames)[:, None]*.01
    return data.view(Frames)


def scene(root):
    built = PeopleScene(Path(root), dict(floor=0.))
    for person in PEOPLE:
        for take in person['takes']:
            frames = take['end']-take['first']
            built.takes.setdefault(person['id'], []).append(SceneTake(
                person['id'], ps.safe_child(root, take['project']), dict(start=take['first']/FPS, fps=FPS, frames=frames), np.eye(4),
                dict(joints=joints(frames, person['id'])), dict(center=np.zeros(3), span=1., height=1.7), None))
    built.finish()
    return built


class GhostViewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.view = MotionView();self.addCleanup(self.view.deleteLater)
        self.view.resize(640, 480)
        self.scene = scene(self.tmp.name)
        self.view.people_mode = True;self.view.set_people_scene(self.scene)
        self.view.scene_visible = {1, 2};self.view.scene_names = {1: 'Anna', 2: 'Ben'};self.view.scene_colors = {1: '#69dbb9', 2: '#b5b5b5'}
        hidden = scene_ghosts.unfilled(self.tmp.name, PEOPLE, 'initial', 'gvhmr')
        self.view.scene_hidden = hidden;self.scene.hidden = hidden

    def seek(self, frame):self.view.scene_time = frame/FPS

    def test_the_view_holds_the_ghost_of_a_hidden_person(self):
        self.seek(110)
        self.assertEqual([(take.ident, frame, span) for take, frame, span in self.view.scene_ghosts()], [(1, 99, (100, 120))])
        self.assertEqual([take.ident for take, _ in self.view.scene_actors()], [2])
        self.seek(150);self.assertEqual(self.view.scene_ghosts(), [])

    def test_a_view_without_a_scene_has_no_ghosts(self):
        self.view.set_people_scene(None);self.assertEqual(self.view.scene_ghosts(), [])

    def test_the_view_says_who_is_hidden_and_names_the_button_only_when_it_can_be_clicked(self):
        self.seek(110)
        actors, found = self.view.scene_actors(), self.view.scene_ghosts()
        self.assertEqual(self.view.scene_hud(actors, found), ('1 person visible, 1 hidden | Shared camera / world', 'Anna: hidden in frames 101–120.'))
        self.view.scene_fill_offered = True
        self.assertEqual(self.view.scene_hud(actors, found)[1], 'Anna: hidden in frames 101–120. ' + TAIL)
        self.seek(150)
        self.assertEqual(self.view.scene_hud(self.view.scene_actors(), self.view.scene_ghosts()), ('2 people visible | Shared camera / world', ''))

    def test_a_scene_with_only_a_ghost_is_painted_without_the_empty_notice(self):
        self.view.scene_visible = {1};self.seek(110)
        self.assertFalse(self.view.grab().isNull())
        self.assertIsNone(self.view._empty_people_notice)
        self.view.scene_visible = set()
        self.view.grab();self.assertIn('No checked person has motion', self.view._empty_people_notice)

    def test_a_drawn_ghost_is_named_as_hidden(self):
        self.seek(110)
        (take, frame, _), = self.view.scene_ghosts()
        self.view.ghosts_presented = {1: (str(take.project), frame)}
        image = QImage(640, 480, QImage.Format_ARGB32);painter = QPainter(image)
        try:self.view.draw_names(painter, self.view.scene_actors())
        finally:painter.end()
        self.assertEqual([name for name, _ in self.view.name_labels_presented], ['Anna (hidden)'])

    def test_a_ghost_that_was_not_drawn_has_no_name(self):
        self.seek(110);self.view.ghosts_presented = {}
        image = QImage(640, 480, QImage.Format_ARGB32);painter = QPainter(image)
        try:self.view.draw_names(painter, self.view.scene_actors())
        finally:painter.end()
        self.assertEqual(self.view.name_labels_presented, [])

    def test_follow_actor_keeps_a_ghost_in_view(self):
        self.view.follow = True;self.view.scene_visible = {1};self.seek(110)
        target = self.view.target()
        # Anna stands where her take ended: x = 1, z = 99 frames * 1 cm.
        self.assertAlmostEqual(float(target[0]), 1.);self.assertAlmostEqual(float(target[2]), .99)


if __name__ == '__main__':
    unittest.main()
