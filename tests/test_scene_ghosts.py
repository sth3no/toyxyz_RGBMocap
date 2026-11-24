"""Which hidden ranges of a scene are unfilled, and where the 3D view shows a hidden person as a ghost."""
import json, tempfile, unittest
from pathlib import Path
from unittest import mock

import numpy as np

from mocap import people_store as ps
from mocap import scene_ghosts as ghosts
from mocap.scene_data import PeopleScene, SceneTake

FPS = 30.
S = lambda who, a, b: f'p{who}/takes/{a:09d}_{b:09d}_scene_v1'
T = lambda who, a, b: dict(project=S(who, a, b), first=a, end=b)
J = 'p1/takes/000000000_000000200_fill_v1'


def stamp(path):
    s = path.stat();return [s.st_size, s.st_mtime_ns]


def people():
    return [dict(id=1, name='Anna', checked=True, takes=[T(1, 0, 100), T(1, 120, 200), T(1, 260, 300)]),
            dict(id=2, name='Ben', checked=True, takes=[T(2, 50, 300)])]


def scene(root, state, skip=()):
    """The scene as the loader leaves it: one SceneTake per take of people.json that has a result."""
    built = PeopleScene(Path(root), dict(floor=0.))
    for person in state:
        for take in person['takes']:
            if take['project'] in skip:continue
            frames = take['end']-take['first']
            manifest = dict(start=take['first']/FPS, fps=FPS, frames=frames)
            summary = dict(center=np.zeros(3), span=1., height=1.7)
            built.takes.setdefault(person['id'], []).append(
                SceneTake(person['id'], ps.safe_child(root, take['project']), manifest, np.eye(4), {}, summary, None))
    built.finish()
    return built


class UnfilledTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_every_gap_between_two_takes_of_a_person_is_unfilled(self):
        found = ghosts.unfilled(self.root, people(), 'initial', 'gvhmr')
        self.assertEqual([(r['first'], r['end'], r['before']['project']) for r in found[1]], [(100, 120, S(1, 0, 100)), (200, 260, S(1, 120, 200))])
        self.assertNotIn(2, found)  # one take: nothing between takes

    def test_a_model_without_hidden_ranges_has_none(self):
        for model in ('flowhmr', 'prompthmr', 'mamma', 'gemx'):self.assertEqual(ghosts.unfilled(self.root, people(), 'initial', model), {}, model)
        self.assertTrue(ghosts.unfilled(self.root, people(), 'initial', 'exmr'))

    def test_a_current_joined_take_removes_its_range(self):
        state = people();person = state[0]
        (self.root/'input.mp4').write_bytes(b'video')
        for take in person['takes'][:2]:
            d = self.root/take['project'];d.mkdir(parents=True)
            (d/'tracking.npz').write_bytes(b'track');(d/'hmr4d_results.pt').write_bytes(b'result')
        folder = self.root/J;folder.mkdir(parents=True)
        fill = dict(version=1, person=1, kind='initial', first=0, end=200, gaps=[dict(first=100, end=120, method='smooth', params={})],
                    sources=[dict(project=t['project'], first=t['first'], end=t['end'],
                                  files={n: stamp(self.root/t['project']/n) for n in ('tracking.npz', 'hmr4d_results.pt')}) for t in person['takes'][:2]])
        (folder/'fill.json').write_text(json.dumps(fill), encoding='utf-8')
        person['gap_fill'] = {'100_120': {'method': 'smooth'}}
        person['joins'] = [dict(project=J, first=0, end=200, sources=[t['project'] for t in person['takes'][:2]])]
        found = ghosts.unfilled(self.root, state, 'initial', 'gvhmr')
        self.assertEqual([(r['first'], r['end'], r['before']['project']) for r in found[1]], [(200, 260, J)])
        # The joined take provides only the result it was made from; for another result the sources are in use again.
        self.assertEqual([(r['first'], r['end']) for r in ghosts.unfilled(self.root, state, 'footmr_initial', 'gvhmr')[1]], [(100, 120), (200, 260)])

    def test_a_person_whose_takes_cannot_be_read_is_left_out(self):
        state = people();state[0]['takes'] = [dict(project='x')];state.append('not a person')
        self.assertEqual(ghosts.unfilled(self.root, state, 'initial', 'gvhmr'), {})

    def test_count_is_over_checked_people(self):
        found = ghosts.unfilled(self.root, people(), 'initial', 'gvhmr')
        self.assertEqual(ghosts.count(found, {1, 2}), (2, 1))
        self.assertEqual(ghosts.count(found, {2}), (0, 0))
        self.assertEqual(ghosts.count({}, {1}), (0, 0))


class GhostTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name);self.state = people()
        self.hidden = ghosts.unfilled(self.root, self.state, 'initial', 'gvhmr')

    def at(self, frame, visible=(1, 2), built=None):
        built = built or scene(self.root, self.state)
        return [(take.ident, take.project.name, index, span) for take, index, span in ghosts.ghosts(built, frame/FPS, set(visible), self.hidden)]

    def test_a_person_in_a_hidden_range_stands_at_the_last_frame_before_it(self):
        name = Path(S(1, 0, 100)).name
        for frame in (100, 110, 119):self.assertEqual(self.at(frame), [(1, name, 99, (100, 120))], frame)
        self.assertEqual(self.at(230), [(1, Path(S(1, 120, 200)).name, 79, (200, 260))])

    def test_no_ghost_while_a_take_shows_the_person(self):
        for frame in (0, 99, 120, 199, 260, 299):self.assertEqual(self.at(frame), [], frame)

    def test_no_ghost_before_the_first_take_or_after_the_last(self):
        self.assertEqual(self.at(10, visible=(2,)), [])  # Ben starts at frame 50
        self.assertEqual(self.at(300), []);self.assertEqual(self.at(400), [])

    def test_an_unchecked_person_has_no_ghost(self):
        self.assertEqual(self.at(110, visible=(2,)), [])

    def test_no_ghost_when_the_take_before_the_range_is_not_in_the_scene(self):
        built = scene(self.root, self.state, skip=(S(1, 0, 100),))
        self.assertEqual(self.at(110, built=built), [])

    def test_no_ghost_when_another_take_of_the_person_covers_the_time(self):
        # An overlapping take reconstructed from another trim covers the range although people.json lists the gap.
        state = people();state[0]['takes'].append(T(1, 90, 130))
        built = scene(self.root, state)
        self.assertEqual(self.at(110, built=built), [])

    def test_looking_for_ghosts_reads_no_folder(self):
        # The view asks several times in every paint, so the take before a range is resolved once, when the ranges are listed.
        self.assertEqual(self.hidden[1][0]['path'], ps.safe_child(self.root, S(1, 0, 100)))
        built = scene(self.root, self.state)
        with mock.patch.object(Path, 'resolve', side_effect=AssertionError('a folder was read')):
            found = ghosts.ghosts(built, 110/FPS, {1, 2}, self.hidden)
        self.assertEqual([(take.ident, index, span) for take, index, span in found], [(1, 99, (100, 120))])

    def test_a_scene_without_hidden_ranges_has_no_ghosts(self):
        built = scene(self.root, self.state)
        self.assertEqual(ghosts.ghosts(built, 110/FPS, {1, 2}, {}), [])
        self.assertEqual(ghosts.ghosts(built, 110/FPS, {1, 2}, None), [])

    def test_the_scene_keeps_a_ghost_take_among_the_takes_in_use(self):
        built = scene(self.root, self.state);built.hidden = self.hidden
        built.actors_at(110/FPS, {1, 2})
        self.assertEqual(sorted(take.project.name for take in built.active), sorted([Path(S(1, 0, 100)).name, Path(S(2, 50, 300)).name]))
        built.actors_at(150/FPS, {1, 2})
        self.assertEqual(sorted(take.project.name for take in built.active), sorted([Path(S(1, 120, 200)).name, Path(S(2, 50, 300)).name]))


if __name__ == '__main__':
    unittest.main()
