"""What the 3D view says about checked people it cannot draw."""
from pathlib import Path
import unittest

from mocap import scene_notes as notes

BUTTON = 'Click Reconstruct 3D.'


class NotesTest(unittest.TestCase):
    def test_one_person_is_singular(self):
        self.assertEqual(notes.visible_line(1), '1 person visible | Shared camera / world')
        self.assertEqual(notes.visible_line(0), '0 people visible | Shared camera / world')
        self.assertEqual(notes.visible_line(4), '4 people visible | Shared camera / world')

    def test_a_missing_result_names_the_button_that_creates_it(self):
        self.assertEqual(notes.missing_result(2, 'handflow_initial', False), 'Person 2: no result for the selected options. ' + BUTTON)
        self.assertEqual(notes.missing_result(2, 'corrected_initial', False), 'Person 2: no result for the selected options. ' + BUTTON)
        self.assertEqual(notes.missing_result(2, 'corrected_initial', True),
                         'Person 2: no corrected result for the current settings. Click Apply Correction.')

    def test_a_missing_result_names_the_take_as_the_take_selector_does(self):
        self.assertEqual(notes.missing_result(2, 'initial', False, (120, 200)),
                         'Person 2, frames 121–200: no result for the selected options. ' + BUTTON)
        self.assertEqual(notes.missing_result(2, 'corrected_initial', True, (0, 100)),
                         'Person 2, frames 1–100: no corrected result for the current settings. Click Apply Correction.')

    def test_frames_are_read_from_the_take_folder(self):
        self.assertEqual(notes.take_frames(Path('p/person_0002/takes/000000120_000000200_scene_v1')), (120, 200))
        self.assertEqual(notes.take_frames(Path('p/person_0002/takes/000000000_000000313_fill_v1')), (0, 313))
        for name in ('gvhmr', 'take_one', '12_x_scene_v1', '000000200_000000120_scene_v1'):
            self.assertIsNone(notes.take_frames(Path('p') / name), name)

    def test_checked_people_without_a_take_are_waiting(self):
        people = [dict(id=1, name='Ann', takes=[dict(project='p')]), dict(id=3, name='Person 3'), dict(id=4, name='Bo', takes=[]),
                  dict(id=5, name='Unchecked')]
        self.assertEqual(notes.waiting_people(people, {1, 3, 4}), ['Person 3', 'Bo'])
        self.assertEqual(notes.waiting_people(people, set()), [])

    def test_text_says_each_thing_once_on_its_own_line_and_names_the_waiting_people(self):
        self.assertEqual(notes.text([], []), '')
        self.assertEqual(notes.text(['a'], []), 'a')
        self.assertEqual(notes.text(['a', 'a', 'b', 'a'], []), 'a\nb')
        self.assertEqual(notes.text(['a', 'a'], ['Person 3', 'Bo']), 'a\nPerson 3, Bo: not reconstructed yet. ' + BUTTON)
        self.assertEqual(notes.text([], ['Person 3']), 'Person 3: not reconstructed yet. ' + BUTTON)

    def test_the_count_line_names_hidden_people_only_when_there_are_some(self):
        self.assertEqual(notes.visible_line(3, 0), '3 people visible | Shared camera / world')
        self.assertEqual(notes.visible_line(3, 1), '3 people visible, 1 hidden | Shared camera / world')
        self.assertEqual(notes.visible_line(1, 2), '1 person visible, 2 hidden | Shared camera / world')

    def test_one_hidden_person_is_named_with_the_frames_of_the_range(self):
        self.assertEqual(notes.hidden_line([('Person 2', 100, 120)], False), 'Person 2: hidden in frames 101–120.')
        self.assertEqual(notes.hidden_line([('Person 2', 100, 120)], True), 'Person 2: hidden in frames 101–120. Click Fill all hidden ranges.')

    def test_several_hidden_people_share_one_line(self):
        entries = [('Person 2', 100, 120), ('Ann', 90, 140)]
        self.assertEqual(notes.hidden_line(entries, False), 'Person 2, Ann: hidden at this frame.')
        self.assertEqual(notes.hidden_line(entries, True), 'Person 2, Ann: hidden at this frame. Click Fill all hidden ranges.')
        self.assertEqual(notes.hidden_line([], True), '')

    def test_the_hidden_line_comes_first_and_does_not_use_up_the_three_notes(self):
        self.assertEqual(notes.text(['a', 'b', 'c', 'd'], ['Bo'], 'Ann: hidden in frames 1–2.'),
                         'Ann: hidden in frames 1–2.\na\nb\nc\n… and 1 more.\nBo: not reconstructed yet. ' + BUTTON)
        self.assertEqual(notes.text([], [], 'Ann: hidden in frames 1–2.'), 'Ann: hidden in frames 1–2.')
        self.assertEqual(notes.text(['a'], [], ''), 'a')

    def test_text_keeps_to_three_notes_and_counts_the_rest(self):
        self.assertEqual(notes.text(['a', 'b', 'c'], []), 'a\nb\nc')
        self.assertEqual(notes.text(['a', 'b', 'c', 'd', 'e', 'a'], ['Bo']), 'a\nb\nc\n… and 2 more.\nBo: not reconstructed yet. ' + BUTTON)


if __name__ == '__main__':
    unittest.main()
