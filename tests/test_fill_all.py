"""What one click on Fill all hidden ranges would fill, with which method, and what it says. No Qt."""
import unittest

from mocap import fill_all as fa
from mocap import scene_ghosts

T = lambda who, a, b: dict(project=f'p{who}/takes/{a:09d}_{b:09d}_scene_v1', first=a, end=b)
SETUP = 'Automatic fill is not set up. Use Set up… and install the runtime in Model Manager.'
NOT_READY = 'Reconstruct and prepare both neighbouring takes first.'
EXPERIMENTAL = 'Experimental support. Generated motion is plausible, not what the person did.'


def people():
    return [dict(id=1, name='Anna', checked=True, takes=[T(1, 0, 100), T(1, 120, 200), T(1, 260, 300)]),   # 20 and 60 frames hidden
            dict(id=2, name='Ben', checked=True, takes=[T(2, 0, 90), T(2, 500, 600)]),                    # 410 frames hidden
            dict(id=3, name='Cy', checked=False, takes=[T(3, 0, 50), T(3, 60, 300)])]


def plan(state=None, checked=(1, 2), fps=30, auto_issue=None, check=None, **kw):
    state = state or people()
    hidden = scene_ghosts.unfilled('unused', state, 'initial', 'gvhmr')
    return fa.plan('unused', state, set(checked), 'initial', 'gvhmr', fps, hidden, auto_issue=auto_issue,
                   check=check or (lambda found, choices, ident: None), **kw)


def rows(person):
    return [(g['first'], g['end'], method) for g, method in person.fills], [(g['first'], g['end'], reason) for g, reason in person.skipped]


class PlanTest(unittest.TestCase):
    def test_short_ranges_get_smooth_join_and_longer_ones_automatic(self):
        found = plan()
        self.assertIsNone(found.reason)
        self.assertEqual([p.name for p in found.people], ['Anna', 'Ben'])
        self.assertEqual(rows(found.people[0]), ([(100, 120, 'smooth'), (200, 260, 'auto')], []))
        self.assertEqual(found.people[0].choices, {'100_120': {'method': 'smooth'}, '200_260': {'method': 'auto', 'seed': 0}})

    def test_a_range_over_the_limit_of_automatic_is_not_filled(self):
        self.assertEqual(rows(plan().people[1]), ([], [(90, 500, 'Automatic fill is limited to 290 frames; this range has 410.')]))

    def test_thirty_frames_are_still_a_smooth_join(self):
        state = [dict(id=1, name='Anna', takes=[T(1, 0, 100), T(1, 130, 200), T(1, 231, 300)])]
        self.assertEqual(rows(plan(state, checked=(1,)).people[0])[0], [(100, 130, 'smooth'), (200, 231, 'auto')])

    def test_without_the_generator_a_longer_range_is_not_filled(self):
        self.assertEqual(rows(plan(auto_issue=SETUP).people[0]), ([(100, 120, 'smooth')], [(200, 260, SETUP)]))

    def test_automatic_needs_thirty_frames_a_second(self):
        self.assertEqual(rows(plan(fps=25).people[0]), ([(100, 120, 'smooth')], [(200, 260, 'Automatic fill needs 30 fps takes; this scene has 25.')]))
        self.assertEqual(rows(plan(fps=29.97).people[0])[1], [(200, 260, 'Automatic fill needs 30 fps takes; this scene has 29.97.')])
        self.assertEqual(rows(plan(fps=30.0).people[0])[1], [])

    def test_the_rate_sentence_is_the_one_the_fill_itself_refuses_with(self):
        try:from mocap import take_join
        except Exception as error:self.skipTest(f'take_join cannot be imported here: {error}')
        self.assertEqual(fa.AUTO_RATE, take_join.AUTO_RATE)

    def test_a_method_chosen_by_hand_is_kept(self):
        state = people();state[0]['gap_fill'] = {'100_120': {'method': 'auto', 'seed': 7}, '200_260': {'method': 'copy', 'person': 2, 'mirror': False, 'offset': 0}}
        person = plan(state).people[0]
        # Ben has no take covering frames 199-261, so the copy cannot run and stays as chosen.
        self.assertEqual(rows(person)[0], [(100, 120, 'auto')])
        self.assertEqual([(a, b) for a, b, _ in rows(person)[1]], [(200, 260)])
        self.assertIn('Ben has no captured', rows(person)[1][0][2])
        self.assertEqual(person.choices, state[0]['gap_fill'])

    def test_unchecked_people_are_left_out(self):
        self.assertEqual([p.ident for p in plan(checked=(1, 2, 3)).people], [1, 2, 3])
        self.assertEqual([p.ident for p in plan(checked=(3,)).people], [3])
        self.assertEqual(plan(checked=()).people, [])

    def test_a_filled_range_is_not_planned_again(self):
        state = people();hidden = {1: [dict(first=200, end=260, before=T(1, 0, 200))]}   # 101-120 is filled by a current joined take
        state[0]['gap_fill'] = {'100_120': {'method': 'smooth'}}
        found = fa.plan('unused', state, {1}, 'initial', 'gvhmr', 30, hidden, check=lambda *a: None)
        self.assertEqual(rows(found.people[0]), ([(200, 260, 'auto')], []))
        self.assertEqual(found.people[0].choices, {'100_120': {'method': 'smooth'}, '200_260': {'method': 'auto', 'seed': 0}})

    def test_a_scene_that_cannot_fill_has_the_reason_and_no_people(self):
        reason = 'Hidden ranges need a fixed camera in this version.'
        found = plan(check=lambda found, choices, ident: reason)
        self.assertEqual((found.reason, found.people), (reason, []))

    def test_a_range_whose_takes_are_not_ready_is_not_filled(self):
        def check(found, choices, ident):
            return NOT_READY if any(g['first'] == 200 for g in found) else None
        person = plan(check=check).people[0]
        self.assertEqual(rows(person), ([(100, 120, 'smooth')], [(200, 260, NOT_READY)]))
        self.assertEqual(person.choices, {'100_120': {'method': 'smooth'}})   # no method is recorded for a range that is not filled

    def test_a_chosen_range_that_is_not_ready_stops_the_whole_person(self):
        # The fill of a person reads every range with a method, so one that cannot be read would refuse the others too.
        state = people();state[0]['gap_fill'] = {'200_260': {'method': 'auto', 'seed': 1}}
        def check(found, choices, ident):
            return NOT_READY if any(g['first'] == 200 for g in found) else None
        person = plan(state, check=check).people[0]
        self.assertEqual(rows(person), ([], [(100, 120, NOT_READY), (200, 260, NOT_READY)]))
        self.assertEqual(person.choices, state[0]['gap_fill'])

    def test_totals_count_what_will_be_filled(self):
        self.assertEqual(fa.totals(plan()), (2, 1))
        self.assertEqual(fa.totals(plan(checked=(2,))), (0, 0))


class ApplyTest(unittest.TestCase):
    def test_apply_records_the_methods_and_leaves_the_given_state_alone(self):
        state = dict(people=people());found = plan(state['people'])
        new = fa.apply(state, found)
        self.assertNotIn('gap_fill', state['people'][0])
        self.assertEqual(new['people'][0]['gap_fill'], {'100_120': {'method': 'smooth'}, '200_260': {'method': 'auto', 'seed': 0}})
        self.assertNotIn('gap_fill', new['people'][1]);self.assertNotIn('gap_fill', new['people'][2])   # nothing to fill; not checked


class TextTest(unittest.TestCase):
    def test_the_note_counts_ranges_and_people(self):
        self.assertEqual(fa.summary(1, 1), '1 hidden range of 1 person is not filled.')
        self.assertEqual(fa.summary(3, 1), '3 hidden ranges of 1 person are not filled.')
        self.assertEqual(fa.summary(3, 2), '3 hidden ranges of 2 people are not filled.')

    def test_the_question_lists_each_range_with_its_method(self):
        self.assertEqual(fa.confirm_text(plan(), False),
                         'Fill 2 hidden ranges of 1 person?\n\n'
                         'Anna: frames 101–120 Smooth join, frames 201–260 Automatic (Experimental)\n\n'
                         'Not filled:\nBen, frames 91–500: Automatic fill is limited to 290 frames; this range has 410.\n\n' + EXPERIMENTAL)

    def test_the_question_names_the_correction_that_follows_and_warns_only_about_generated_motion(self):
        state = [dict(id=1, name='Anna', takes=[T(1, 0, 100), T(1, 120, 200)])]
        self.assertEqual(fa.confirm_text(plan(state, checked=(1,)), True),
                         'Fill 1 hidden range of 1 person?\n\nAnna: frames 101–120 Smooth join\n\n'
                         'Apply Correction then runs for all people, because the shared floor of the scene changes.')

    def test_nothing_to_fill_says_why(self):
        found = plan(checked=(2,))
        self.assertEqual(fa.nothing_text(found), 'No hidden range can be filled. Ben, frames 91–500: Automatic fill is limited to 290 frames; this range has 410.')
        self.assertEqual(fa.nothing_text(fa.Plan([], 'Hidden ranges need a fixed camera in this version.')), 'Hidden ranges need a fixed camera in this version.')
        self.assertEqual(fa.nothing_text(fa.Plan([], None)), 'No hidden range is waiting to be filled.')

    def test_the_result_names_who_was_not_filled(self):
        self.assertEqual(fa.result_text(2, []), 'Fill all hidden ranges finished: 2 people filled.')
        self.assertEqual(fa.result_text(1, []), 'Fill all hidden ranges finished: 1 person filled.')
        self.assertEqual(fa.result_text(1, [('Ben', 'It broke.')]), 'Fill all hidden ranges finished: 1 person filled, 1 not. Ben: It broke.')
        self.assertEqual(fa.result_text(0, [('Ben', 'It broke.'), ('Cy', 'No.')]), 'Fill all hidden ranges finished: nobody filled, 2 not. Ben: It broke. Cy: No.')

    def test_a_cancelled_run_says_how_far_it_got(self):
        self.assertEqual(fa.cancelled_text(0), 'Fill all hidden ranges was cancelled. Nobody was filled.')
        self.assertEqual(fa.cancelled_text(1), 'Fill all hidden ranges was cancelled after 1 person.')
        self.assertEqual(fa.cancelled_text(2), 'Fill all hidden ranges was cancelled after 2 people.')


if __name__ == '__main__':
    unittest.main()
