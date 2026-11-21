import unittest
from mocap import hidden_ranges as h
T=lambda a,b:dict(project=f'p/takes/{a:09d}_{b:09d}_scene_v1',first=a,end=b)

class HiddenRangesTest(unittest.TestCase):
    def test_gap_between_neighbours(self):
        g=h.gaps([T(200,300),T(0,100)])
        self.assertEqual([(x['first'],x['end']) for x in g],[(100,200)])
        self.assertEqual(g[0]['before']['end'],100);self.assertEqual(h.gap_key(g[0]),'100_200')
    def test_overlapping_takes_offer_no_gap(self):
        self.assertEqual(h.gaps([T(0,100),T(50,150)]),[])
        self.assertEqual([(x['first'],x['end']) for x in h.gaps([T(0,100),T(50,150),T(180,260)])],[(150,180)])
    def test_single_take(self):
        self.assertEqual(h.gaps([T(0,100)]),[]);self.assertEqual(h.gaps([]),[])
    def test_smooth_limit(self):
        self.assertIsNone(h.method_issue('smooth',dict(first=100,end=130)))
        self.assertIn('31',h.method_issue('smooth',dict(first=100,end=131)))
    def test_chain_breaks_on_unfilled_gap(self):
        takes=[T(0,100),T(120,200),T(220,300),T(320,400)]
        choices={'100_120':{'method':'smooth'},'200_220':{'method':'none'},'300_320':{'method':'smooth'}}
        self.assertEqual([[t['first'] for t in c] for c in h.chains(takes,choices)],[[0,120],[220,320]])
    def test_chain_skips_gap_over_limit(self):
        self.assertEqual(h.chains([T(0,100),T(140,200)],{'100_140':{'method':'smooth'}}),[])
    def test_label(self):
        self.assertEqual(h.gap_label(dict(first=240,end=360),30),'Frames 241–360 (4.0 s)')
