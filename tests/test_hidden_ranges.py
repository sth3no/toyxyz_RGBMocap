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
    def test_chain_ignores_nested_take(self):
        takes=[T(0,100),T(120,300),T(150,200),T(310,400)]
        choices={'100_120':{'method':'smooth'},'300_310':{'method':'smooth'}}
        self.assertEqual([[t['first'] for t in c] for c in h.chains(takes,choices)],[[0,120,310]])
    def test_method_issue_none_and_unknown(self):
        self.assertIsNone(h.method_issue('none',dict(first=0,end=999)))
        with self.assertRaises(ValueError):h.method_issue('bogus',dict(first=0,end=1))
    def test_gaps_unsorted_with_nested_take(self):
        g=h.gaps([T(250,300),T(50,100),T(0,200)])
        self.assertEqual([(x['first'],x['end'],x['before']['first']) for x in g],[(200,250,0)])

    def test_choice_params(self):
        for c in(None,{},{'method':'none'}):self.assertEqual(h.choice_params(c),('none',{}))
        self.assertEqual(h.choice_params({'method':'smooth','person':3}),('smooth',{}))
        self.assertEqual(h.choice_params({'method':'copy','person':2}),('copy',dict(person=2,mirror=False,offset=0)))
        self.assertEqual(h.choice_params({'method':'copy','person':2,'mirror':True,'offset':-600}),('copy',dict(person=2,mirror=True,offset=-600)))
        self.assertEqual(h.choice_params(dict(method='copy',person=2,offset=600))[1]['offset'],600)
        for bad in({'method':'bogus'},{'method':'copy'},{'method':'copy','person':'2'},{'method':'copy','person':True},{'method':'copy','person':2.0},
                {'method':'copy','person':2,'offset':601},{'method':'copy','person':2,'offset':-601},{'method':'copy','person':2,'offset':1.5},
                {'method':'copy','person':2,'offset':True},{'method':'copy','person':2,'mirror':1},{'method':'copy','person':2,'mirror':'yes'},'copy',[1]):
            with self.assertRaises(ValueError,msg=bad):h.choice_params(bad)
    def test_donor_take(self):
        g=dict(first=100,end=120)
        self.assertEqual(h.donor_take([T(0,90),T(95,130)],g,0),T(95,130))
        self.assertEqual(h.donor_take([T(0,91),T(95,130)],g,-30),T(0,91));self.assertIsNone(h.donor_take([T(0,90),T(95,130)],g,-30))
        self.assertIsNone(h.donor_take([T(0,110),T(110,200)],g,0))
        self.assertEqual(h.donor_take([T(50,150),T(60,160)],g,0),T(60,160))
        self.assertEqual(h.donor_take([T(60,160),T(50,150)],g,0),T(50,150))
        self.assertIsNone(h.donor_take([T(0,90)],g,0))
        self.assertIsNone(h.donor_take([],g,0))
    def test_donor_offsets(self):
        g=dict(first=100,end=120)
        self.assertEqual(h.donor_offsets([T(0,100),T(120,200),T(225,313)],g),[(-99,-21),(21,79),(126,192)])
        self.assertEqual(h.donor_offsets([T(0,300),T(50,150)],g),[(-99,179)])                 # overlapping takes merge
        self.assertEqual(h.donor_offsets([T(0,1000)],g),[(-99,600)])                          # clipped to the offset limit
        self.assertEqual(h.donor_offsets([T(100,121)],g),[]);self.assertEqual(h.donor_offsets([T(0,22)],g),[(-99,-99)])
        self.assertEqual(h.donor_offsets('bad',g),[]);self.assertEqual(h.donor_offsets([None,{'first':'a'}],g),[])
        for lo,hi in h.donor_offsets([T(0,100),T(120,200)],g):                                # every hinted offset has a donor take
            for off in(lo,hi):self.assertIsNotNone(h.donor_take([T(0,100),T(120,200)],g,off),off)
            for off in(lo-1,hi+1):self.assertIsNone(h.donor_take([T(0,100),T(120,200)],g,off),off)
    def test_offset_hint(self):
        self.assertEqual(h.offset_hint([(-99,-21),(21,79)]),' Offsets that work: -99 to -21, 21 to 79.')
        self.assertEqual(h.offset_hint([(5,5)]),' Offsets that work: 5.')
        self.assertEqual(h.offset_hint([(-500,-400),(-99,-21),(21,79),(126,192),(300,310)]),' Offsets that work: -99 to -21, 21 to 79, 126 to 192.')
    def test_method_issue_copy(self):
        g=dict(first=100,end=120);people=[dict(id=1,name='Person 1',takes=[T(0,200)]),dict(id=2,name="Person 2",takes=[T(0,91)])]
        c=lambda **k:dict(method='copy',**k)
        ask='Choose a person to copy from.'
        self.assertEqual(h.method_issue('copy',g),ask)
        for ch in(c(),c(person=1),c(person=9),None,{}):self.assertEqual(h.method_issue('copy',g,ch,people,1),ask,ch)
        self.assertEqual(h.method_issue('copy',g,c(person=2),people,1),'Person 2 has no captured motion covering this range with this offset. Offsets that work: -99 to -30.')
        short=[people[0],dict(id=2,name='Person 2',takes=[T(100,121)])]
        self.assertEqual(h.method_issue('copy',g,c(person=2),short,1),'Person 2 has no captured take long enough to cover this range.')
        self.assertIsNone(h.method_issue('copy',g,c(person=2,offset=-30),people,1))
        self.assertIsNone(h.method_issue('copy',g,c(person=1),people,2))
        self.assertIsNone(h.method_issue('smooth',g));self.assertIsNone(h.method_issue('none',g))
    def test_chains_with_copy(self):
        takes=[T(0,100),T(120,200),T(220,300)]
        people=[dict(id=1,takes=takes),dict(id=2,name='Two',takes=[T(0,400)])]
        ch={'100_120':{'method':'smooth'},'200_220':{'method':'copy','person':2}}
        self.assertEqual([[t['first'] for t in c] for c in h.chains(takes,ch,people,1)],[[0,120,220]])
        self.assertEqual(h.chains(takes,ch),[[takes[0],takes[1]]])
        ch['200_220']={'method':'copy','person':1}
        self.assertEqual([[t['first'] for t in c] for c in h.chains(takes,ch,people,1)],[[0,120]])
        for bad in({'method':'copy','person':'x'},{'method':'copy','person':2,'offset':9999},{'method':'copy','person':2,'mirror':1},'copy',5):
            ch['200_220']=bad;self.assertEqual([[t['first'] for t in c] for c in h.chains(takes,ch,people,1)],[[0,120]],bad)
        ch['200_220']={'method':'copy','person':2}
        self.assertEqual(h.chains(takes,ch,[None,5,dict(id=2,takes='x')],1),[[takes[0],takes[1]]])
