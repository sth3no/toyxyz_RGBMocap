import tempfile,unittest
from pathlib import Path
import numpy as np
from mocap import hidden_ranges as h
from mocap import hidden_ranges_state as ui
T=lambda a,b:dict(project=f'p/takes/{a:09d}_{b:09d}_scene_v1',first=a,end=b)
TAKES=[T(0,100),T(120,200),T(260,300)]

class PanelDecisionTest(unittest.TestCase):
    def test_reason_order(self):
        self.assertEqual(ui.disabled_reason('flowhmr',False,'corrected_refined',lambda:False),'Hidden ranges are available for GVHMR and ExMR in this version.')
        self.assertEqual(ui.disabled_reason('exmr',False,'initial',lambda:True),'Hidden ranges need a fixed camera in this version.')
        self.assertEqual(ui.disabled_reason('exmr',False,'refined',lambda:False),'Hidden ranges need a fixed camera in this version.')
        self.assertIsNone(ui.disabled_reason('exmr',True,'initial',lambda:True))
        self.assertIsNone(ui.disabled_reason('gvhmr',False,'initial',lambda:True))
    def test_unsupported_results(self):
        for kind in ('refined','corrected_refined','handflow_initial','corrected_handflow_footmr_initial','footmr_refined'):
            self.assertEqual(ui.disabled_reason('gvhmr',False,kind,lambda:True),'Hidden ranges support the initial and FootMR results in this version.')
        for kind in ('initial','corrected_initial','footmr_initial','corrected_footmr_initial'):
            self.assertIsNone(ui.disabled_reason('gvhmr',False,kind,lambda:True))
    def test_not_ready(self):
        self.assertEqual(ui.disabled_reason('gvhmr',False,'initial',lambda:False),'Reconstruct and prepare both neighbouring takes first.')
    def test_readiness_read_only_when_needed(self):
        calls=[]
        ui.disabled_reason('gvhmr',False,'refined',lambda:calls.append(1));self.assertEqual(calls,[])
    def test_rows(self):
        rows=ui.gap_rows(h.gaps(TAKES),{'100_120':{'method':'smooth'},'200_260':{'method':'smooth'}},30)
        self.assertEqual([(r['key'],r['label'],r['method']) for r in rows],
                         [('100_120','Frames 101–120 (0.7 s)','smooth'),('200_260','Frames 201–260 (2.0 s)','none')])
        # An over-limit Smooth join still shows as Do not fill, with the limit kept for the disabled item.
        self.assertIsNone(rows[0]['issue']);self.assertIsNone(rows[1]['issue'])
        self.assertIsNone(rows[0]['smooth_issue']);self.assertIn('60',rows[1]['smooth_issue'])
    def test_selected_sources(self):
        g=h.gaps(TAKES)
        self.assertEqual(ui.fill_sources(g,{}),[])
        self.assertEqual([t['first'] for t in ui.fill_sources(g,{'100_120':{'method':'smooth'}})],[0,120])
        # A choice over the Smooth join limit is not filled, so its neighbours do not matter.
        self.assertEqual(ui.fill_sources(g,{'200_260':{'method':'smooth'}}),[])
    def test_can_fill(self):
        rows=ui.gap_rows(h.gaps(TAKES),{'100_120':{'method':'smooth'}},30)
        self.assertTrue(ui.can_fill(rows,None));self.assertFalse(ui.can_fill(rows,'reason'))
        self.assertFalse(ui.can_fill(ui.gap_rows(h.gaps(TAKES),{},30),None))
        self.assertFalse(ui.can_fill(ui.gap_rows(h.gaps(TAKES),{'200_260':{'method':'smooth'}},30),None))
    def test_unsupported_project_locks_choices(self):
        for reason in(ui.disabled_reason('flowhmr',False,'initial',lambda:True),ui.disabled_reason('exmr',False,'initial',lambda:True),
                      ui.disabled_reason('gvhmr',False,'refined',lambda:True)):self.assertFalse(ui.editable(reason),reason)
        self.assertTrue(ui.editable(None));self.assertTrue(ui.editable(ui.disabled_reason('gvhmr',False,'initial',lambda:False)))
    def test_malformed_take_path_does_not_raise(self):
        bad=[dict(g,before=dict(g['before'],project='../../outside')) for g in h.gaps(TAKES)]
        reason,manual=ui.panel_state(Path(tempfile.gettempdir())/'scene',bad,{'100_120':{'method':'smooth'}},'gvhmr','initial')
        self.assertFalse(ui.editable(reason));self.assertFalse(manual)
    def test_choose(self):
        choices={}
        ui.choose(choices,'100_120','smooth');self.assertEqual(choices,{'100_120':{'method':'smooth'}})
        ui.choose(choices,'100_120','none');self.assertEqual(choices,{})
        ui.choose(choices,'100_120','none');self.assertEqual(choices,{})
        self.assertRaises(ValueError,ui.choose,choices,'100_120','x');self.assertEqual(choices,{})
    def test_choose_copy(self):
        choices={}
        ui.choose(choices,'100_120','copy',2,True,-30)
        self.assertEqual(choices,{'100_120':{'method':'copy','person':2,'mirror':True,'offset':-30}})
        self.assertIs(type(choices['100_120']['mirror']),bool)
        ui.choose(choices,'100_120','copy');self.assertEqual(choices,{'100_120':{'method':'copy','mirror':False,'offset':0}})
        ui.choose(choices,'100_120','copy',2,1,900);self.assertEqual(choices['100_120'],{'method':'copy','person':2,'mirror':True,'offset':600})
        ui.choose(choices,'100_120','copy',2,0,-900);self.assertEqual(choices['100_120'],{'method':'copy','person':2,'mirror':False,'offset':-600})
        self.assertIs(choices['100_120']['mirror'],False)
        ui.choose(choices,'100_120','smooth');self.assertEqual(choices,{'100_120':{'method':'smooth'}})

PEOPLE=[dict(id=1,name='Anna',takes=TAKES),dict(id=2,name='Ben',takes=[T(0,300)]),dict(id=3,name='',takes=[T(150,300)])]
COPY=lambda **k:{'method':'copy','person':2,'mirror':False,'offset':0,**k}

class CopyRowsTest(unittest.TestCase):
    def rows(self,choices,people=PEOPLE):return ui.gap_rows(h.gaps(TAKES),choices,30,people,1)
    def test_valid_donor(self):
        r=self.rows({'100_120':COPY(mirror=True,offset=-30)})[0]
        self.assertEqual((r['method'],r['person'],r['mirror'],r['offset'],r['issue']),('copy',2,True,-30,None))
        self.assertEqual(r['donors'],[(2,'Ben'),(3,'Person 3')])
    def test_no_person(self):
        choices={};ui.choose(choices,'100_120','copy')
        r=self.rows(choices)[0]
        self.assertEqual((r['method'],r['person'],r['issue']),('copy',None,'Choose a person to copy from.'))
        self.assertEqual(self.rows({'100_120':COPY(person=1)})[0]['issue'],'Choose a person to copy from.')
        self.assertEqual(self.rows({'100_120':COPY(person=9)})[0]['issue'],'Choose a person to copy from.')
    def test_offset_outside_donor(self):
        r=self.rows({'100_120':COPY(offset=200)})[0]
        self.assertEqual((r['method'],r['offset'],r['issue']),('copy',200,'Ben has no captured motion covering this range with this offset.'))
        self.assertEqual(self.rows({'100_120':COPY(person=3)})[0]['issue'],'Person 3 has no captured motion covering this range with this offset.')
    def test_copy_has_no_length_limit(self):
        r=self.rows({'200_260':COPY()})[1]
        self.assertEqual((r['method'],r['issue']),('copy',None));self.assertIn('60',r['smooth_issue'])
    def test_malformed_shows_none(self):
        for bad in ({'method':'copy','person':'2'},{'method':'copy','person':2,'mirror':1},{'method':'copy','person':2,'offset':601},
                    {'method':'copy','person':2,'offset':1.5},{'method':'zz'},'copy',['copy'],{}):
            r=self.rows({'100_120':bad})[0]
            self.assertEqual((r['method'],r['issue']),('none',None),bad)
        self.assertEqual([r['method'] for r in self.rows(['garbage'])],['none','none'])
        for people in (None,'x',[None,'x',dict(id='2',name='Q')],[dict(id=2,name=5,takes='bad')]):
            r=ui.gap_rows(h.gaps(TAKES),{'100_120':COPY()},30,people,1)[0]
            self.assertEqual(r['method'],'copy');self.assertIsNotNone(r['issue'])
    def test_can_fill(self):
        self.assertTrue(ui.can_fill(self.rows({'100_120':COPY()}),None))
        self.assertFalse(ui.can_fill(self.rows({'100_120':COPY()}),'reason'))
        self.assertFalse(ui.can_fill(self.rows({'100_120':COPY(offset=200)}),None))
        self.assertFalse(ui.can_fill(self.rows({'100_120':{'method':'smooth'},'200_260':COPY(person=None)}),None))
        self.assertTrue(ui.can_fill(self.rows({'100_120':{'method':'smooth'},'200_260':COPY()}),None))
    def test_sources_include_donor(self):
        g=h.gaps(TAKES)
        self.assertEqual([t['first'] for t in ui.fill_sources(g,{'100_120':COPY()},PEOPLE,1)],[0,120,0])
        self.assertEqual(ui.fill_sources(g,{'100_120':COPY(offset=200)},PEOPLE,1),[])
        self.assertEqual(ui.fill_sources(g,{'100_120':COPY()}),[])
        self.assertEqual([t['first'] for t in ui.fill_sources(g,{'100_120':{'method':'smooth'}},PEOPLE,1)],[0,120])

class MaskTest(unittest.TestCase):
    def test_runs(self):
        self.assertEqual(ui.fill_runs(np.array([0,1,1,0,0,1,2,0,1],np.uint8)),((1,3),(5,7),(8,9)))
        self.assertEqual(ui.fill_runs(np.array([0,2,2,0],np.uint8)),((1,3),))
        self.assertEqual(ui.fill_runs(np.zeros(5,np.uint8)),());self.assertEqual(ui.fill_runs(None),())
    def test_line(self):
        self.assertEqual(ui.export_line('Anna_filled',np.array([0,1,1,0],np.uint8)),
                         'Anna_filled: 2 of 4 frames are interpolated across hidden ranges.')
        mask=np.zeros(100,np.uint8);mask[10:20]=1;mask[40:60]=2
        self.assertEqual(ui.export_line('P_filled',mask),
                         'P_filled: 30 of 100 frames are filled across hidden ranges (10 interpolated, 20 copied from another person).')
        mask[10:20]=0
        self.assertEqual(ui.export_line('P_filled',mask),
                         'P_filled: 20 of 100 frames are filled across hidden ranges (0 interpolated, 20 copied from another person).')
    def test_export_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);take=root/'takes/j';take.mkdir(parents=True)
            np.save(take/'fill_mask.npy',np.array([0,1,2,2],np.uint8))
            state={'people':[{'id':1,'joins':[{'project':'takes/j'}]}]}
            self.assertEqual(ui.export_lines(root,state,[(str(take),None,None,str(root/'Anna_filled.fbx'))]),
                             ['Anna_filled: 3 of 4 frames are filled across hidden ranges (1 interpolated, 2 copied from another person).'])

if __name__=='__main__':unittest.main()
