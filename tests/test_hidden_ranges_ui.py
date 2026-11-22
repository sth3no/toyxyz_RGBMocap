import json,tempfile,unittest
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
        for kind in ('refined','corrected_refined','footmr_refined','handflow_refined','corrected_handflow_footmr_refined'):
            self.assertEqual(ui.disabled_reason('gvhmr',False,kind,lambda:True),'Hidden ranges do not support HTD Refine results in this version.')
        self.assertEqual(ui.UNSUPPORTED,'Hidden ranges do not support HTD Refine results in this version.')
        for kind in ('initial','corrected_initial','footmr_initial','corrected_footmr_initial',
                     'handflow_initial','corrected_handflow_initial','handflow_footmr_initial','corrected_handflow_footmr_initial'):
            self.assertIsNone(ui.disabled_reason('gvhmr',False,kind,lambda:True),kind)
    def test_a_donor_without_fingers_is_named(self):
        reason=ui.DONOR_NO_FINGERS.format('Anna')
        self.assertEqual(reason,'Anna has no finger motion for this result. Reconstruct that person with HandFlow.')
        self.assertEqual(ui.disabled_reason('gvhmr',False,'handflow_initial',lambda:reason),reason)
        self.assertTrue(ui.editable(reason))
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
        ui.choose(choices,'100_120','none');self.assertEqual(choices,{})   # again: removing a choice that is not there is fine
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
        self.assertEqual((r['method'],r['offset'],r['issue']),('copy',200,'Ben has no captured motion covering this range with this offset. Offsets that work: -99 to 179.'))
        self.assertEqual(self.rows({'100_120':COPY(person=3)})[0]['issue'],'Person 3 has no captured motion covering this range with this offset. Offsets that work: 51 to 179.')
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

AUTO=lambda seed=7:{'method':'auto','seed':seed}
SETUP='Close RGB MoCap and run Setup_Kimodo.cmd to prepare automatic fill.'
LONG=[T(0,100),T(400,500)]                                                    # 300 hidden frames, over the automatic limit

class AutoRowsTest(unittest.TestCase):
    def rows(self,choices,auto_issue=None,takes=TAKES):return ui.gap_rows(h.gaps(takes),choices,30,PEOPLE,1,auto_issue)
    def test_choose(self):
        choices={}
        ui.choose(choices,'100_120','auto');self.assertEqual(choices,{'100_120':{'method':'auto','seed':0}})
        ui.choose(choices,'100_120','auto',2,True,30,7);self.assertEqual(choices,{'100_120':{'method':'auto','seed':7}})
        ui.choose(choices,'100_120','auto',seed=5000);self.assertEqual(choices['100_120'],{'method':'auto','seed':999})
        ui.choose(choices,'100_120','auto',seed=-3);self.assertEqual(choices['100_120'],{'method':'auto','seed':0})
        ui.choose(choices,'100_120','copy',2,seed=7);self.assertEqual(choices['100_120'],COPY())   # other methods ignore the seed
    def test_saved_choice(self):
        r=self.rows({'100_120':AUTO()})[0]
        self.assertEqual((r['method'],r['seed'],r['person'],r['mirror'],r['offset'],r['issue'],r['auto_issue']),('auto',7,None,False,0,None,None))
        self.assertEqual(self.rows({'100_120':COPY()})[0]['seed'],0)
        r=self.rows({'100_120':AUTO()},SETUP)[0]
        self.assertEqual((r['method'],r['seed'],r['issue'],r['auto_issue']),('auto',7,SETUP,SETUP))
        self.assertEqual(self.rows({},SETUP)[1]['auto_issue'],SETUP)                # every row carries it for the disabled item
    def test_over_limit(self):
        r=self.rows({'100_400':AUTO()},None,LONG)[0]
        sentence='Automatic fill is limited to 290 frames; this range has 300.'
        self.assertEqual((r['method'],r['auto_issue'],r['issue']),('auto',sentence,sentence))
        self.assertEqual(self.rows({},SETUP,LONG)[0]['auto_issue'],sentence)        # the range's own limit comes first
    def test_setup_flag(self):
        self.assertFalse(self.rows({'100_120':AUTO()})[0]['setup'])                  # no issue
        self.assertFalse(self.rows({'100_400':AUTO()},SETUP,LONG)[0]['setup'])       # the range's own limit is not setup
        self.assertTrue(self.rows({'100_120':AUTO()},SETUP)[0]['setup'])             # the runtime or model is missing
        self.assertFalse(self.rows({'100_120':AUTO()},SETUP)[1]['setup'])            # only a chosen Automatic row needs it
        self.assertFalse(self.rows({'100_120':{'method':'smooth'}},SETUP)[0]['setup'])
        self.assertFalse(self.rows({'100_120':COPY(person=None)},SETUP)[0]['setup'])  # a copy's issue is not setup
    def test_malformed_shows_none(self):
        for bad in ({'method':'auto','seed':1000},{'method':'auto','seed':-1},{'method':'auto','seed':'7'},{'method':'auto','seed':7.0},{'method':'auto','seed':True}):
            r=self.rows({'100_120':bad})[0]
            self.assertEqual((r['method'],r['seed'],r['issue']),('none',0,None),bad)
        self.assertEqual(self.rows({'100_120':{'method':'auto'}})[0]['method'],'auto')   # a missing seed is 0
    def test_can_fill(self):
        self.assertTrue(ui.can_fill(self.rows({'100_120':AUTO()}),None))
        self.assertFalse(ui.can_fill(self.rows({'100_120':AUTO()},SETUP),None))    # a setup issue stops the fill
        self.assertFalse(ui.can_fill(self.rows({'100_400':AUTO()},None,LONG),None))
        self.assertTrue(ui.can_fill(self.rows({'100_120':{'method':'smooth'}},SETUP),None))   # the issue only stops Automatic rows
    def test_sources(self):
        g=h.gaps(TAKES)
        self.assertEqual([t['first'] for t in ui.fill_sources(g,{'200_260':AUTO()},PEOPLE,1)],[120,260])
        self.assertEqual([t['first'] for t in ui.fill_sources(g,{'200_260':AUTO()})],[120,260])
        self.assertEqual(ui.fill_sources(h.gaps(LONG),{'100_400':AUTO()}),[])
    def test_pending_state(self):
        g=h.gaps(TAKES);choices={'100_120':COPY(offset=200),'200_260':AUTO()};keep=json.dumps(choices)
        self.assertEqual(ui.pending_state(g,choices,{'100_120':(2,False,0)},None,PEOPLE,1),({'100_120':None,'200_260':None},True))
        self.assertEqual(ui.pending_state(g,choices,{'100_120':(2,False,0)},None,PEOPLE,1,SETUP),({'100_120':None,'200_260':SETUP},False))
        self.assertEqual(json.dumps(choices),keep)
    def test_choose_description(self):
        choices={}
        ui.choose(choices,'100_120','auto',seed=7,text='  A person\n waves  with\tthe right hand. ')
        self.assertEqual(choices['100_120'],{'method':'auto','seed':7,'text':'A person waves with the right hand.'})
        ui.choose(choices,'100_120','auto',seed=7,text=' \n ');self.assertEqual(choices['100_120'],{'method':'auto','seed':7})   # emptied: removed
        ui.choose(choices,'100_120','auto',seed=7,text='a'*199+'  b'+'c'*50)        # cut to 200, without the cut's trailing space
        self.assertEqual(choices['100_120']['text'],'a'*199)
        ui.choose(choices,'100_120','auto',seed=7,text='x'*300);self.assertEqual(choices['100_120']['text'],'x'*200)
        self.assertEqual(h.choice_params(choices['100_120']),('auto',{'seed':7,'text':'x'*200}))
        ui.choose(choices,'100_120','smooth',text='waves');self.assertEqual(choices['100_120'],{'method':'smooth'})   # other methods ignore it
    def test_control_characters_are_not_saved(self):
        # The stored choice is the text the join reads and the text worker is given.
        choices={};ui.choose(choices,'100_120','auto',seed=7,text='A\x00 person\x07 wa​ves\x1b.')
        self.assertEqual(choices['100_120']['text'],'A person waves.')
        self.assertEqual(h.choice_params(choices['100_120'])[1]['text'],choices['100_120']['text'])
        self.assertEqual(h.normalise_text(choices['100_120']['text']),choices['100_120']['text'])
        self.assertEqual(ui.description('\x00\x01 '),'')
    def test_description_rows(self):
        TEXT='Get the text adapters.'
        r=self.rows({'100_120':dict(AUTO(),text='Waves.')})[0]
        self.assertEqual((r['text'],r['issue'],r['setup']),('Waves.',None,False))
        self.assertEqual(self.rows({'100_120':AUTO()})[0]['text'],'')
        self.assertEqual(self.rows({'100_120':COPY()})[0]['text'],'')
        rows=lambda choice,auto_issue=None,takes=TAKES,key='100_120':ui.gap_rows(h.gaps(takes),{key:choice},30,PEOPLE,1,auto_issue,TEXT)
        r=rows(dict(AUTO(),text='Waves.'))[0];self.assertEqual((r['issue'],r['setup']),(TEXT,True))
        r=rows(AUTO())[0];self.assertEqual((r['issue'],r['setup']),(None,False))                 # no text: no text issue
        r=rows(dict(AUTO(),text='Waves.'),SETUP)[0];self.assertEqual((r['issue'],r['setup']),(SETUP,True))   # runtime first
        sentence='Automatic fill is limited to 290 frames; this range has 300.'
        r=rows(dict(AUTO(),text='Waves.'),SETUP,LONG,'100_400')[0];self.assertEqual((r['issue'],r['setup']),(sentence,False))   # limit first
        self.assertEqual(self.rows({'100_120':dict(AUTO(),text=7)})[0]['method'],'none')   # malformed
        self.assertFalse(ui.can_fill(rows(dict(AUTO(),text='Waves.')),None))
        self.assertTrue(ui.can_fill(rows(AUTO()),None))
    def test_description_leaves_sources_and_pending_alone(self):
        g=h.gaps(TAKES);choices={'100_120':COPY(offset=200),'200_260':dict(AUTO(),text='Waves.')};keep=json.dumps(choices)
        self.assertEqual([t['first'] for t in ui.fill_sources(g,choices,PEOPLE,1)],[120,260])
        self.assertEqual(ui.pending_state(g,choices,{'100_120':(2,False,0)},None,PEOPLE,1),({'100_120':None,'200_260':None},True))
        self.assertEqual(ui.pending_state(g,choices,{'100_120':(2,False,0)},None,PEOPLE,1,None,'T'),({'100_120':None,'200_260':'T'},False))
        self.assertEqual(json.dumps(choices),keep)
    def test_panel_state_reads_the_neighbours(self):
        from unittest import mock
        root=Path(tempfile.gettempdir())/'scene';g=h.gaps(TAKES);read=[]
        with mock.patch('mocap.motion_tools.selected_export',lambda p,kind:read.append(p) or p):
            self.assertIsNone(ui.panel_state(root,g,{'200_260':AUTO()},'gvhmr','initial',PEOPLE,1)[0])
        self.assertEqual(sorted(read),sorted(root/TAKES[i]['project'] for i in (1,2)))

class PendingAndReadinessTest(unittest.TestCase):
    def test_pending_offset_is_checked_without_saving(self):
        g=h.gaps(TAKES);choices={'100_120':COPY(offset=200)};keep=json.dumps(choices)
        self.assertEqual(ui.pending_state(g,choices,{},None,PEOPLE,1),({'100_120':'Ben has no captured motion covering this range with this offset. Offsets that work: -99 to 179.','200_260':None},False))
        self.assertEqual(ui.pending_state(g,choices,{'100_120':(2,False,0)},None,PEOPLE,1),({'100_120':None,'200_260':None},True))
        self.assertEqual(json.dumps(choices),keep)                                   # nothing saved
        self.assertFalse(ui.pending_state(g,choices,{'100_120':(2,False,0)},'reason',PEOPLE,1)[1])
        both={'100_120':COPY(),'200_260':COPY()}
        self.assertEqual(ui.pending_state(g,both,{'200_260':(2,False,100)},None,PEOPLE,1),
                         ({'100_120':None,'200_260':'Ben has no captured motion covering this range with this offset. Offsets that work: -199 to 39.'},False))
        self.assertEqual(ui.pending_state(g,both,{'200_260':(None,False,0)},None,PEOPLE,1)[0]['200_260'],'Choose a person to copy from.')
    def test_donor_take_not_ready(self):
        from unittest import mock
        root=Path(tempfile.gettempdir())/'scene';g=h.gaps(TAKES);donor=PEOPLE[1]['takes'][0]['project']
        def run(waiting):
            with mock.patch('mocap.motion_tools.selected_export',lambda p,kind:None if any(p==root/w for w in waiting) else p):
                return ui.panel_state(root,g,{'100_120':COPY()},'gvhmr','initial',PEOPLE,1)[0]
        self.assertIsNone(run([]))
        self.assertEqual(run([TAKES[0]['project']]),'Reconstruct and prepare both neighbouring takes first.')
        self.assertEqual(run([donor]),"Ben's take must be reconstructed and prepared before it can be copied from.")
        self.assertEqual(run([donor,TAKES[1]['project']]),"Ben's take must be reconstructed and prepared before it can be copied from.")
        self.assertTrue(ui.editable(run([donor])))                                  # the donor or offset can still be changed
        self.assertEqual(ui.disabled_reason('gvhmr',False,'initial',lambda:'X'),'X')
        self.assertEqual(ui.fill_sources(g,{'100_120':COPY(person=9)},PEOPLE,1),[])  # an unknown donor does not raise
    def test_a_donor_without_fingers_is_named_by_the_panel(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()/'scene';g=h.gaps(TAKES);donor=root/PEOPLE[1]['takes'][0]['project']
            for t in (*TAKES[:2],PEOPLE[1]['takes'][0]):(root/t['project']).mkdir(parents=True,exist_ok=True)
            ready={(root/t['project'],k) for t in TAKES[:2] for k in ('initial','handflow_initial')}|{(donor,'initial')}   # current body results everywhere
            def run(kind):
                with mock.patch('mocap.motion_tools.selected_export',lambda p,k:p if (p,k) in ready else None):
                    return ui.panel_state(root,g,{'100_120':COPY()},'gvhmr',kind,PEOPLE,1)[0]
            self.assertIsNone(run('initial'))
            reason=run('handflow_initial')
            self.assertEqual(reason,'Ben has no finger motion for this result. Reconstruct that person with HandFlow.');self.assertTrue(ui.editable(reason))
            self.assertEqual(run('corrected_handflow_initial'),reason)
            ready.discard((donor,'initial'))   # without the body result the donor is simply not ready
            self.assertEqual(run('handflow_initial'),"Ben's take must be reconstructed and prepared before it can be copied from.")
            ready.add((donor,'handflow_initial'));ready.add((donor,'initial'));self.assertIsNone(run('handflow_initial'))

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
    def test_line_with_generated_frames(self):
        mask=np.zeros(300,np.uint8);mask[10:30]=1;mask[100:140]=3
        self.assertEqual(ui.export_line('P1',mask),'P1: 60 of 300 frames are filled across hidden ranges (20 interpolated, 40 generated).')
        mask[10:30]=0
        self.assertEqual(ui.export_line('P1',mask),'P1: 40 of 300 frames are filled across hidden ranges (40 generated).')
        mask[10:30]=1;mask[50:55]=2
        self.assertEqual(ui.export_line('P1',mask),'P1: 65 of 300 frames are filled across hidden ranges (20 interpolated, 5 copied from another person, 40 generated).')
        mask[10:30]=0
        self.assertEqual(ui.export_line('P1',mask),'P1: 45 of 300 frames are filled across hidden ranges (5 copied from another person, 40 generated).')
    def test_export_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);take=root/'takes/j';take.mkdir(parents=True)
            np.save(take/'fill_mask.npy',np.array([0,1,2,2],np.uint8))
            state={'people':[{'id':1,'joins':[{'project':'takes/j'}]}]}
            self.assertEqual(ui.export_lines(root,state,[(str(take),None,None,str(root/'Anna_filled.fbx'))]),
                             ['Anna_filled: 3 of 4 frames are filled across hidden ranges (1 interpolated, 2 copied from another person).'])

class TakeInUseTest(unittest.TestCase):
    """Which take the view shows, and which joined takes the selector lists, after a hidden range choice changes."""
    def setUp(self):
        import test_take_joins as tj
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root,self.state,self.person=tj.build(self.tmp.name);self.S,self.J=tj.S,tj.J
    def show(self,project,frame=None,kind='initial'):return ui.take_to_show(self.root,self.person,kind,self.root/project,frame)
    def test_stays_on_a_take_in_use(self):
        self.assertIsNone(self.show(self.J,50))
        del self.person['gap_fill'];self.assertIsNone(self.show(self.S(0,100),50))
    def test_leaves_a_source_the_join_replaced(self):
        self.assertEqual(self.show(self.S(0,100),50),self.J)
    def test_leaves_a_join_switched_off(self):
        del self.person['gap_fill']
        self.assertEqual(self.show(self.J,10),self.S(0,100));self.assertEqual(self.show(self.J,150),self.S(120,200))
        self.assertEqual(self.show(self.J,105),self.S(0,100));self.assertEqual(self.show(self.J,110),self.S(120,200))  # inside the gap: the nearer take
        self.assertEqual(self.show(self.J),self.S(120,200));self.assertEqual(self.show(self.J,5000),self.S(120,200))
    def test_leaves_a_join_that_lacks_the_result(self):
        self.assertEqual(self.show(self.J,10,'htd_refined'),self.S(0,100))
    def test_other_folders_and_bad_records_stay(self):
        self.assertIsNone(self.show('p1',10));self.assertIsNone(self.show('elsewhere',10))
        self.assertIsNone(ui.take_to_show(self.root,dict(id=1,takes='bad',joins=[5]),'initial',self.root/self.J,1))
    def unlisted(self,people=None,kind='initial'):return ui.unlisted_joins(self.root,self.person,kind,people)
    def test_join_in_use_is_listed(self):
        self.assertEqual(self.unlisted(),set())
    def test_join_switched_off_is_not_listed(self):
        del self.person['gap_fill'];self.assertEqual(self.unlisted(),{self.J})
    def test_out_of_date_join_that_fill_would_rewrite_is_listed(self):
        (self.root/self.S(120,200)/'hmr4d_results.pt').write_bytes(b'different longer content')
        self.assertEqual(ui.take_to_show(self.root,self.person,'initial',self.root/self.J,10),self.S(0,100))
        self.assertEqual(self.unlisted(),set())
        donor=dict(id=2,name='Ben',takes=[T(0,300)])
        self.person['gap_fill']={'100_120':COPY()};self.assertEqual(self.unlisted([self.person,donor]),set())
        self.person['gap_fill']={'100_120':COPY(offset=300)};self.assertEqual(self.unlisted([self.person,donor]),{self.J})
        self.person['gap_fill']={'100_120':{'method':'copy','mirror':False,'offset':0}};self.assertEqual(self.unlisted([self.person,donor]),{self.J})
        self.person['gap_fill']={'100_120':AUTO()};self.assertEqual(self.unlisted(),set())
    def test_bad_records_never_raise(self):
        self.assertEqual(ui.unlisted_joins(self.root,dict(id=1,takes=[]),'initial'),set())
        self.assertEqual(ui.unlisted_joins(self.root,dict(id=1,takes='bad',joins=[5,{}]),'initial'),set())

if __name__=='__main__':unittest.main()
