import copy,json,sqlite3,tempfile,unittest
from contextlib import closing
from pathlib import Path
from unittest import mock
import numpy as np
from mocap import people_store as ps, person_merge as m

DB='people/gen/observations.sqlite'
PREVIEW=lambda obs:f'people/gen/person_{obs:04d}'
def take(obs,a,b):return f'{PREVIEW(obs)}/takes/{a:09d}_{b:09d}_scene_v1'
def box(obs,f):return (float(f),float(obs),f+10.,obs+10.)

def add_person(root,ident,name,obs,frames,span,db=True):
    root=Path(root);frames=list(frames)
    with closing(sqlite3.connect(root/DB)) as c,c:
        c.execute('CREATE TABLE IF NOT EXISTS observations(person INTEGER,frame INTEGER,x1 REAL,y1 REAL,x2 REAL,y2 REAL,score REAL,PRIMARY KEY(person,frame))')
        c.executemany('INSERT INTO observations VALUES(?,?,?,?,?,?,1.0)',[(obs,f,*box(obs,f)) for f in frames])
    boxes=np.full((300,4),np.nan,np.float32);observed=np.zeros(300,bool)
    for f in frames:boxes[f]=box(obs,f);observed[f]=True
    (root/PREVIEW(obs)).mkdir(parents=True,exist_ok=True)
    np.savez_compressed(root/PREVIEW(obs)/'tracking.npz',boxes=boxes,observed=observed,K=np.eye(3))
    takes=[]
    for a,b in span:
        d=root/take(obs,a,b);d.mkdir(parents=True)
        (d/'project.json').write_text(json.dumps(dict(people_root='../../../../..',person_id=ident,person_frames=[a,b])),encoding='utf-8')
        (d/'hmr4d_results.pt').write_bytes(b'motion%d_%d_%d'%(obs,a,b));takes.append(dict(project=take(obs,a,b),first=a,end=b))
    state=ps.load_people(root) if (root/'people.json').exists() else dict(version=ps.VERSION,input=ps.media_receipt(root),observations=DB,active=1,people=[])
    state['people'].append(dict(id=ident,name=name,color='#b5b5b5',checked=True,first=min(frames),last=max(frames)+1,observations=len(frames),
        preview=PREVIEW(obs),takes=takes,observation_id=obs))
    ps.save_people(root,state)
    return state['people'][-1]

def build(tmp):
    root=Path(tmp);(root/'input.mp4').write_bytes(b'video');(root/DB).parent.mkdir(parents=True)
    add_person(root,1,'Anna',1,[*range(0,120),*range(166,300)],[(0,120),(166,300)])
    add_person(root,2,'Ben',2,range(0,300),[(0,300)])
    add_person(root,3,'Cover',7,range(118,136),[(121,136)])
    state=ps.load_people(root);state['people'][0]['gap_fill']={'120_166':{'method':'smooth'}};ps.save_people(root,state)
    return root

def rows(root,obs,a=0,b=300):
    with closing(sqlite3.connect(root/DB)) as c:return [r[0] for r in c.execute('SELECT frame FROM observations WHERE person=? AND frame>=? AND frame<? ORDER BY frame',(obs,a,b))]

def tracking(root,preview):
    with np.load(root/preview/'tracking.npz') as z:return z['boxes'],z['observed'],z['K']

def snapshot(root):
    with closing(sqlite3.connect(root/DB)) as c:db=sorted(c.execute('SELECT * FROM observations'))
    projects={str(p.relative_to(root)):json.loads(p.read_text(encoding='utf-8')) for p in sorted(root.glob('people/gen/*/takes/*/project.json'))}
    previews={}
    for p in sorted(root.glob('people/gen/*/tracking.npz')):
        with np.load(p) as z:previews[str(p.relative_to(root))]=(z['boxes'].copy(),z['observed'].copy(),z['K'].copy())
    return json.loads((root/'people.json').read_text(encoding='utf-8')),db,projects,previews

def assert_same(t,a,b):
    t.assertEqual(a[0],b[0]);t.assertEqual(a[1],b[1]);t.assertEqual(a[2],b[2]);t.assertEqual(a[3].keys(),b[3].keys())
    for k in a[3]:
        for x,y in zip(a[3][k],b[3][k]):t.assertTrue(np.array_equal(x,y,equal_nan=True),k)

def motion(root):
    return {str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(root.glob('people/gen/*/takes/*/hmr4d_results.pt'))}

class PersonMergeTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=build(self.tmp.name)
    def test_continue_as_joins_the_people(self):
        root=self.root;state=m.continue_as(root,3,1,'gvhmr',30)
        self.assertEqual([p['id'] for p in state['people']],[1,2]);self.assertEqual(state['active'],1)
        anna=state['people'][0]
        self.assertEqual([(t['first'],t['end']) for t in anna['takes']],[(0,120),(166,300),(121,136)])
        self.assertEqual((anna['first'],anna['last'],anna['observations']),(0,300,270))
        self.assertEqual(anna['gap_fill'],{'120_166':{'method':'smooth'}})
        self.assertEqual(m.absorbed(anna),[(3,'Cover')]);self.assertEqual(state,ps.load_people(root))
        self.assertEqual(rows(root,1,120,136),list(range(120,136)))
        self.assertEqual(json.loads((root/take(7,121,136)/'project.json').read_text())['person_id'],1)
        boxes,observed,K=tracking(root,PREVIEW(1));self.assertTrue(observed[120:136].all());self.assertFalse(observed[136:166].any())
        self.assertFalse((root/m.PENDING).exists())
    def test_shared_frames_stay_with_the_source(self):
        m.continue_as(self.root,3,1,'gvhmr',30)
        self.assertEqual(rows(self.root,7),[118,119])
        self.assertEqual(ps.load_people(self.root)['people'][0]['absorbed'][0]['frames'],[[120,136]])
    def test_separate_restores_everything(self):
        root=self.root;before=snapshot(root);m.continue_as(root,3,1,'gvhmr',30);state=m.separate(root,1,3)
        after=snapshot(root);before[0]['active']=after[0]['active']
        assert_same(self,before,after);self.assertNotIn('absorbed',state['people'][0])
    def test_separate_keeps_takes_made_after_the_merge(self):
        root=self.root;m.continue_as(root,3,1,'gvhmr',30)
        state=ps.load_people(root);state['people'][0]['takes'].append(dict(project=take(1,0,136),first=0,end=136));ps.save_people(root,state)
        anna=m.separate(root,1,3)['people'][0]
        self.assertEqual([(t['first'],t['end']) for t in anna['takes']],[(0,120),(166,300),(0,136)])
    def test_two_merges_separated_in_any_order(self):
        root=self.root
        for order in((3,4),(4,3)):
            with self.subTest(order=order):
                with tempfile.TemporaryDirectory() as tmp:
                    root=build(tmp);add_person(root,4,'Late',9,range(140,151),[(140,151)]);before=snapshot(root)
                    m.continue_as(root,3,1,'gvhmr',30);m.continue_as(root,4,1,'gvhmr',30)
                    self.assertEqual(m.absorbed(ps.load_people(root)['people'][0]),[(4,'Late'),(3,'Cover')])
                    for ident in order:m.separate(root,1,ident)
                    after=snapshot(root);before[0]['active']=after[0]['active'];assert_same(self,before,after)
    def test_separate_restores_the_original_counts_in_any_order(self):
        for drop in(False,True):
            for order in((3,4),(4,3)):
                with self.subTest(drop=drop,order=order),tempfile.TemporaryDirectory() as tmp:
                    root=build(tmp);add_person(root,4,'Late',9,range(140,151),[(140,151)])
                    state=ps.load_people(root);anna=state['people'][0];anna.update(first=1,last=999,observations=7)
                    if drop:del anna['observations']
                    ps.save_people(root,state)
                    m.continue_as(root,3,1,'gvhmr',30);m.continue_as(root,4,1,'gvhmr',30)
                    for ident in order:got=m.separate(root,1,ident)['people'][0]
                    self.assertEqual((got['first'],got['last']),(1,999))
                    if drop:self.assertNotIn('observations',got)
                    else:self.assertEqual(got['observations'],7)
    def test_a_joined_person_keeps_its_own_absorbed_people(self):
        root=self.root;add_person(root,4,'Late',9,range(140,151),[(140,151)]);before=snapshot(root)
        m.continue_as(root,4,3,'gvhmr',30);state=m.continue_as(root,3,1,'gvhmr',30);anna=state['people'][0]
        self.assertEqual([(t['first'],t['end']) for t in anna['takes']],[(0,120),(166,300),(121,136),(140,151)])
        self.assertEqual(rows(root,1,120,136)+rows(root,1,140,151),[*range(120,136),*range(140,151)])
        cover=[p for p in m.separate(root,1,3)['people'] if p['id']==3][0]
        self.assertEqual(m.absorbed(cover),[(4,'Late')]);m.separate(root,3,4)
        after=snapshot(root);before[0]['active']=after[0]['active'];assert_same(self,before,after)
    def test_refusals_change_nothing(self):
        root=self.root;before=snapshot(root)
        for args,text in (((3,1,'flowhmr',30),'Continue as is available for GVHMR and ExMR in this version.'),
                          ((1,1,'gvhmr',30),'Choose another person of this scene.'),((9,1,'gvhmr',30),'Choose another person of this scene.'),
                          ((2,1,'gvhmr',30),'Ben and Anna are both visible in 254 frames, so they cannot be the same person.')):
            with self.assertRaises(m.Refused) as c:m.continue_as(root,*args)
            self.assertEqual(str(c.exception),text)
        with self.assertRaises(m.Refused) as c:m.separate(root,1,2)
        self.assertEqual(str(c.exception),'Choose a person that was joined into this one.')
        assert_same(self,before,snapshot(root));self.assertFalse((root/m.PENDING).exists())
    def test_separate_refuses_when_the_id_is_in_the_list(self):
        root=self.root;m.continue_as(root,3,1,'gvhmr',30)
        state=ps.load_people(root);state['people'].append(dict(id=3,color='#b5b5b5',preview=PREVIEW(7),takes=[]));ps.save_people(root,state)
        with self.assertRaises(m.Refused):m.separate(root,1,3)
    def test_missing_database_is_refused(self):
        (self.root/DB).unlink()
        with self.assertRaises(m.Refused) as c:m.continue_as(self.root,3,1,'gvhmr',30)
        self.assertEqual(str(c.exception),'The person detection data of this scene is missing. Run Find person again.')
        state=ps.load_people(self.root);del state['observations'];ps.save_people(self.root,state)
        with self.assertRaises(m.Refused):m.continue_as(self.root,3,1,'gvhmr',30)
    def test_separate_without_database_is_refused(self):
        root=self.root;m.continue_as(root,3,1,'gvhmr',30);(root/DB).unlink();before=(root/'people.json').read_bytes()
        with self.assertRaises(m.Refused) as c:m.separate(root,1,3)
        self.assertEqual(str(c.exception),'The person detection data of this scene is missing. Run Find person again.')
        self.assertFalse((root/DB).exists());self.assertEqual((root/'people.json').read_bytes(),before);self.assertFalse((root/m.PENDING).exists())
    def test_recover_without_database_is_refused_and_creates_nothing(self):
        root=self.root;m.continue_as(root,3,1,'gvhmr',30)
        with mock.patch.object(m,'_relabel',side_effect=OSError('stopped')),self.assertRaises(OSError):m.separate(root,1,3)
        (root/DB).unlink();pending=(root/m.PENDING).read_bytes();people=(root/'people.json').read_bytes()
        for call in(lambda:m.recover(root),lambda:m.continue_as(root,3,1,'gvhmr',30)):
            with self.assertRaises(m.Refused) as c:call()
            self.assertEqual(str(c.exception),'The person detection data of this scene is missing. Run Find person again.')
        self.assertFalse((root/DB).exists());self.assertEqual((root/m.PENDING).read_bytes(),pending);self.assertEqual((root/'people.json').read_bytes(),people)
    def test_malformed_records_and_pending_shapes_are_refused(self):
        root=self.root;m.continue_as(root,3,1,'gvhmr',30)
        state=ps.load_people(root);state['people'][0]['absorbed'].insert(0,'bad');state['people'][0]['absorbed'].insert(0,dict(person=5));ps.save_people(root,state)
        with self.assertRaises(m.Refused):m.separate(root,1,9)
        self.assertEqual(m.separate(root,1,3)['people'][2]['id'],3)
        pending=dict(version=1,state=dict(observations=DB,people=[]),move={'from':1,'to':7,'frames':[[120,136]]},person_ids={},tracking=dict(preview=PREVIEW(1)))
        (root/m.PENDING).write_text(json.dumps(pending),encoding='utf-8')
        with self.assertRaises(m.Refused) as c:m.recover(root)
        self.assertEqual(str(c.exception),'A previous Continue as did not finish and cannot be read. Run Find person again.')
    def test_interrupted_operation_is_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref=build(tmp);m.continue_as(ref,3,1,'gvhmr',30);want=snapshot(ref)
            real=m.atomic_json
            def stop_at_people(path,value):
                if Path(path).name=='people.json':raise OSError('stopped')
                real(path,value)
            cases=(('before the database',mock.patch.object(m,'_relabel',side_effect=OSError('stopped'))),
                   ('after the database',mock.patch.object(m,'_write_person_ids',side_effect=OSError('stopped'))),
                   ('after the tracking',mock.patch.object(m,'atomic_json',side_effect=stop_at_people)))
            for name,patch in cases:
                with self.subTest(name),tempfile.TemporaryDirectory() as tmp2:
                    root=build(tmp2)
                    with patch,self.assertRaises(OSError):m.continue_as(root,3,1,'gvhmr',30)
                    self.assertTrue((root/m.PENDING).is_file())
                    self.assertTrue(m.recover(root));self.assertFalse((root/m.PENDING).exists());self.assertFalse(m.recover(root))
                    got=snapshot(root);want[0]['input']=got[0]['input'];self.assertEqual(got[0],want[0]);assert_same(self,want,got)
    def test_interrupted_separate_is_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=build(tmp);before=snapshot(root);m.continue_as(root,3,1,'gvhmr',30)
            with mock.patch.object(m,'_write_tracking',side_effect=OSError('stopped')),self.assertRaises(OSError):m.separate(root,1,3)
            self.assertTrue(m.recover(root));after=snapshot(root);before[0]['active']=after[0]['active'];assert_same(self,before,after)
    def test_recover_without_pending_file(self):self.assertFalse(m.recover(self.root))
    def test_unreadable_pending_file_is_refused_and_kept(self):
        before=snapshot(self.root)
        for text in('{',json.dumps(dict(version=2,state={}))):
            (self.root/m.PENDING).write_text(text,encoding='utf-8')
            for call in(lambda:m.recover(self.root),lambda:m.continue_as(self.root,3,1,'gvhmr',30),lambda:m.separate(self.root,1,3)):
                with self.assertRaises(m.Refused) as c:call()
                self.assertEqual(str(c.exception),'A previous Continue as did not finish and cannot be read. Run Find person again.')
            self.assertEqual((self.root/m.PENDING).read_text(encoding='utf-8'),text)
        (self.root/m.PENDING).unlink();assert_same(self,before,snapshot(self.root))
    def test_candidates_and_absorbed(self):
        state=ps.load_people(self.root)
        self.assertEqual(m.candidates(state,3),[(1,'Anna'),(2,'Ben')])
        state['people'][1]['name']='';self.assertEqual(m.candidates(state,3),[(1,'Anna'),(2,'Person 2')])
        self.assertEqual(m.absorbed(dict(id=1)),[]);self.assertEqual(m.absorbed(dict(id=1,absorbed='bad')),[])
        self.assertEqual(m.absorbed(dict(id=1,absorbed=[dict(person={})])),[])
        self.assertEqual(m.absorbed(dict(id=1,absorbed=[dict(person=dict(id=5))])),[(5,'Person 5')])
    def test_motion_files_are_never_touched(self):
        before=motion(self.root);m.continue_as(self.root,3,1,'gvhmr',30);self.assertEqual(motion(self.root),before)
        m.separate(self.root,1,3);self.assertEqual(motion(self.root),before)

if __name__=='__main__':unittest.main()
