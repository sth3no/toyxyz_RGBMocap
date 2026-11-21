import json,os,tempfile,unittest
from pathlib import Path
from mocap import people_store as ps
S=lambda a,b:f'p1/takes/{a:09d}_{b:09d}_scene_v1'
J='p1/takes/000000000_000000200_fill_v1'

def stamp(path):
    s=path.stat();return [s.st_size,s.st_mtime_ns]

def build(root):
    root=Path(root);(root/'input.mp4').write_bytes(b'video')
    takes=[]
    for a,b in((0,100),(120,200)):
        d=root/S(a,b);d.mkdir(parents=True)
        (d/'tracking.npz').write_bytes(b'track%d'%a);(d/'hmr4d_results.pt').write_bytes(b'result%d'%a)
        takes.append(dict(project=S(a,b),first=a,end=b))
    j=root/J;j.mkdir(parents=True)
    (j/'project.json').write_text(json.dumps(dict(people_root='../../..')),encoding='utf-8')
    fill=dict(version=1,person=1,kind='refined',first=0,end=200,gaps=[dict(first=100,end=120,method='smooth',params={})],
        sources=[dict(project=t['project'],first=t['first'],end=t['end'],
            files={n:stamp(root/t['project']/n) for n in('tracking.npz','hmr4d_results.pt')}) for t in takes])
    (j/'fill.json').write_text(json.dumps(fill),encoding='utf-8')
    person=dict(id=1,color='#b5b5b5',preview='p1',takes=takes,joins=[dict(project=J,first=0,end=200,sources=[t['project'] for t in takes])])
    state=dict(version=ps.VERSION,input=ps.media_receipt(root),people=[person])
    ps.save_people(root,state)
    return root,state,person

class TakeJoinsTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root,self.state,self.person=build(self.tmp.name)
    def test_current_join_replaces_sources(self):
        self.assertTrue(ps.join_current(self.root,self.person,self.person['joins'][0]))
        got=ps.active_takes(self.root,self.person)
        self.assertEqual(got,[dict(project=J,first=0,end=200,join=True)])
        self.assertNotIn('join',self.person['joins'][0])
    def test_changed_source_file_makes_join_stale(self):
        (self.root/S(120,200)/'hmr4d_results.pt').write_bytes(b'different longer content')
        self.assertFalse(ps.join_current(self.root,self.person,self.person['joins'][0]))
        self.assertEqual(ps.active_takes(self.root,self.person),self.person['takes'])
    def test_missing_source_file_makes_join_stale(self):
        (self.root/S(0,100)/'tracking.npz').unlink()
        self.assertFalse(ps.join_current(self.root,self.person,self.person['joins'][0]))
    def test_pending_marker_makes_join_stale(self):
        (self.root/J/'fill_pending.json').write_text('{}',encoding='utf-8')
        self.assertFalse(ps.join_current(self.root,self.person,self.person['joins'][0]))
    def test_removed_source_take_makes_join_stale(self):
        self.person['takes']=self.person['takes'][:1]
        self.assertFalse(ps.join_current(self.root,self.person,self.person['joins'][0]))
    def test_unreadable_fill_json_is_stale(self):
        (self.root/J/'fill.json').write_text('{',encoding='utf-8')
        self.assertFalse(ps.join_current(self.root,self.person,self.person['joins'][0]))
        self.assertEqual(ps.active_takes(self.root,self.person),self.person['takes'])
    def test_person_without_joins_is_unchanged(self):
        p=dict(self.person);del p['joins'];p['takes']=list(reversed(p['takes']))
        self.assertEqual(ps.active_takes(self.root,p),p['takes'])
        p['joins']=[];self.assertEqual(ps.active_takes(self.root,p),p['takes'])
    def test_join_anchor(self):
        self.assertEqual(ps.join_anchor(self.state,J),S(0,100))
        self.assertEqual(ps.join_anchor(self.state,S(120,200)),S(120,200))
    def test_owned_projects(self):
        self.assertEqual(ps.owned_projects(self.person),[S(0,100),S(120,200),J])
    def test_load_people_checks_join_paths(self):
        self.assertEqual(ps.load_people(self.root)['people'][0]['joins'][0]['project'],J)
        self.person['joins'][0]['project']='../../escape'
        ps.save_people(self.root,self.state)
        with self.assertRaises(ValueError):ps.load_people(self.root)
    def test_joined_project_current(self):
        self.assertTrue(ps.joined_project_current(self.root/J))
        self.assertIsNone(ps.joined_project_current(self.root/S(0,100)))
        (self.root/S(120,200)/'tracking.npz').write_bytes(b'changed content!')
        self.assertFalse(ps.joined_project_current(self.root/J))
    def test_malformed_join_records_are_stale(self):
        j=self.person['joins'][0];del j['sources']
        self.assertFalse(ps.join_current(self.root,self.person,j))
        self.assertEqual(ps.active_takes(self.root,self.person),self.person['takes'])
        for bad in(None,'x',{'a':1},[None,5]):
            self.person['joins']=bad;self.assertEqual(ps.active_takes(self.root,self.person),self.person['takes'])
    def test_join_must_match_fill_json(self):
        j=self.person['joins'][0]
        j['sources']=j['sources'][:1];self.assertFalse(ps.join_current(self.root,self.person,j))
        j['sources']=[S(120,200),S(0,100)];self.assertFalse(ps.join_current(self.root,self.person,j))
        j['sources']=[S(0,100),S(120,200)];j['end']=199;self.assertFalse(ps.join_current(self.root,self.person,j))
    def test_load_people_rejects_non_list_joins(self):
        self.person['joins']=None;ps.save_people(self.root,self.state);ps.load_people(self.root)
        self.person['joins']='x';ps.save_people(self.root,self.state)
        with self.assertRaises(ValueError):ps.load_people(self.root)
    def test_empty_fingerprints_are_stale(self):
        f=self.root/J/'fill.json';d=json.loads(f.read_text(encoding='utf-8'));d['sources'][0]['files']={}
        f.write_text(json.dumps(d),encoding='utf-8')
        self.assertFalse(ps.join_current(self.root,self.person,self.person['joins'][0]))
    def test_overlapping_joins_first_wins(self):
        j=self.person['joins'][0];k=dict(j,project='p1/takes/other');self.person['joins']=[j,k]
        (self.root/'p1/takes/other').mkdir();(self.root/'p1/takes/other/fill.json').write_text((self.root/J/'fill.json').read_text(encoding='utf-8'),encoding='utf-8')
        self.assertEqual([t['project'] for t in ps.active_takes(self.root,self.person)],[J])
if __name__=='__main__':unittest.main()
