"""What a sequence block looks like on disk, from synthetic clip folders built as tests/test_parts_window.py builds them
(the files the app's own readiness checks read)."""
import json,shutil,tempfile,unittest
from pathlib import Path

from mocap import people_store as ps
from mocap import sequence_state
from mocap.sequence import Block
from test_parts_window import IDENTITY,scene,stamp,write_result
from test_take_joins import J,build

OUT_OF_DATE='Person detection is out of date. Run Find person again.'


class StateTest(unittest.TestCase):
    """A 10 s ExMR clip folder (frames 0-299 of its video) after Find person with Anna and Ben."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);base=self.base=Path(self.tmp.name)
        video=self.video=scene.video=base/'video.mp4';video.write_bytes(b'video')
        self.manifest=dict(version=1,frames=300,fps=30,start=0.,end=10.,width=64,height=48,camera='static',source=dict(path=str(video),fps=30.),
                           pipeline=dict(model='exmr'),exmr_options=dict(static_camera=True,smoothing='none'),stages={})
        self.root=base/'scene'
    def block(self,folder=None,video=None):
        return Block('b1',str(video or self.video),0.,10.,10.,{} if folder is None else {'exmr':str(folder)})
    def people(self):scene(self.root,self.manifest,((1,'Anna',range(300)),(2,'Ben',range(100))))
    def take(self,ident,a,b,ready=True):
        """A take folder as the person job and the batch leave it (tests/test_parts_window.py RunnerTest.take), listed in people.json."""
        relative=f'p{ident}/takes/{a:09d}_{b:09d}_scene_v1';d=self.root/relative;d.mkdir(parents=True,exist_ok=True)
        shutil.copy(self.root/'input.mp4',d/'input.mp4')
        (d/'project.json').write_text(json.dumps(dict(self.manifest,frames=b-a,start=a/30,end=b/30,people_root='../../..',person_id=ident,person_preview=False,
                                                    person_frames=[a,b],pipeline=dict(model='exmr',result='initial'),correction={})),encoding='utf-8')
        if ready:
            write_result(d,'initial',b'result %d %d %d'%(ident,a,b))
            (d/'scene_camera.json').write_text(json.dumps(dict(sources={'hmr4d_results.pt':stamp(d/'hmr4d_results.pt')},model=[1,2],world_to_camera=IDENTITY)),encoding='utf-8')
        state=ps.load_people(self.root);person=next(p for p in state['people'] if p['id']==ident)
        person['takes']=[t for t in person['takes'] if (t['first'],t['end'])!=(a,b)]+[dict(project=relative,first=a,end=b)];ps.save_people(self.root,state)
        return d
    def read(self,folder=None):return sequence_state.read(self.block(self.root if folder is None else folder),'exmr')

    def test_no_folder(self):
        state=sequence_state.read(self.block(),'exmr')
        self.assertEqual((state.kind,state.reason,state.label,state.people),('untracked','','not tracked',[]))
        gone=self.base/'gone.mp4';state=sequence_state.read(self.block(video=gone),'exmr')
        self.assertEqual((state.kind,state.reason,state.label),('missing',f'The video file is missing: {gone}','video missing'))
        self.people();self.assertEqual(sequence_state.read(self.block(self.root),'gvhmr').kind,'untracked')   # no folder for that model
    def test_folder_gone_or_other_range(self):
        state=self.read(self.base/'deleted')
        self.assertEqual((state.kind,state.reason,state.people),('untracked',f'The saved results are missing: {self.base/"deleted"}',[]))
        self.people();(self.root/'project.json').write_text(json.dumps(dict(self.manifest,start=1.,end=11.)),encoding='utf-8')
        state=self.read()
        self.assertEqual((state.kind,state.reason,state.label,state.people),('untracked','The saved results belong to another range of the video.','not tracked',[]))
        (self.root/'project.json').write_text(json.dumps(dict(self.manifest,start=5e-7)),encoding='utf-8')
        self.assertEqual(self.read().kind,'found')   # within 1e-6 s is the same range
    def test_people_states(self):
        self.people();state=self.read()
        self.assertEqual((state.kind,state.reason,state.label,state.frames,state.fps),('found','','2 people',300,30.))
        self.assertEqual([(p.id,p.name,p.color,p.checked,p.takes) for p in state.people],[(1,'Anna','#b5b5b5',True,[]),(2,'Ben','#3366aa',True,[])])
        self.take(1,0,150,ready=False);self.take(2,0,100,ready=False)
        self.assertEqual((self.read().kind,self.read().label),('found','2 people'))   # takes, none ready
        self.take(1,0,150);state=self.read()
        self.assertEqual((state.kind,state.label),('partly','partly tracked'))
        self.assertEqual([[(t.first,t.end,t.ready,t.joined) for t in p.takes] for p in state.people],[[(0,150,True,False)],[(0,100,False,False)]])
        self.take(2,0,100);self.assertEqual((self.read().kind,self.read().label),('tracked','tracked'))   # every take of every checked person is ready
        self.take(1,150,300,ready=False);self.assertEqual(self.read().kind,'partly')
    def test_unchecked_people_do_not_count(self):
        self.people();self.take(1,0,300)
        state=ps.load_people(self.root);state['people'][1]['checked']=False;ps.save_people(self.root,state)
        state=self.read()
        self.assertEqual((state.kind,state.label),('tracked','tracked'))
        self.assertEqual([(p.name,p.checked,len(p.takes)) for p in state.people],[('Anna',True,1),('Ben',False,0)])
    def test_joined_take_replaces_its_sources(self):
        """tests/test_take_joins.py build(): two takes 0-100 and 120-200 joined into 0-200 by a current fill.json."""
        self.root.mkdir();root=build(self.root)[0];(root/'project.json').write_text(json.dumps(self.manifest),encoding='utf-8')
        state=self.read()
        self.assertEqual([(p.id,p.name,p.checked) for p in state.people],[(1,'Person 1',False)])   # a row without a name or a check
        self.assertEqual([(t.first,t.end,t.ready,t.joined) for t in state.people[0].takes],[(0,200,False,True)])
        (root/J/'project.json').write_text(json.dumps(dict(self.manifest,frames=200,end=200/30,people_root='../../..',person_id=1,person_preview=False,
                                                         person_frames=[0,200],pipeline=dict(model='exmr',result='initial'))),encoding='utf-8')
        write_result(root/J,'initial',b'joined')
        people=ps.load_people(root);people['people'][0]['checked']=True;ps.save_people(root,people)
        state=self.read()
        self.assertEqual([(t.first,t.end,t.ready,t.joined) for t in state.people[0].takes],[(0,200,True,True)])
        self.assertEqual(state.kind,'tracked')   # the joined take stands in for its two unfinished sources
    def test_stale_people_list_is_reported(self):
        self.people();(self.root/'input.mp4').write_bytes(b'another video')
        state=self.read()
        self.assertEqual((state.kind,state.reason,state.people,state.label),('found',OUT_OF_DATE,[],'0 people'))
    def test_single_person_without_people_list(self):
        self.root.mkdir();(self.root/'input.mp4').write_bytes(b'video')
        (self.root/'project.json').write_text(json.dumps(dict(self.manifest,pipeline=dict(model='exmr',result='initial'))),encoding='utf-8')
        state=self.read()
        self.assertEqual((state.kind,state.reason,state.people,state.label),('untracked','',[],'not tracked'))   # no result yet
        write_result(self.root,'initial',b'result')
        state=self.read()
        self.assertEqual((state.kind,state.reason,state.label,state.frames),('tracked','','tracked',300))
        self.assertEqual([(p.name,p.checked,[(t.first,t.end,t.ready,t.joined) for t in p.takes]) for p in state.people],[('Person',True,[(0,300,True,False)])])
        self.video.unlink();self.assertEqual(self.read().kind,'tracked')   # a missing video does not matter when the folder is good
    def test_unreadable_manifest_never_raises(self):
        self.people();(self.root/'project.json').write_bytes(b'{')
        state=self.read()
        self.assertEqual((state.kind,state.reason),('untracked',f'The saved results are missing: {self.root}'))
        (self.root/'project.json').write_text(json.dumps(dict(self.manifest,start='0')),encoding='utf-8')
        self.assertEqual(self.read().reason,f'The saved results are missing: {self.root}')
        (self.root/'project.json').write_text(json.dumps(self.manifest),encoding='utf-8');(self.root/'people.json').write_bytes(b'[1]')
        self.assertEqual((self.read().kind,self.read().reason),('found','Invalid person list'))
    def test_kind_for(self):
        self.assertEqual([sequence_state.kind_for(m) for m in ('gvhmr','exmr','flowhmr')],['handflow_footmr_refined','initial','initial'])


if __name__=='__main__':unittest.main()
