"""Which person of a block can continue a person of the block before, the rows, the automatic link, stale links and a
linked row's state on disk, from synthetic clip folders built as tests/test_sequence_state.py builds them."""
import json,shutil,tempfile,unittest
from pathlib import Path

from mocap import people_store as ps
from mocap import sequence_links as links
from mocap import sequence_state
from mocap.sequence import Link,Sequence
from test_parts_window import scene,write_result
import test_sequence_state   # not its StateTest by name, which would run its tests here again
from test_sequence_state import overlap
from test_take_joins import J,S,build

ANNA=((1,'Anna',range(300)),)
TWO=((1,'Anna',range(300)),(2,'Ben',range(100)))
THREE=TWO+((3,'Cleo',range(300)),)


class LinksTest(unittest.TestCase):
    """Blocks of 10 s, each with its own ExMR clip folder (frames 0-299) after Find person."""
    take=test_sequence_state.StateTest.take
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);base=self.base=Path(self.tmp.name)
        video=self.video=scene.video=base/'video.mp4';video.write_bytes(b'video')
        self.manifest=dict(version=1,frames=300,fps=30,start=0.,end=10.,width=64,height=48,camera='static',source=dict(path=str(video),fps=30.),
                           pipeline=dict(model='exmr'),exmr_options=dict(static_camera=True,smoothing='none'),stages={})
        self.sequence=Sequence()
    def add(self,root=None):
        folders={} if root is None else {'exmr':str(root),'flowhmr':str(root)}
        return self.sequence.add(self.video,10.,folders=folders)
    def clip(self,name,people,*takes):
        """A block whose clip folder holds `people` and the takes (ident, first, end, ready) as the batch leaves them."""
        self.root=self.base/name;scene(self.root,self.manifest,people)
        for take in takes:self.take(*take)
        return self.add(self.root)
    def joined(self,name):
        """A block whose person 1 has two takes joined over a smoothed hidden range (tests/test_take_joins.py build())."""
        root=self.base/name;root.mkdir();build(root);(root/'project.json').write_text(json.dumps(self.manifest),encoding='utf-8')
        return self.add(root)
    def states(self,model='exmr'):return {b.id:sequence_state.read(b,model) for b in self.sequence.blocks}
    def refusal(self,source,target,model='exmr'):return links.refusal(self.sequence,self.states(model),source,target,model)

    def test_rows_follow_links(self):
        a=self.clip('a',TWO).id;b=self.clip('b',THREE).id
        self.assertEqual(links.rows(self.sequence,self.states()),{(a,1):0,(a,2):1,(b,1):0,(b,2):1,(b,3):2})
        self.sequence.link((a,2),(b,3));rows=links.rows(self.sequence,self.states())
        self.assertEqual(rows,{(a,1):0,(a,2):1,(b,3):1,(b,1):0,(b,2):2})
        self.assertEqual(links.row_count(rows),3);self.assertEqual(links.row_count({}),0)
    def test_rows_of_two_links_into_one_block_and_a_chain_of_three(self):
        a=self.clip('a',TWO).id;b=self.clip('b',THREE).id;c=self.clip('c',TWO).id
        self.sequence.link((a,1),(b,3));self.sequence.link((a,2),(b,1));self.sequence.link((b,3),(c,2))
        rows=links.rows(self.sequence,self.states())
        self.assertEqual(rows,{(a,1):0,(a,2):1,(b,3):0,(b,1):1,(b,2):2,(c,2):0,(c,1):1})
        self.assertEqual(links.row_count(rows),3)
    def test_refusals(self):
        a=self.clip('a',ANNA,(1,0,300)).id;b=self.add().id   # the later block has no folder for the model
        self.assertEqual(self.refusal((a,1),(b,1)),'Both blocks must be tracked with the same model.')
        self.assertEqual(self.refusal((b,1),(a,1)),'Both blocks must be tracked with the same model.')
        b=self.clip('b',ANNA,(1,0,300)).id
        self.assertIsNone(self.refusal((a,1),(b,1)))
        self.assertEqual(self.refusal((a,1),(b,1),'flowhmr'),'FlowHMR results cannot be joined.')
        c=self.clip('c',ANNA,(1,0,300,False)).id
        self.assertEqual(self.refusal((c,1),(b,1)),'Anna has no finished take in the earlier block.')
        self.assertEqual(self.refusal((a,2),(b,1)),'Person 2 has no finished take in the earlier block.')   # not in the list
        self.assertEqual(self.refusal((a,1),(c,1)),"Anna's take is not finished.")
        e=self.clip('e',ANNA,(1,0,100),(1,150,300)).id   # two takes in use, both finished
        self.assertEqual([[(t.first,t.end,t.ready) for t in p.takes] for p in self.states()[e].people],[[(0,100,True),(150,300,True)]])
        self.assertEqual(self.refusal((a,1),(e,1)),'Anna has several takes in this block. Fill their hidden ranges or reconstruct the block in one piece first.')
        self.assertIsNone(self.refusal((e,1),(b,1)))   # the earlier person may have several: the last one is continued
        f=self.clip('f',ANNA,(1,0,100),(1,150,300,False)).id
        self.assertEqual(self.refusal((f,1),(b,1)),"Anna's take is not finished.")   # its last take 150-300 is the one continued
        d=self.joined('d').id
        self.assertEqual(self.refusal((a,1),(d,1)),"Person 1's take contains generated frames, which cannot be joined again.")
    def test_a_block_reconstructed_without_find_person_cannot_receive(self):   # no people list: refused as the earlier block (also by the automatic link), allowed as the later one
        a=self.clip('a',ANNA,(1,0,300)).id
        root=self.base/'single';root.mkdir();(root/'project.json').write_text(json.dumps(dict(self.manifest,pipeline=dict(model='exmr',result='initial'))),encoding='utf-8')
        write_result(root,'initial',b'single');s=self.add(root).id
        self.assertEqual([(p.id,p.name) for p in self.states()[s].people],[(1,'Person 1')]);self.assertIsNone(self.states()[s].fingerprint)
        self.assertIsNone(self.refusal((a,1),(s,1)))   # as the source of the import
        b=self.clip('b',ANNA,(1,0,300)).id
        self.assertEqual(self.refusal((s,1),(b,1)),'Run Find person in video.mp4 first. A block reconstructed without it has no person to continue.')
        self.sequence.links=[];self.sequence.blocks=[self.sequence.block(i) for i in (s,b)]
        self.assertEqual(links.automatic(self.sequence,self.states(),'exmr'),[])
    def test_overlap_joined_take_can_link(self):
        a=self.clip('a',ANNA,(1,0,300)).id;b=self.joined('b')
        root=Path(b.folders['exmr']);overlap(root)
        (root/J/'project.json').write_text(json.dumps(dict(self.manifest,frames=200,end=200/30,people_root='../../..',person_id=1,person_preview=False,
                                                          person_frames=[0,200],pipeline=dict(model='exmr',result='initial'))),encoding='utf-8')
        write_result(root/J,'initial',b'joined')
        take=self.states()[b.id].people[0].takes
        self.assertEqual([(t.first,t.end,t.ready,t.joined,t.filled) for t in take],[(0,200,True,True,False)])
        self.assertIsNone(self.refusal((a,1),(b.id,1)))
    def test_automatic_only_when_one_person_each(self):
        a=self.clip('a',ANNA,(1,0,300)).id;b=self.clip('b',ANNA,(1,0,300)).id
        self.assertEqual(links.automatic(self.sequence,self.states(),'exmr'),[((a,1),(b,1))])
        self.assertEqual(links.automatic(self.sequence,self.states(),'flowhmr'),[])   # refused
        c=self.clip('c',TWO,(1,0,300),(2,0,100)).id
        self.assertEqual(self.states()[c].kind,'tracked')
        self.assertEqual(links.automatic(self.sequence,self.states(),'exmr'),[((a,1),(b,1))])   # 1 + 2: no pair
        self.sequence.decided.add(b);self.assertEqual(links.automatic(self.sequence,self.states(),'exmr'),[])   # decided
        self.sequence.decided.clear();d=self.clip('d',ANNA,(1,0,300)).id;self.sequence.move(d,1)   # a, d, b, c
        self.assertEqual(links.automatic(self.sequence,self.states(),'exmr'),[((a,1),(d,1)),((d,1),(b,1))])
        self.sequence.link((d,1),(b,1));self.sequence.decided.discard(b)
        self.assertEqual(links.automatic(self.sequence,self.states(),'exmr'),[])   # already linked: d's person has a link
        e=self.clip('e',ANNA,(1,0,300,False)).id;self.sequence.move(e,0)   # e, a, d, b, c: e is partly tracked
        self.assertEqual(links.automatic(self.sequence,self.states(),'exmr'),[])
    def test_stale_links(self):
        a=self.clip('a',ANNA,(1,0,300)).id;b=self.clip('b',TWO,(1,0,300),(2,0,100)).id
        good=self.sequence.link((a,1),(b,1),prints=(None,'old'));missing=Link((a,1),(b,5));self.sequence.links.append(missing)
        self.assertEqual(links.stale(self.sequence,self.states()),[])   # lists without a fingerprint are not judged
        for name,mark in(('a','first'),('b','old')):
            state=ps.load_people(self.base/name);state['fingerprint']=mark;ps.save_people(self.base/name,state)
        self.assertEqual(links.stale(self.sequence,self.states()),[missing])   # person missing from the state
        self.assertEqual(links.stale(self.sequence,self.states('gvhmr')),[])   # another model: both blocks untracked, nothing judged
        self.sequence.links.remove(missing)
        state['fingerprint']='new';ps.save_people(self.base/'b',state)   # Find person again
        self.assertEqual(links.stale(self.sequence,self.states()),[good])
        good.prints=(None,None);self.assertEqual(links.stale(self.sequence,self.states()),[])   # a print of None matches anything
        good.prints=(None,'old');(self.base/'b'/'input.mp4').write_bytes(b'another video')   # the list is out of date: not judged
        self.assertEqual(self.states()[b].reason,'Person detection is out of date. Run Find person again.')
        self.assertEqual(links.stale(self.sequence,self.states()),[])
        g=self.sequence.add(self.base/'gone.mp4',10.).id;self.sequence.link((b,1),(g,1),prints=('x','y'))
        self.assertEqual(self.states()[g].kind,'missing');self.assertEqual(links.stale(self.sequence,self.states()),[])   # missing video
        gone=Link(('deleted',1),(a,1));self.sequence.links.append(gone)   # block deleted
        self.assertEqual(links.stale(self.sequence,self.states()),[gone])
    def test_status(self):
        """Block a's person 1 received block b's take as S(120,200) (the copy) and joined it over a smoothed range."""
        a=self.joined('a').id;b=self.clip('b',ANNA,(1,0,300)).id;root=self.base/'a';source=self.base/'b'/'p1/takes/000000000_000000300_scene_v1'
        link=self.sequence.link((a,1),(b,1))
        self.assertEqual(links.status(self.sequence,self.states(),(b,1),'exmr'),'linked')   # no built
        link.built=dict(copy=S(120,200),take=str(source),stamp=links.result_stamp(source),gap=30,method='smooth')
        status=lambda:links.status(self.sequence,self.states(),(b,1),'exmr')
        self.assertEqual(status(),'joined');self.assertEqual(links.status(self.sequence,self.states(),(a,1),'exmr'),'joined')   # the whole row
        self.sequence.set_gap(b,2.);self.assertEqual(status(),'out of date');self.sequence.set_gap(b,1.);self.assertEqual(status(),'joined')
        self.sequence.set_method((b,1),'auto');self.assertEqual(status(),'out of date');self.sequence.set_method((b,1),'smooth')
        (source/'hmr4d_results.pt').write_bytes(b'tracked again, longer');self.assertEqual(status(),'out of date')   # source stamp changed
        link.built['stamp']=links.result_stamp(source);self.assertEqual(status(),'joined')
        link.built['copy']=S(0,100);self.assertEqual(status(),'joined')   # also a source of the current join
        link.built['copy']='p1/takes/elsewhere';self.assertEqual(status(),'out of date')
        link.built['copy']=str((root/S(120,200)).resolve());self.assertEqual(status(),'joined')   # an absolute path names the same take
        later=self.base/'b';state=ps.load_people(later);take=state['people'][0]['takes'][0];old=take['project']
        shutil.copytree(source,later/'p1/takes/000000000_000000250_scene_v1');take.update(project='p1/takes/000000000_000000250_scene_v1',end=250)
        ps.save_people(later,state);self.assertEqual(status(),'out of date')   # tracked again over another range: the old folder is untouched
        take.update(project=old,end=300);ps.save_people(later,state);self.assertEqual(status(),'joined')
        state=ps.load_people(root);state['people'][0]['takes']=state['people'][0]['takes'][:1];ps.save_people(root,state)   # copy withdrawn
        self.assertEqual(status(),'out of date')
    def test_status_of_a_partly_built_row(self):
        a=self.joined('a').id;b=self.clip('b',ANNA,(1,0,300)).id;c=self.clip('c',ANNA,(1,0,300)).id
        self.sequence.link((a,1),(b,1));self.sequence.link((b,1),(c,1))
        self.assertEqual(links.status(self.sequence,self.states(),(c,1),'exmr'),'linked')
        source=self.base/'b'/'p1/takes/000000000_000000300_scene_v1'
        self.sequence.link_into(b,1).built=dict(copy=S(120,200),take=str(source),stamp=links.result_stamp(source),gap=30,method='smooth')
        self.assertEqual(links.status(self.sequence,self.states(),(c,1),'exmr'),'out of date')   # the link into c is not built yet
    def test_gap_frames_and_issues(self):
        a=self.add().id;b=self.add().id
        self.assertEqual(links.gap_frames(self.sequence,b),0)   # no link, no transition
        self.sequence.link((a,1),(b,1));self.assertEqual(links.gap_frames(self.sequence,b),30)
        self.sequence.set_gap(b,1/30);self.assertEqual(links.gap_frames(self.sequence,b),1)
        self.sequence.set_gap(b,9.6);self.assertEqual(links.gap_frames(self.sequence,b),288)
        self.assertIsNone(links.gap_issue('smooth',30));self.assertIsNone(links.gap_issue('auto',290))
        self.assertEqual(links.gap_issue('smooth',31),'Smooth join is limited to 30 frames; this range has 31.')
        self.assertEqual(links.gap_issue('auto',291),'Automatic fill is limited to 290 frames; this range has 291.')


if __name__=='__main__':unittest.main()
