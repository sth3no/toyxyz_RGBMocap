"""Parts of one video that share frames: alignment over the shared frames, a cross-fade, and the overlap record beside a gap."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R
from mocap.gap_fill import rigid_params
from mocap.take_join import assemble
from test_take_join import P0,PB,WA,WB,WC,rigid,source,tpair

T=lambda a,b:dict(project=f'p/takes/{a:09d}_{b:09d}_scene_v1',first=a,end=b)
def overlap(a,b):return dict(first=b['first'],end=a['end'],method='overlap')
def gap(a,b):return dict(first=a['end'],end=b['first'],method='smooth')
def rot(v):return R.from_rotvec(np.asarray(v).reshape(-1,3)).as_matrix()
def truth(out,frames):
    """The joined world motion the shared camera says: camera rows seen through A's world."""
    c=out['incam'];return rigid_params({k:c[k][frames] for k in('global_orient','transl')},np.linalg.inv(WA),P0,P0)


class WeightsTest(unittest.TestCase):
    def test_smoothstep_from_one_part_to_the_other(self):
        from mocap.overlap_join import weights
        w=weights(9);self.assertEqual(len(w),9)
        self.assertTrue((np.diff(w)>0).all());self.assertAlmostEqual(w[4],.5);np.testing.assert_allclose(w+w[::-1],1.)
        self.assertTrue(0<w[0]<.05 and .95<w[-1]<1)


class AlignTest(unittest.TestCase):
    def test_one_frame_is_the_seam_transform(self):
        from mocap.overlap_join import align
        from mocap.take_join import upright
        full=rigid(R.from_euler('yx',[30,4],degrees=True).as_rotvec(),(.4,.07,-.3));p=np.array([.2,1.,.5])
        np.testing.assert_allclose(align([full],[p]),upright(full,p),atol=1e-12)
    def test_yaw_is_averaged_across_the_wrap(self):
        from mocap.overlap_join import align
        fulls=[rigid((0.,np.radians(a),0.)) for a in(179.,-179.,178.,-178.)];pivots=[np.zeros(3)]*4
        T=align(fulls,pivots)
        self.assertAlmostEqual(abs(np.degrees(R.from_matrix(T[:3,:3]).as_rotvec()[1])),180.,places=9)
    def test_offset_is_the_least_squares_fit_of_the_pivots(self):
        from mocap.overlap_join import align
        full=rigid((0.,.5,0.),(1.,.3,-2.));pivots=[np.array([x,.9,z]) for x,z in((0,0),(1,0),(0,1))]
        noisy=[full@rigid(t=(d,0.,-d)) for d in(-.01,0.,.01)]
        T=align(noisy,pivots);np.testing.assert_allclose(T[:3,:3],full[:3,:3],atol=1e-12)
        moved=[f[:3,:3]@p+f[:3,3] for f,p in zip(noisy,pivots)]
        np.testing.assert_allclose((T[:3,:3]@np.mean(pivots,0)+T[:3,3])[[0,2]],np.mean(moved,0)[[0,2]],atol=1e-12)
        self.assertEqual(T[1,3],0.)


class AssembleTest(unittest.TestCase):
    """Parts cut from one known motion seen by one camera; every part lives in its own world."""
    def two(self,**second):
        a=source(0,110);b=source(100,200,**{'W':WB,'pelvis':PB,**second});return a,b,assemble([a,b],[overlap(a,b)])
    def test_rows_before_the_overlap_are_the_first_part(self):
        a,_,out=self.two()
        for k,v in a['params']['initial'].items():
            if k!='betas':self.assertTrue(np.array_equal(out['params']['initial'][k][:100],v[:100]),k)
        for k in('global_orient','body_pose','transl'):self.assertTrue(np.array_equal(out['incam'][k][:100],a['incam'][k][:100]),k)
        self.assertTrue(np.array_equal(out['logits'][:100],a['logits'][:100]));self.assertTrue(np.array_equal(out['boxes'][:100],a['boxes'][:100]))
    def test_the_second_part_is_moved_into_the_first_world(self):
        _,b,out=self.two();g,c=out['params']['initial'],out['incam']
        t=truth(out,slice(110,200))
        np.testing.assert_allclose(g['transl'][110:],t['transl'],atol=1e-9);np.testing.assert_allclose(rot(g['global_orient'][110:]),rot(t['global_orient']),atol=1e-9)
        self.assertTrue(np.array_equal(g['body_pose'][110:],b['params']['initial']['body_pose'][10:]))   # rigid move only
        np.testing.assert_allclose(c['transl'][110:],b['incam']['transl'][10:]+PB-P0,atol=1e-12)   # the shape swap keeps the camera pelvis
    def test_the_whole_join_is_the_known_motion(self):
        _,_,out=self.two();t=truth(out,slice(0,200));g=out['params']['initial']
        np.testing.assert_allclose(g['transl'],t['transl'],atol=1e-9);np.testing.assert_allclose(rot(g['global_orient']),rot(t['global_orient']),atol=1e-9)
    def test_the_overlap_fades_from_one_estimate_to_the_other(self):
        from mocap.overlap_join import weights
        a=source(0,110);b=source(100,200,W=WB,pelvis=PB);b['params']['initial']['body_pose']=b['params']['initial']['body_pose']+.02
        b['logits']=b['logits']+1.;out=assemble([a,b],[overlap(a,b)]);w=weights(10)
        pose=out['params']['initial']['body_pose'][100:110];A,B=a['params']['initial']['body_pose'][100:],b['params']['initial']['body_pose'][:10]
        self.assertLess(np.abs(pose-(A+(B-A)*w[:,None])).max(),1e-4)   # per-joint slerp of near rotations is near linear
        self.assertTrue(np.array_equal(out['params']['initial']['body_pose'][110:],b['params']['initial']['body_pose'][10:]))
        np.testing.assert_allclose(out['logits'][100:110],a['logits'][100:]+w[:,None],atol=1e-12)
        t=truth(out,slice(100,110));np.testing.assert_allclose(out['params']['initial']['transl'][100:110],t['transl'],atol=1e-9)
    def test_blended_frames_count_as_captured(self):
        a=source(0,110);a['observed'][100:105]=False;b=source(100,200,W=WB);b['observed'][5:10]=False
        out=assemble([a,b],[overlap(a,b)])
        self.assertEqual(out['mask'].shape,(200,));self.assertFalse(out['mask'].any())
        self.assertTrue(out['observed'].all())   # each blended frame was seen in one part or the other
        self.assertEqual(out['residuals'],[None]);self.assertEqual(out['reports'],[None])
    def test_yaw_near_half_a_turn(self):
        half=WA@rigid((0.,np.pi,0.),(.7,0.,-1.2))
        a=source(0,110);b=source(100,200,W=half,pelvis=PB)
        # Each frame of the overlap disagrees by half a degree, alternately: the estimates straddle +-180 degrees.
        for j in range(10):
            turn=rigid((0.,np.radians(.5 if j%2 else -.5),0.));p=b['params']['initial']
            row=rigid_params({k:p[k][j:j+1] for k in('global_orient','transl')},turn,PB,PB)
            p['global_orient'][j],p['transl'][j]=row['global_orient'][0],row['transl'][0]
        out=assemble([a,b],[overlap(a,b)]);t=truth(out,slice(110,200))
        np.testing.assert_allclose(out['params']['initial']['transl'][110:],t['transl'],atol=2e-2)
        self.assertLess(np.abs(rot(out['params']['initial']['global_orient'][110:])-rot(t['global_orient'])).max(),2e-2)
    def test_three_parts(self):
        a,b,c=source(0,110),source(100,210,W=WB,pelvis=PB),source(200,300,W=WC)
        out=assemble([a,b,c],[overlap(a,b),overlap(b,c)]);g=out['params']['initial'];t=truth(out,slice(0,300))
        self.assertEqual(len(out['mask']),300);self.assertFalse(out['mask'].any())
        np.testing.assert_allclose(g['transl'],t['transl'],atol=1e-9);np.testing.assert_allclose(rot(g['global_orient']),rot(t['global_orient']),atol=1e-9)
    def test_an_overlap_and_a_gap_in_one_chain(self):
        a,b,c=source(0,110),source(100,200,W=WB,pelvis=PB),source(215,300,W=WC)
        out=assemble([a,b,c],[overlap(a,b),gap(b,c)]);g=out['params']['initial']
        self.assertEqual(list(np.flatnonzero(out['mask'])),list(range(200,215)));self.assertTrue((out['mask'][200:215]==1).all())
        frames=np.r_[0:200,215:300];t=truth(out,frames)
        np.testing.assert_allclose(g['transl'][frames],t['transl'],atol=1e-9)
        self.assertFalse(out['observed'][200:215].any())
    def test_drift_is_absorbed_by_the_overlap(self):
        a=source(0,110,drift=(.06,0.,-.04));b=source(100,200,W=WB,pelvis=PB)
        out=assemble([a,b],[overlap(a,b)]);x=out['params']['initial']['transl']
        steps=np.linalg.norm(np.diff(x[:,[0,2]],axis=0),axis=1)
        self.assertLess(steps[95:115].max(),.08)   # a fade, not a jump of the whole drift in one frame
    def test_rejects_an_overlap_that_does_not_match_its_parts(self):
        a,b=source(0,110),source(100,200,W=WB)
        for bad in(dict(first=100,end=111,method='overlap'),dict(first=101,end=110,method='overlap')):
            with self.assertRaises(ValueError):assemble([a,b],[bad])
        c=source(105,250,W=WC)   # b's frames would all be blended
        b2=source(100,200,W=WB)
        with self.assertRaises(ValueError):assemble([a,b2,c],[overlap(a,b2),dict(first=105,end=200,method='overlap')])


class SeamsTest(unittest.TestCase):
    def test_gaps_are_unchanged_and_overlaps_are_listed_beside_them(self):
        from mocap import hidden_ranges as h
        takes=[T(0,110),T(100,200),T(215,300)]
        self.assertEqual(h.gaps(takes),[r for r in h.seams(takes) if 'kind' not in r])
        o,g=h.seams(takes)
        self.assertEqual((o['kind'],o['first'],o['end'],o['before'],o['after'],o['joinable']),('overlap',100,110,takes[0],takes[1],True))
        self.assertEqual((g['first'],g['end']),(200,215))
    def test_touching_takes_are_a_touch_record(self):
        from mocap import hidden_ranges as h
        r,=h.seams([T(0,100),T(100,200)]);self.assertEqual((r['kind'],r['first'],r['end']),('touch',100,100))
    def test_nested_takes_are_no_seam(self):
        from mocap import hidden_ranges as h
        self.assertEqual(h.seams([T(0,200),T(50,100)]),[]);self.assertEqual(h.seams([T(0,200),T(0,100)]),[])
    def test_an_overlap_that_blends_all_of_a_take_is_not_joinable(self):
        from mocap import hidden_ranges as h
        a,b,c=T(0,110),T(100,210),T(105,300)
        x,y=h.seams([a,b,c]);self.assertTrue(x['joinable']);self.assertFalse(y['joinable'])
    def test_chains_join_marked_parts_without_a_method(self):
        from mocap import hidden_ranges as h
        a,b,c=T(0,110),T(100,200),T(215,300);parts={'parts_100_110':{'method':'parts'}}
        self.assertEqual(h.chains([a,b,c],parts),[[a,b]])
        self.assertEqual(h.chains([a,b,c],{**parts,'200_215':{'method':'smooth'}}),[[a,b,c]])
        self.assertEqual(h.chains([a,b,c],{'200_215':{'method':'smooth'}}),[[b,c]])   # takes of other trims that share frames: as before
        self.assertEqual(h.chains([T(0,100),T(100,200)],{'parts_100_100':{'method':'parts'}}),[])   # touching takes are refused
        self.assertEqual(h.chains([T(0,110),T(100,210),T(105,300)],{'parts_100_110':{'method':'parts'},'parts_105_210':{'method':'parts'}}),[[T(0,110),T(100,210)]])
    def test_listed_rows_are_gaps_and_marked_seams(self):
        from mocap import hidden_ranges as h
        takes=[T(0,110),T(100,200),T(200,300),T(315,400)]
        self.assertEqual([(r.get('kind'),r['first']) for r in h.listed(takes,{})],[(None,300)])
        self.assertEqual([(r.get('kind'),r['first']) for r in h.listed(takes,{'parts_100_110':{'method':'parts'},'parts_200_200':{'method':'parts'}})],
                         [('overlap',100),('touch',200),(None,300)])
        self.assertEqual(h.seam_key(dict(first=100,end=110)),'parts_100_110')
    def test_the_overlap_label(self):
        from mocap import hidden_ranges as h
        self.assertEqual(h.OVERLAP_NOTE,'Parts overlap: blended')


import importlib.util,json,subprocess,tempfile
from pathlib import Path
from unittest import mock
from mocap import hidden_ranges as h,paths,people_store as ps
import test_take_join
from test_take_join import FFMPEG,TAKE,stamp


@unittest.skipUnless(importlib.util.find_spec('torch') and FFMPEG.is_file(),'needs torch and the bundled ffmpeg')
class OverlapWriteTest(unittest.TestCase):
    """A synthetic scene on disk whose person has two parts sharing frames 100-110, and a third part after a gap."""
    MODEL='gvhmr';take=test_take_join.WriteTest.take;join=test_take_join.WriteTest.join
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);root=self.root=Path(self.tmp.name)/'scene'
        models=Path(self.tmp.name)/'models';(models/'body_models/smplx').mkdir(parents=True)
        rng=np.random.default_rng(0);template=rng.normal(size=(5,3));self.pelvis=np.full(5,.2)@template
        np.savez(models/'body_models/smplx/SMPLX_NEUTRAL.npz',J_regressor=np.full((2,5),.2),v_template=template,shapedirs=rng.normal(size=(5,3,10))*.01)
        patch=mock.patch.object(paths,'MODELS',models);patch.start();self.addCleanup(patch.stop)
        root.mkdir();subprocess.run([str(FFMPEG),'-v','error','-f','lavfi','-i','testsrc=size=64x48:rate=30','-frames:v','220',
            '-pix_fmt','yuv420p','-c:v','libx264',str(root/'input.mp4')],check=True,creationflags=0x08000000)
        (root/'project.json').write_text(json.dumps(dict(frames=220,fps=30,start=0.,end=220/30,width=64,height=48,pipeline=dict(model=self.MODEL),stages={})),encoding='utf-8')
    def people(self,ranges,choices=None):
        takes=[self.take(a,b,W) for (a,b),W in zip(ranges,(WA,WB,WC))]
        marks={h.seam_key(x):dict(h.PARTS_SEAM) for x in h.seams(takes) if x.get('kind')}   # as Reconstruct in parts marks its parts
        person=dict(id=1,color='#b5b5b5',preview='p1',takes=takes,gap_fill={**marks,**(choices or {})})
        ps.save_people(self.root,dict(version=ps.VERSION,input=ps.media_receipt(self.root),people=[person]))
    def people_marks(self):
        state=ps.load_people(self.root);state['people'][0]['gap_fill']={'parts_100_110':dict(h.PARTS_SEAM)};ps.save_people(self.root,state)
        self.assertTrue(ps.join_current(self.root,ps.load_people(self.root)['people'][0],ps.load_people(self.root)['people'][0]['joins'][0]))
    def test_two_parts_become_one_current_joined_take(self):
        import torch
        self.people([(0,110),(100,200)])
        code,events=self.join();self.assertEqual(code,0,(events,self.stderr.getvalue()))
        self.assertEqual([e['type'] for e in events],['progress','takes_joined','complete'])
        J=self.root/'p1/takes/000000000_000000200_fill_v1';fill=json.loads((J/'fill.json').read_text(encoding='utf-8'))
        self.assertEqual((fill['gaps'],fill['overlaps'],fill['donors']),([],[dict(first=100,end=110)],[]))
        person=ps.load_people(self.root)['people'][0];join=person['joins'][0]
        self.assertEqual(join['sources'],[TAKE(0,110),TAKE(100,200)]);self.assertTrue(ps.join_current(self.root,person,join))
        self.assertEqual(ps.active_takes(self.root,person),[dict(project=join['project'],first=0,end=200,join=True)])
        self.assertFalse(np.load(J/'fill_mask.npy').any())
        record=torch.load(J/'hmr4d_results.pt',weights_only=False);g,c=record['smpl_params_global'],record['smpl_params_incam']
        t=rigid_params({k:c[k].double().numpy() for k in('global_orient','transl')},np.linalg.inv(WA),self.pelvis,self.pelvis)
        np.testing.assert_allclose(g['transl'].numpy(),t['transl'],atol=1e-4)   # one motion in A's world, blended frames included
        self.assertIs(self.join()[1][-1]['cached'],True)
        state=ps.load_people(self.root);state['people'][0]['gap_fill']={};ps.save_people(self.root,state)
        self.assertFalse(ps.join_current(self.root,ps.load_people(self.root)['people'][0],join))   # the parts mark is gone
        self.people_marks()
        fill['overlaps']=[dict(first=101,end=110)];(J/'fill.json').write_text(json.dumps(fill),encoding='utf-8')
        self.assertFalse(ps.join_current(self.root,person,join))   # a recorded overlap the takes no longer have
    def test_an_overlap_and_a_gap_in_one_joined_take(self):
        self.people([(0,110),(100,190),(200,220)],{'190_200':{'method':'smooth'}})
        code,events=self.join();self.assertEqual(code,0,(events,self.stderr.getvalue()))
        J=self.root/'p1/takes/000000000_000000220_fill_v1';fill=json.loads((J/'fill.json').read_text(encoding='utf-8'))
        self.assertEqual(fill['gaps'],[dict(first=190,end=200,method='smooth',params={})]);self.assertEqual(fill['overlaps'],[dict(first=100,end=110)])
        self.assertEqual(list(np.flatnonzero(np.load(J/'fill_mask.npy'))),list(range(190,200)))
        person=ps.load_people(self.root)['people'][0];self.assertTrue(ps.join_current(self.root,person,person['joins'][0]))
        state=ps.load_people(self.root);del state['people'][0]['gap_fill']['190_200'];ps.save_people(self.root,state)   # the gap is switched off
        person=ps.load_people(self.root)['people'][0];self.assertFalse(ps.join_current(self.root,person,person['joins'][0]))
    def test_overlapping_takes_that_are_not_marked_parts_are_not_joined(self):
        self.people([(0,110),(100,200)]);state=ps.load_people(self.root);state['people'][0]['gap_fill']={};ps.save_people(self.root,state)
        self.assertEqual(self.join()[1][-1]['message'],'No hidden range is selected for filling.')
    def test_touching_parts_are_not_joined(self):
        self.people([(0,100),(100,200)]);before={p:stamp(p) for p in self.root.rglob('*') if p.is_file()}
        code,events=self.join()
        self.assertEqual((code,events[-1]['message']),(1,'No hidden range is selected for filling.'))
        self.assertEqual({p:stamp(p) for p in self.root.rglob('*') if p.is_file()},before)


if __name__=='__main__':unittest.main()
