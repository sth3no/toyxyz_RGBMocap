"""Planning a long range in parts, and deleting a finished part's temporary ExMR frames; no Qt."""
import json,os,tempfile,unittest
from pathlib import Path
import numpy as np
from mocap import parts


class PlanTest(unittest.TestCase):
    def test_parts_overlap_and_the_last_absorbs_a_short_remainder(self):
        self.assertEqual(parts.plan(0,1000,300,60),[(0,300),(240,540),(480,780),(720,1000)])
        self.assertEqual(parts.plan(0,800,300,60),[(0,300),(240,540),(480,800)])   # 20 frames left: the last part takes them
        self.assertEqual(parts.plan(100,350,300,60),[(100,350)])   # shorter than a part: one part
        self.assertEqual(parts.plan(0,54000,1800,60)[-1][1],54000)
        for a,b in zip(parts.plan(0,54000,1800,60),parts.plan(0,54000,1800,60)[1:]):self.assertEqual(a[1]-b[0],60)
    def test_a_remainder_of_a_third_of_a_part_gets_its_own_part(self):
        self.assertEqual(parts.plan(0,640,300,60),[(0,300),(240,540),(480,640)])   # 100 frames left = a third: a part of its own
        self.assertEqual(parts.plan(0,639,300,60),[(0,300),(240,639)])
    def test_parts_are_the_frames_the_person_job_reconstructs(self):
        self.assertEqual(parts.job_range((225,300),0.,10.,30,300),(225,300))
        self.assertEqual(parts.job_range((225,300),0.,10.-1/30,30,300),(225,299))   # a scene end one frame short clips the last part
        self.assertEqual(parts.job_range((75,165),12.5,22.5,30,300),(75,165))   # a scene that starts later in the video
        self.assertEqual(parts.job_range((0,90),0.,10.,30,60),(0,60))   # the job keeps a range inside the scene's frames
    def test_seconds_become_frames_of_the_clip(self):
        self.assertEqual(parts.frames(60.,30),1800);self.assertEqual(parts.frames(2.,30),60);self.assertEqual(parts.frames(2.5,25),62)
    def test_invalid_settings_have_a_plain_reason(self):
        self.assertIsNone(parts.issue(1800,60))
        self.assertEqual(parts.issue(1800,900),'The overlap must be shorter than half a part.')
        self.assertEqual(parts.issue(1800,0),'The overlap must be at least one frame.')
        self.assertEqual(parts.issue(3,1),'A part must be at least 4 frames long.')
        with self.assertRaisesRegex(ValueError,'half a part'):parts.plan(0,5000,300,150)
        with self.assertRaisesRegex(ValueError,'The range is empty'):parts.plan(10,10,300,60)


class VisibleTest(unittest.TestCase):
    def test_runs_inside_a_part_are_those_the_person_job_makes(self):
        seen=np.zeros(300,bool);seen[20:100]=True;seen[200:280]=True
        self.assertEqual(parts.visible(seen,30,(0,150)),[(20,100)])
        self.assertEqual(parts.visible(seen,30,(90,250)),[(90,100),(200,250)])
        self.assertEqual(parts.visible(seen,30,(100,200)),[])
        self.assertEqual(parts.visible(seen,30,(98,200)),[])   # two frames: too short for a take
    def test_a_part_is_finished_when_every_run_has_a_ready_take_of_that_range(self):
        takes=[dict(project='a',first=20,end=100),dict(project='b',first=200,end=250)];ready=lambda p:p!='c'
        self.assertTrue(parts.finished(takes,[(20,100),(200,250)],ready))
        self.assertFalse(parts.finished(takes,[(20,100),(200,251)],ready))
        self.assertFalse(parts.finished(takes+[dict(project='c',first=300,end=310)],[(300,310)],ready))
        self.assertFalse(parts.finished(takes,[],ready))   # nothing to reconstruct is not "finished"


class MarksTest(unittest.TestCase):
    def test_only_seams_between_planned_parts_are_marked(self):
        T=lambda a,b:dict(project=f't{a}_{b}',first=a,end=b)
        takes=[T(0,90),T(75,165),T(150,240),T(160,250),T(240,300)]   # 160-250: a take of another trim
        runs={(0,90),(75,165),(150,240),(240,300)}
        self.assertEqual(parts.marks(takes,runs),{'parts_75_90':dict(method='parts',takes=['t0_90','t75_165']),
                                                  'parts_150_165':dict(method='parts',takes=['t75_165','t150_240'])})
        self.assertEqual(list(parts.marks([T(0,90),T(90,165)],{(0,90),(90,165)})),['parts_90_90'])   # meeting parts are marked, and refused when joined
        self.assertEqual(parts.marks([T(0,90),T(100,165)],{(0,90),(100,165)}),{})   # a gap keeps its own choice


class EstimateTest(unittest.TestCase):
    def test_frame_size_is_scaled_from_the_measured_1080p_frame(self):
        self.assertEqual(parts.FRAME_BYTES,188_000);self.assertEqual(parts.frame_bytes(1920,1080),188_000)
        self.assertEqual(parts.frame_bytes(1280,720),round(188_000*1280*720/(1920*1080)))
    def test_summary_line(self):
        plan=parts.plan(0,54000,1800,60)
        self.assertEqual(len(plan),31)
        self.assertEqual(parts.summary(plan,1920,1080,1,True),'31 parts, about 0.34 GB of temporary frames at a time.')
        self.assertEqual(parts.summary(plan,1920,1080,1,False),'31 parts, about 10.5 GB of temporary frames in total.')
        self.assertEqual(parts.summary(plan,1920,1080,2,False),'31 parts, about 21.0 GB of temporary frames in total.')
        self.assertEqual(parts.summary([(0,300)],1920,1080,1,True),'1 part, about 0.06 GB of temporary frames at a time.')
        self.assertEqual(parts.summary(plan,1920,1080,1,True,frames=False),'31 parts.')


class DeleteTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);base=Path(self.tmp.name)
        self.take=base/'scene/p1/takes/000000000_000000300_scene_v1';self.frames=self.take/'exmr_frames';self.frames.mkdir(parents=True)
        for i in range(5):(self.frames/f'{i:06d}.jpg').write_bytes(b'jpg')
        (self.frames/'receipt.json').write_text('{}',encoding='utf-8');(self.frames/'notes.txt').write_text('keep',encoding='utf-8')
        for name in('hmr4d_results.pt','input.mp4','exmr_keypoints.pt','frame.jpg'):(self.take/name).write_bytes(b'x')
        self.outside=base/'outside';self.outside.mkdir();(self.outside/'000001.jpg').write_bytes(b'other')
    def files(self):return sorted(p.relative_to(Path(self.tmp.name)).as_posix() for p in Path(self.tmp.name).rglob('*') if p.is_file())
    def test_only_the_frames_and_their_receipt_go(self):
        from mocap.parts_runner import delete_frames
        before=self.files();self.assertEqual(delete_frames(self.take),(6,0))
        gone=[p for p in before if p not in self.files()]
        self.assertEqual(gone,[f'scene/p1/takes/000000000_000000300_scene_v1/exmr_frames/{i:06d}.jpg' for i in range(5)]+
                         ['scene/p1/takes/000000000_000000300_scene_v1/exmr_frames/receipt.json'])
        self.assertTrue(self.frames.is_dir());self.assertEqual(delete_frames(self.take),(0,0))   # nothing left: nothing more
    def test_a_frame_another_program_holds_open_stays_and_is_counted(self):
        from mocap.parts_runner import delete_frames
        with open(self.frames/'000002.jpg','rb'):
            self.assertEqual(delete_frames(self.take),(5,1))
        self.assertEqual(sorted(p.name for p in self.frames.iterdir()),['000002.jpg','notes.txt'])
    def junction(self,link,target):
        import subprocess
        made=subprocess.run(['cmd','/c','mklink','/J',str(link),str(target)],capture_output=True,creationflags=0x08000000)
        if made.returncode or not link.is_dir():self.skipTest('a directory junction could not be made: '+made.stdout.decode(errors='replace'))
        self.addCleanup(lambda:link.exists() and os.rmdir(link))   # removes the junction itself, never what it leads to
    def test_a_frame_folder_that_is_a_junction_is_left_alone(self):
        import shutil
        from mocap.parts_runner import delete_frames
        shutil.rmtree(self.frames);self.junction(self.frames,self.outside)
        self.assertEqual(delete_frames(self.take),(0,0));self.assertTrue((self.outside/'000001.jpg').is_file())
    def test_a_junction_inside_the_frame_folder_is_not_followed(self):
        from mocap.parts_runner import delete_frames
        self.junction(self.frames/'000099.jpg',self.outside)   # a folder link named like a frame
        self.assertEqual(delete_frames(self.take),(6,0));self.assertTrue((self.outside/'000001.jpg').is_file())
    def test_a_link_out_of_the_folder_is_not_followed(self):
        import os
        from mocap.parts_runner import delete_frames
        try:os.symlink(self.outside,self.frames/'linked',target_is_directory=True);os.symlink(self.outside/'000001.jpg',self.frames/'000099.jpg')
        except OSError:self.skipTest('symbolic links need Developer Mode on Windows')
        delete_frames(self.take);self.assertTrue((self.outside/'000001.jpg').is_file())
    def test_a_frame_folder_that_leads_elsewhere_is_left_alone(self):
        import os,shutil
        from mocap.parts_runner import delete_frames
        shutil.rmtree(self.frames)
        try:os.symlink(self.outside,self.frames,target_is_directory=True)
        except OSError:self.skipTest('symbolic links need Developer Mode on Windows')
        self.assertEqual(delete_frames(self.take),(0,0));self.assertTrue((self.outside/'000001.jpg').is_file())
    def test_no_frame_folder(self):
        from mocap.parts_runner import delete_frames
        import shutil;shutil.rmtree(self.frames);self.assertEqual(delete_frames(self.take),(0,0))


if __name__=='__main__':unittest.main()
