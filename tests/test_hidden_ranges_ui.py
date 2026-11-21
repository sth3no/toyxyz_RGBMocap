import unittest
import numpy as np
from mocap import hidden_ranges as h
from mocap import hidden_ranges_ui as ui
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
        self.assertIsNone(rows[0]['issue']);self.assertIn('60',rows[1]['issue'])
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
    def test_choose(self):
        choices={}
        ui.choose(choices,'100_120','smooth');self.assertEqual(choices,{'100_120':{'method':'smooth'}})
        ui.choose(choices,'100_120','none');self.assertEqual(choices,{})
        self.assertRaises(ValueError,ui.choose,choices,'100_120','copy')

class MaskTest(unittest.TestCase):
    def test_runs(self):
        self.assertEqual(ui.fill_runs(np.array([0,1,1,0,0,1,2,0,1],np.uint8)),((1,3),(5,7),(8,9)))
        self.assertEqual(ui.fill_runs(np.zeros(5,np.uint8)),());self.assertEqual(ui.fill_runs(None),())
    def test_line(self):
        self.assertEqual(ui.interpolated_line('Anna_filled',np.array([0,1,1,0],np.uint8)),
                         'Anna_filled: 2 of 4 frames are interpolated across hidden ranges.')

if __name__=='__main__':unittest.main()
