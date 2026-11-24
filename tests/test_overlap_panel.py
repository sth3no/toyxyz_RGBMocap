"""Hidden ranges rows of parts that share frames: Qt-free rows, and the panel widget offscreen (needs PySide6)."""
import os,unittest
from mocap import hidden_ranges as h,hidden_ranges_state as ui

T=lambda a,b:dict(project=f'p1/takes/{a:09d}_{b:09d}_scene_v1',first=a,end=b)
PARTS=[T(0,110),T(100,200),T(215,300)]
M=lambda *takes:{h.seam_key(s):h.parts_mark(s) for s in h.seams(list(takes)) if s.get('kind')}   # as Reconstruct in parts marks
MARK=M(*PARTS[:2]);TOUCH=M(T(0,100),T(100,200))


class RowsTest(unittest.TestCase):
    def test_overlaps_of_other_trims_have_no_row(self):
        self.assertEqual([r['label'] for r in ui.gap_rows(h.listed(PARTS,{}),{},30)],['Frames 201–215 (0.5 s)'])
    def test_an_overlap_row_needs_no_choice(self):
        o,g=ui.gap_rows(h.listed(PARTS,MARK),MARK,30)
        self.assertEqual((o['kind'],o['method'],o['label'],o['note'],o['issue']),('overlap','overlap','Frames 101–110 (0.3 s)','Parts overlap: blended',None))
        self.assertEqual((g.get('kind','gap'),g['method'],g['label']),('gap','none','Frames 201–215 (0.5 s)'))
        self.assertTrue(ui.can_fill([o,g],None));self.assertFalse(ui.can_fill([o,g],'Reconstruct and prepare both neighbouring takes first.'))
        self.assertEqual(ui.fill_sources(h.listed(PARTS,MARK),MARK),PARTS[:2])
        self.assertEqual(ui.fill_sources(h.listed(PARTS,MARK),{**MARK,'200_215':{'method':'smooth'}}),PARTS)
    def test_touching_takes_say_why_and_do_not_count(self):
        r,=ui.gap_rows(h.listed([T(0,100),T(100,200)],TOUCH),TOUCH,30)
        self.assertEqual((r['kind'],r['method'],r['note'],r['label']),('touch','none',h.TOUCH_NOTE,'Frame 101'))
        self.assertFalse(ui.can_fill([r],None));self.assertEqual(ui.fill_sources(h.listed([T(0,100),T(100,200)],TOUCH),TOUCH),[])
        rows=ui.gap_rows(h.listed([T(0,100),T(100,200),T(210,300)],TOUCH),{**TOUCH,'200_210':{'method':'smooth'}},30)
        self.assertTrue(ui.can_fill(rows,None))   # a touch elsewhere does not stop a chosen range
    def test_an_overlap_that_would_blend_a_whole_take_is_not_joined(self):
        both=M(T(0,110),T(100,210),T(105,300))
        x,y=ui.gap_rows(h.listed([T(0,110),T(100,210),T(105,300)],both),both,30)
        self.assertEqual(x['method'],'overlap');self.assertEqual((y['kind'],y['method'],y['note']),('overlap','none',h.SHARED_NOTE))


class PanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        try:from PySide6.QtWidgets import QApplication
        except ImportError:raise unittest.SkipTest('needs PySide6')
        cls.app=QApplication.instance() or QApplication([])
    def test_the_overlap_row_has_a_note_and_no_method_list(self):
        from PySide6.QtWidgets import QComboBox,QLabel
        from mocap.hidden_ranges_ui import HiddenRangesPanel
        panel=HiddenRangesPanel();self.addCleanup(panel.deleteLater)
        panel.show_gaps(h.listed(PARTS,MARK),MARK,30,None,[dict(id=1,takes=PARTS)],1);self.app.processEvents()
        texts=[w.text() for w in panel._widgets if isinstance(w,QLabel)]
        self.assertEqual(texts[:3],['Frames 101–110 (0.3 s)','Parts overlap: blended','Frames 201–215 (0.5 s)'])
        self.assertEqual(sum(isinstance(w,QComboBox) for w in panel._widgets),1)   # only the gap has a method
        self.assertEqual(panel._widgets[1].objectName(),'muted');self.assertTrue(panel.fill.isEnabled())
        self.assertEqual(panel.fill.text(),'Fill hidden ranges')
        panel.show_gaps(h.listed([T(0,100),T(100,200)],TOUCH),TOUCH,30,None,[],1);self.app.processEvents()
        self.assertEqual([w.text() for w in panel._widgets],['Frame 101',h.TOUCH_NOTE]);self.assertFalse(panel.fill.isEnabled())


if __name__=='__main__':unittest.main()
