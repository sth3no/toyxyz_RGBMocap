"""Import take from another project in the People list, its dialog, and the real main window offscreen; needs the GUI runtime (PySide6)."""
import importlib.util,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication,QComboBox,QDialogButtonBox,QDoubleSpinBox

from mocap import paths,people_store as ps
from test_take_import import FFMPEG,IMPORTED,SOURCE,TAKE,Scene

APP=QApplication.instance() or QApplication([])
ACTION='Import take from another project...'


def application():
    if not getattr(application,'configured',False):
        from mocap.theme import configure_application
        configure_application(APP);application.configured=True


class MenuTest(unittest.TestCase):
    def test_the_action_sits_next_to_continue_as_and_follows_the_window(self):
        from mocap.people_ui import PeopleList
        lst=PeopleList();lst.person_choices=lambda ident:([(1,'Anna')],[]);got=[]
        lst.importRequested.connect(got.append)
        for allowed in(True,False):
            lst.import_choice=lambda ident,allowed=allowed:allowed
            menu,actions=lst.person_menu(3)
            self.assertEqual([a.text() for a in menu.actions()],['Rename...',ACTION,'Continue as'])
            action=menu.actions()[1];self.assertIs(action.isEnabled(),allowed)
            self.assertEqual(action.toolTip(),"Continue this person with a finished take from another video's project.")
        actions[action]();self.assertEqual(got,[3])
        lst.import_choice=lambda ident:None   # a scene that cannot take an import: not offered
        self.assertNotIn(ACTION,[a.text() for a in lst.person_menu(3)[0].actions()])


@unittest.skipUnless(importlib.util.find_spec('torch') and FFMPEG.is_file(),'needs torch and the bundled ffmpeg')
class DialogTest(Scene,unittest.TestCase):
    def setUp(self):
        application();self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.build(Path(self.tmp.name))
    def dialog(self):
        from mocap.take_import_ui import ImportTakeDialog
        d=ImportTakeDialog(None,'Anna');self.addCleanup(d.deleteLater);return d
    def test_lists_finished_takes_with_model_and_length(self):
        d=self.dialog();ok=d.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok);gap=d.findChild(QDoubleSpinBox)
        self.assertFalse(ok.isEnabled());self.assertIsNone(d.choice());self.assertEqual(gap.value(),1.0);self.assertEqual(ok.text(),'Import')
        d.set_folder(self.other.parent);takes=d.findChild(QComboBox)
        self.assertEqual([takes.itemText(i) for i in range(takes.count())],['Dancer · frames 1–90 · GVHMR · 3.0 s'])
        self.assertTrue(ok.isEnabled());gap.setValue(2.5);self.assertEqual(d.choice(),(self.other/SOURCE,2.5))
        self.assertEqual((gap.minimum(),gap.maximum()),(0.1,9.6))
        d.set_folder(self.root/'p1');self.assertEqual(takes.count(),0);self.assertFalse(ok.isEnabled())
        self.assertTrue(d.empty.isVisibleTo(d));self.assertEqual(d.empty.text(),'This folder has no finished take. Choose the folder of another project.')


def pump(until,seconds=120):
    deadline=time.monotonic()+seconds
    while not until() and time.monotonic()<deadline:APP.processEvents();time.sleep(.02)
    return until()


@unittest.skipUnless(importlib.util.find_spec('torch') and FFMPEG.is_file(),'needs torch and the bundled ffmpeg')
class WindowTest(Scene,unittest.TestCase):
    """The real window: the action, the job through the window's own queue, and the Hidden ranges row of the new gap."""
    def setUp(self):
        application();self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);base=Path(self.tmp.name);self.build(base)
        home=mock.patch.object(paths,'HOME',base);home.start();self.addCleanup(home.stop)
        self.stages=[];models=paths.MODELS
        def command(stage,*args):   # the import runs for real in its own process; export would need the real body model, so it only reports
            self.stages.append(stage)
            script=(f"import sys;from pathlib import Path;from mocap import paths;paths.MODELS=Path({str(models)!r});from mocap import take_import;sys.exit(take_import.main(sys.argv[1:]))"
                    if stage=='take_import' else "print('@@MOCAP@@{\"type\": \"complete\", \"stage\": \"export\", \"elapsed\": 0}')")
            return [sys.executable,'-B','-c',script,*args]
        patch=mock.patch('mocap.gui.worker_command',command);patch.start();self.addCleanup(patch.stop)
        from mocap.gui import MainWindow
        w=self.w=MainWindow()
        def close():
            w.close();w.deleteLater();APP.processEvents()
        self.addCleanup(close);w.show();APP.processEvents()
        w.load_project(self.root/TAKE);APP.processEvents()
        self.assertEqual(w.people_root,self.root.resolve())
    def run_import(self,seconds=1.0):
        with mock.patch('mocap.take_import_ui.ask',lambda parent,person:(self.other/SOURCE,seconds)):self.w.people_list.importRequested.emit(1)
        self.assertTrue(pump(lambda:not self.w.busy),self.w.status.text())
    def test_import_lists_the_new_gap_like_a_tracker_gap(self):
        from mocap.hidden_ranges import gaps
        from mocap.hidden_ranges_ui import HiddenRangesPanel
        w=self.w;self.assertIs(w.import_choice(1),True)
        self.run_import()
        self.assertFalse(w.failed,w.status.text());self.assertEqual(self.stages,['take_import','export'])
        person=next(p for p in w.people_state['people'] if p['id']==1)
        self.assertEqual(person['takes'][-1],dict(project=IMPORTED,first=130,end=220))
        panel=w.hidden_ranges;self.assertTrue(panel.isVisibleTo(w))
        tracker=HiddenRangesPanel();self.addCleanup(tracker.deleteLater)   # a gap the tracker left between two takes of the same frames
        tracker.show_gaps(gaps([dict(project='a',first=0,end=100),dict(project='b',first=130,end=220)]),{},30,None,[person],1)
        for a,b in((panel,tracker),):
            self.assertEqual(a._widgets[0].text(),'Frames 101–130 (1.0 s)');self.assertEqual(a._widgets[0].text(),b._widgets[0].text())
            items=lambda c:[(c.itemText(i),c.itemData(i),c.model().item(i).isEnabled()) for i in range(c.count())]
            self.assertEqual(items(a._widgets[1]),items(b._widgets[1]))
        self.assertEqual(w.project,(self.root/TAKE).resolve())   # the view stays on the take that was open
    def test_the_action_is_disabled_while_a_job_runs(self):
        self.w.busy=True
        try:self.assertIs(self.w.import_choice(1),False)
        finally:self.w.busy=False
    def test_a_refusal_is_shown_in_the_status_bar_and_starts_nothing(self):
        from mocap import take_import_check as check
        self.run_import(seconds=20.)
        self.assertEqual(self.w.status.text(),check.GAP);self.assertEqual(self.stages,[])
        self.assertEqual(len(ps.load_people(self.root)['people'][0]['takes']),1)
    def test_the_imported_take_opens_beyond_the_video_end(self):
        self.run_import();w=self.w
        w.activate_person(1,ps.safe_child(self.root,IMPORTED));APP.processEvents()
        self.assertEqual(w.project,(self.root/IMPORTED).resolve());self.assertEqual(w.slider.maximum(),89)
        w.slider.setValue(89)
        self.assertTrue(pump(lambda:w.video_view.index==89 and w.video_view.frame is not None,20),w.video_view.index)   # the padded video has the frame
        w.video_view.grab();self.assertFalse(w.failed)   # drawing it with the take's boxes (none) raises nothing
        self.assertEqual(json.loads((self.root/IMPORTED/'project.json').read_text(encoding='utf-8'))['frames'],90)


if __name__=='__main__':unittest.main()
