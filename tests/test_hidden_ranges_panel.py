"""Hidden ranges widgets and their glue in the People panel; needs the GUI runtime (PySide6), so it is skipped in the shared one."""
import json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
try:
    from PySide6.QtWidgets import QApplication,QCheckBox,QComboBox,QSpinBox
except ImportError:
    QApplication=None

T=lambda a,b:dict(project=f'p/takes/{a:09d}_{b:09d}_scene_v1',first=a,end=b)
PEOPLE=[dict(id=1,name='Person 1',takes=[T(0,100),T(120,200)]),dict(id=2,name='Person 2',takes=[T(0,300)])]


def application():
    app=QApplication.instance() or QApplication([])
    from mocap.theme import configure_application
    configure_application(app)
    return app


@unittest.skipIf(QApplication is None,'needs the GUI runtime (PySide6)')
class NarrowPanelTest(unittest.TestCase):
    def test_copy_controls_fit_a_narrow_people_panel(self):
        app=application()
        from mocap.hidden_ranges import gaps
        from mocap.hidden_ranges_ui import HiddenRangesPanel
        panel=HiddenRangesPanel()
        panel.show_gaps(gaps(PEOPLE[0]['takes']),{'100_120':dict(method='copy',person=2,mirror=False,offset=int(os.environ.get('RGB_MOCAP_TEST_OFFSET','-30')))},30,None,PEOPLE,1)
        panel.setFixedWidth(150);panel.show();app.processEvents()   # the People column of a default-size window
        panel.setFixedHeight(panel.heightForWidth(150) if panel.hasHeightForWidth() else panel.sizeHint().height());app.processEvents()
        shot=os.environ.get('RGB_MOCAP_TEST_SHOT')
        if shot:panel.grab().save(shot)
        mirror=panel.findChild(QCheckBox);offset=panel.findChild(QSpinBox)
        try:
            self.assertGreaterEqual(mirror.width(),mirror.sizeHint().width())
            self.assertGreaterEqual(offset.width(),offset.sizeHint().width())
        finally:panel.close()
    def test_rows_scroll_instead_of_overlapping_in_a_short_column(self):
        app=application()
        from PySide6.QtWidgets import QLabel,QScrollArea
        from mocap.hidden_ranges import gaps
        from mocap.hidden_ranges_ui import HiddenRangesPanel
        people=[dict(id=1,name='Person 1',takes=[T(0,100),T(120,200),T(225,313)]),dict(id=2,name='Person 2',takes=[T(0,100),T(120,200),T(225,313)])]
        copy=dict(method='copy',person=2,mirror=False,offset=0)                    # both rows show the long refusal
        panel=HiddenRangesPanel()
        panel.show_gaps(gaps(people[0]['takes']),{'100_120':copy,'200_225':copy},30,None,people,1)
        panel.setFixedSize(150,230);panel.show();app.processEvents();app.processEvents()   # what the default window leaves for it
        shot=os.environ.get('RGB_MOCAP_TEST_SHOT_SHORT')
        if shot:panel.grab().save(shot)
        try:
            self.assertGreaterEqual(panel.fill.geometry().bottom(),0);self.assertLessEqual(panel.fill.geometry().bottom(),panel.height())
            area=panel.findChild(QScrollArea);self.assertIsNotNone(area)
            shown=[w for w in panel._widgets if w.isVisible()]
            for w in shown:self.assertGreaterEqual(w.height(),w.minimumSizeHint().height(),w)
            for a,b in zip(shown,shown[1:]):self.assertLessEqual(a.geometry().bottom(),b.geometry().top(),(a,b))
            mirror=panel.findChild(QCheckBox);person=panel._widgets[2].findChild(QComboBox)
            self.assertLess(person.geometry().bottom(),mirror.geometry().top()+1)
        finally:panel.close()


@unittest.skipIf(QApplication is None,'needs the GUI runtime (PySide6)')
class ChoiceGlueTest(unittest.TestCase):
    """PeopleWorkspace methods run on a stand-in window over a real people folder."""
    def setUp(self):
        import test_take_joins as tj
        application()
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root,self.state,self.person=tj.build(self.tmp.name);self.root=self.root.resolve();self.S,self.J=tj.S,tj.J
        self.calls=[]
    def host(self,loaded,frame=150,**extra):
        from mocap import people_store as ps
        host=SimpleNamespace(people_state=ps.load_people(self.root),people_root=self.root,active_person_id=1,project=self.root/loaded,
            busy=False,contact_saving=False,info=dict(fps=30),source_offset=0.,slider=SimpleNamespace(value=lambda:frame),active_result='initial',
            status=SimpleNamespace(setText=lambda text:self.calls.append(('status',text))),
            activate_person=lambda ident,path=None:self.calls.append(('activate',ident,path)),
            populate_person_takes=lambda:self.calls.append(('populate',)),
            request_people_scene=lambda force=False:self.calls.append(('scene',force)),
            update_readiness=lambda:None,refresh_hidden_ranges=lambda person:None)
        host.current_person=lambda:next(p for p in host.people_state['people'] if p['id']==1)
        for key,value in extra.items():setattr(host,key,value)
        return host
    def choose(self,host,method):
        from mocap.people_ui import PeopleWorkspace
        PeopleWorkspace.gap_choice_changed(host,'100_120',method)
        return json.loads((self.root/'people.json').read_text(encoding='utf-8'))['people'][0]
    def test_switching_the_fill_off_leaves_the_joined_take(self):
        saved=self.choose(self.host(self.J,150),'none')
        self.assertNotIn('gap_fill',saved)
        self.assertEqual(self.calls,[('activate',1,self.root/self.S(120,200)),('scene',True)])
    def test_switching_the_fill_back_on_returns_to_the_joined_take(self):
        self.choose(self.host(self.J,10),'none');self.calls.clear()
        saved=self.choose(self.host(self.S(0,100),10),'smooth')
        self.assertEqual(saved['gap_fill'],{'100_120':{'method':'smooth'}})
        self.assertEqual(self.calls,[('activate',1,self.root/self.J),('scene',True)])
    def test_an_unchanged_take_in_use_is_not_reloaded(self):
        self.choose(self.host(self.J),'smooth')
        self.assertEqual(self.calls,[('populate',),('scene',True)])
    def test_a_running_job_keeps_the_loaded_take(self):
        self.choose(self.host(self.J,busy=True),'none')
        self.assertEqual(self.calls,[('populate',),('scene',True)])
    def ready(self,host,generation=7):
        from collections import OrderedDict
        from mocap.people_ui import PeopleWorkspace
        host.scene_loader=SimpleNamespace(generation=7);host.scene_cache=OrderedDict();host._scene_key=('root','initial')
        host.motion_view=SimpleNamespace(set_people_scene=lambda scene:None)
        for name in('sync_shared_floor_controls','sync_scene_visibility'):setattr(host,name,lambda:None)
        host.restore_workspace_camera=lambda consume=False:None
        host.refresh_results=lambda:self.calls.append(('reload',host.preview_signature))
        scene=SimpleNamespace(close=lambda:None)
        PeopleWorkspace.people_scene_ready(host,generation,scene)
    def test_a_choice_reads_the_loaded_take_again_once_the_scene_is_ready(self):
        # The shared floor is resolved again while the scene loads; only then is a corrected take current again.
        host=self.host(self.J,preview_signature='loaded before the floor was resolved',_hidden_refresh=False)
        self.choose(host,'smooth');self.calls.clear()
        self.ready(host,generation=6);self.assertEqual(self.calls,[])               # a superseded load keeps the request
        host.busy=True;self.ready(host);self.assertEqual(self.calls,[])              # so does a running job; its end loads the scene again
        host.busy=False;self.ready(host);self.assertEqual(self.calls,[('reload',None),('populate',)])
        self.calls.clear();self.ready(host);self.assertEqual(self.calls,[])           # an ordinary scene load does not
    def test_nothing_is_read_without_a_loaded_take(self):
        host=self.host(self.J,preview_signature=None,_hidden_refresh=False);host.project=None
        host._hidden_refresh=True
        self.ready(host);self.assertEqual(self.calls,[('populate',)])
    def listed(self,host):
        from mocap.people_ui import PeopleWorkspace
        from mocap.take_selection import TakeChoice
        person=host.current_person()
        rows=[(False,n,2,t) for n,t in enumerate(person['takes'],1)]+[(True,1,1,t) for t in person.get('joins') or []]
        host.take_choices=lambda people:[TakeChoice(1,'Person 1',n,total,t['first'],t['end'],self.root/t['project'],Path('motion.bvh'),None,joined)
                                         for joined,n,total,t in rows]
        host.person_takes=QComboBox()
        PeopleWorkspace.populate_person_takes(host)
        return [host.person_takes.itemText(i) for i in range(host.person_takes.count())],host.person_takes.currentText()
    def test_selector_lists_the_joined_take_in_use(self):
        items,current=self.listed(self.host(self.J))
        self.assertEqual(items,['Frames 1–100 · Ready','Frames 121–200 · Ready','Frames 1–200 · Ready · filled'])
        self.assertEqual(current,'Frames 1–200 · Ready · filled')
    def test_selector_leaves_out_a_joined_take_that_is_switched_off(self):
        self.choose(self.host(self.J),'none')
        items,_=self.listed(self.host(self.S(0,100)))
        self.assertEqual(items,['Frames 1–100 · Ready','Frames 121–200 · Ready'])

if __name__=='__main__':unittest.main()
