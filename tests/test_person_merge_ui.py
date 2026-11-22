"""Continue as / Separate again in the People list and its handlers; needs the GUI runtime (PySide6), so it is skipped in the shared one."""
import os,tempfile,unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
try:
    from PySide6.QtWidgets import QApplication
except ImportError:
    QApplication=None


@unittest.skipIf(QApplication is None,'needs the GUI runtime (PySide6)')
class PersonMergeUiTest(unittest.TestCase):
    def setUp(self):
        import test_person_merge as tm
        from mocap import people_store as ps
        from mocap.people_ui import PeopleList,PeopleWorkspace
        self.tm,self.ps,self.W,self.PeopleList=tm,ps,PeopleWorkspace,PeopleList
        self.app=QApplication.instance() or QApplication([])
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=tm.build(self.tmp.name)
        self.loaded=[];self.requested=[]
    def host(self,project=None,model='gvhmr',busy=False):
        status=SimpleNamespace(text='');status.setText=lambda t:setattr(status,'text',t)
        host=SimpleNamespace(busy=busy,contact_saving=False,people_root=self.root,people_state=self.ps.load_people(self.root),
            model_choice=SimpleNamespace(currentData=lambda:model),info=None,project=project,active_result='gvhmr',status=status,
            load_project=self.loaded.append,_hidden_refresh=False,
            request_people_scene=lambda force=False:self.requested.append(force),update_readiness=lambda:self.requested.append('ready'))
        host._merge_people=lambda *args:self.W._merge_people(host,*args)   # the one helper the handlers call on the window
        return host
    def path(self,obs,a,b):return self.ps.safe_child(self.root,self.tm.take(obs,a,b))

    def test_menu_of_a_person(self):
        lst=self.PeopleList();lst.person_choices=lambda ident:([(1,'Anna'),(2,'Ben')],[(4,'Late')])
        menu,actions=lst.person_menu(3)
        self.assertEqual([a.text() for a in menu.actions()],['Rename...','Continue as','Separate again'])
        self.assertEqual([a.text() for a in menu.actions()[1].menu().actions()],['Anna','Ben'])
        self.assertEqual([a.text() for a in menu.actions()[2].menu().actions()],['Late'])
        self.assertEqual(menu.actions()[1].toolTip(),"This person and the chosen one are the same person. Their frames and takes become one person's.")
        self.assertEqual(menu.actions()[2].toolTip(),'Undo Continue as for the chosen person.')
        got=[];lst.continueRequested.connect(lambda s,t:got.append(('continue',s,t)));lst.separateRequested.connect(lambda t,s:got.append(('separate',t,s)))
        actions[menu.actions()[1].menu().actions()[1]]();actions[menu.actions()[2].menu().actions()[0]]()
        self.assertEqual(got,[('continue',3,2),('separate',3,4)])
    def test_menu_without_choices_is_rename_only(self):
        lst=self.PeopleList()
        for choices in(None,lambda ident:([],[])):
            lst.person_choices=choices
            menu,actions=lst.person_menu(3)
            self.assertEqual([a.text() for a in menu.actions()],['Rename...']);self.assertEqual(len(actions),1)
    def test_choices_follow_model_and_busy(self):
        self.assertEqual(self.W.person_choices(self.host(model='flowhmr'),3),([],[]))
        self.assertEqual(self.W.person_choices(self.host(busy=True),3),([],[]))
        h=self.host();h.contact_saving=True;self.assertEqual(self.W.person_choices(h,3),([],[]))
        h=self.host();h.people_state=None;self.assertEqual(self.W.person_choices(h,3),([],[]))
        self.assertEqual(self.W.person_choices(self.host(),3),([(1,'Anna'),(2,'Ben')],[]))
        h=self.host();self.W.continue_person_as(h,3,1)
        self.assertEqual(self.W.person_choices(h,1),([(2,'Ben')],[(3,'Cover')]))
    def test_continue_reloads_a_take_of_the_target(self):
        anna=self.path(1,0,120);h=self.host(project=anna)
        self.W.continue_person_as(h,3,1)
        self.assertEqual(self.loaded,[anna]);self.assertEqual([p['id'] for p in h.people_state['people']],[1,2])
        self.assertTrue(h._hidden_refresh);self.assertEqual(self.requested,[True,'ready'])
        self.assertEqual(h.status.text,'Cover now continues as Anna. Right-click Anna and choose Separate again to undo.')
    def test_joining_the_person_whose_take_is_open(self):
        cover=self.path(7,121,136);h=self.host(project=cover)
        self.W.continue_person_as(h,3,1);self.assertEqual(self.loaded,[cover])
    def test_joining_the_person_whose_preview_is_open(self):
        h=self.host(project=self.ps.safe_child(self.root,self.tm.PREVIEW(7)))
        self.W.continue_person_as(h,3,1)
        anna=next(p for p in h.people_state['people'] if p['id']==1)
        self.assertEqual(self.loaded,[self.ps.safe_child(self.root,self.ps.active_takes(self.root,anna,'gvhmr')[-1]['project'])])
    def test_refusal_goes_to_the_status_bar(self):
        h=self.host();before=self.tm.snapshot(self.root)
        self.W.continue_person_as(h,2,1)
        self.assertEqual(h.status.text,'Ben and Anna are both visible in 254 frames, so they cannot be the same person.')
        self.assertEqual(self.loaded,[]);self.assertEqual(self.requested,[]);self.assertEqual(len(h.people_state['people']),3)
        self.tm.assert_same(self,before,self.tm.snapshot(self.root))
    def test_an_error_is_reported_with_its_cause(self):
        h=self.host()
        with mock.patch('mocap.person_merge.continue_as',side_effect=OSError('disk full')):self.W.continue_person_as(h,3,1)
        self.assertEqual(h.status.text,'Cannot change the people of this scene: disk full');self.assertEqual(self.loaded,[])
    def test_separate_reloads_and_reports(self):
        h=self.host(project=self.path(1,0,120));self.W.continue_person_as(h,3,1);self.loaded.clear();self.requested.clear()
        self.W.separate_person(h,1,3)
        self.assertEqual(len(h.people_state['people']),3);self.assertEqual(h.status.text,'Cover is a separate person again.')
        self.assertEqual(len(self.loaded),1);self.assertTrue(h._hidden_refresh);self.assertEqual(self.requested,[True,'ready'])
    def test_busy_does_nothing(self):
        before=self.tm.snapshot(self.root)
        for kw in({'busy':True},{'contact_saving':True},{'people_state':None}):
            h=self.host();h.__dict__.update(kw)
            self.W.continue_person_as(h,3,1);self.W.separate_person(h,1,3)
            self.assertEqual(h.status.text,'')
        self.assertEqual(self.loaded,[]);self.tm.assert_same(self,before,self.tm.snapshot(self.root))
    def test_sync_people_finishes_an_interrupted_operation(self):
        from mocap import person_merge as m
        with mock.patch.object(m,'_write_person_ids',side_effect=OSError('stopped')),self.assertRaises(OSError):m.continue_as(self.root,3,1,'gvhmr',30)
        self.assertTrue((self.root/m.PENDING).is_file())
        h=self.host();self.W.recover_people_merge(h,self.root)
        self.assertEqual(len(self.ps.load_people(self.root)['people']),2);self.assertFalse((self.root/m.PENDING).exists())
        (self.root/m.PENDING).write_text('{',encoding='utf-8')
        self.W.recover_people_merge(h,self.root)
        self.assertEqual(h.status.text,m.UNREADABLE)
        h.status.text=''
        (self.root/m.PENDING).unlink();self.W.recover_people_merge(h,self.root)   # no pending file: nothing happens
        self.assertEqual(h.status.text,'')


if __name__=='__main__':unittest.main()
