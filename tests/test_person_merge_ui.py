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
        self.loaded=[];self.requested=[];self.kept=[];self.sought=[]
    def host(self,project=None,model='gvhmr',busy=False):
        status=SimpleNamespace(value='');status.setText=lambda t:setattr(status,'value',t);status.text=lambda:status.value
        slider=SimpleNamespace(position=0,maximum=lambda:299);slider.value=lambda:slider.position;slider.setValue=lambda v:setattr(slider,'position',v)
        host=SimpleNamespace(busy=busy,contact_saving=False,people_root=self.root,people_state=self.ps.load_people(self.root),
            model_choice=SimpleNamespace(currentData=lambda:model),info=None,project=project,active_result='gvhmr',status=status,log=[],
            load_project=lambda path,keep_current_options=False:self.kept.append(keep_current_options) or self.loaded.append(path),_hidden_refresh=False,source_offset=0,slider=slider,motion_view=SimpleNamespace(auto_fit=True),
            request_people_scene=lambda force=False:self.requested.append(force),update_readiness=lambda:self.requested.append('ready'),
            seek=lambda index:self.sought.append(index))
        for name in('_merge_people','_resync_people','_show_people','_scene_fps'):   # the helpers the handlers call on the window
            setattr(host,name,lambda *args,name=name:getattr(self.W,name)(host,*args))
        return host
    def fail_later(self):
        """Leave the pending file of Continue as Cover into Anna, as a failure after the database step does."""
        from mocap import person_merge as m
        with mock.patch.object(m,'_write_person_ids',side_effect=OSError('stopped')),self.assertRaises(m.Unfinished):m.continue_as(self.root,3,1,'gvhmr',30)
        self.assertTrue((self.root/m.PENDING).is_file())
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
        self.assertEqual(h.status.text(),'Cover now continues as Anna. Right-click Anna and choose Separate again to undo.')
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
        self.assertEqual(h.status.text(),'Ben and Anna are both visible in 254 frames, so they cannot be the same person.')
        self.assertEqual(self.loaded,[]);self.assertEqual(self.requested,[]);self.assertEqual(len(h.people_state['people']),3)
        self.tm.assert_same(self,before,self.tm.snapshot(self.root))
    def test_an_error_is_reported_with_its_cause(self):
        h=self.host()
        with mock.patch('mocap.person_merge.continue_as',side_effect=OSError('disk full')):self.W.continue_person_as(h,3,1)
        self.assertEqual(h.status.text(),'Cannot change the people of this scene: disk full');self.assertEqual(self.loaded,[])
    def test_the_reload_keeps_the_options_in_use(self):
        h=self.host(project=self.path(1,0,120));self.W.continue_person_as(h,3,1)
        self.assertEqual(self.kept,[True])   # like switching the person: the camera background and a pending result stay
    def test_a_take_that_cannot_be_opened_is_reported(self):
        h=self.host(project=self.path(1,0,120))
        def refuse(path,keep_current_options=False):h.status.setText('Cannot open project: broken');return False
        h.load_project=refuse
        self.W.continue_person_as(h,3,1)
        self.assertEqual([p['id'] for p in h.people_state['people']],[1,2]);self.assertEqual(self.requested,[]);self.assertFalse(h._hidden_refresh)
        self.assertEqual(h.status.text(),'Cover now continues as Anna. Right-click Anna and choose Separate again to undo. Cannot open project: broken')
    def test_person_row_tooltip_is_shared(self):
        import inspect
        from mocap import people_ui
        self.assertIn('Right-click to rename, or to join this person with another one.',people_ui.person_tooltip('Anna',5))
        self.assertEqual(inspect.getsource(people_ui).count('Right-click to rename'),1)   # sync_people and set_person_name both use person_tooltip
    def test_separate_reloads_and_reports(self):
        h=self.host(project=self.path(1,0,120));self.W.continue_person_as(h,3,1);self.loaded.clear();self.requested.clear()
        self.W.separate_person(h,1,3)
        self.assertEqual(len(h.people_state['people']),3);self.assertEqual(h.status.text(),'Cover is a separate person again.')
        self.assertEqual(len(self.loaded),1);self.assertTrue(h._hidden_refresh);self.assertEqual(self.requested,[True,'ready'])
    def test_busy_does_nothing(self):
        before=self.tm.snapshot(self.root)
        for kw in({'busy':True},{'contact_saving':True},{'people_state':None}):
            h=self.host();h.__dict__.update(kw)
            self.W.continue_person_as(h,3,1);self.W.separate_person(h,1,3)
            self.assertEqual(h.status.text(),'')
        self.assertEqual(self.loaded,[]);self.tm.assert_same(self,before,self.tm.snapshot(self.root))
    def test_sync_people_finishes_an_interrupted_operation(self):
        from mocap import person_merge as m
        self.fail_later()
        h=self.host();self.W.recover_people_merge(h,self.root)
        self.assertEqual(len(self.ps.load_people(self.root)['people']),2);self.assertFalse((self.root/m.PENDING).exists())
        (self.root/m.PENDING).write_text('{',encoding='utf-8')
        self.W.recover_people_merge(h,self.root)
        self.assertEqual(h.status.text(),m.UNREADABLE);self.assertEqual(h.log,[m.UNREADABLE])
        self.assertFalse((self.root/m.PENDING).exists());self.assertTrue((self.root/m.SET_ASIDE).is_file())
        h.status.value=''
        self.W.recover_people_merge(h,self.root)   # no pending file: nothing happens
        self.assertEqual(h.status.text(),'')
    def test_a_failing_recovery_does_not_stop_the_scene_from_opening(self):
        self.fail_later();h=self.host()
        with mock.patch('mocap.person_merge.recover',side_effect=RuntimeError('database disk image is malformed')):self.W.recover_people_merge(h,self.root)
        self.assertEqual(h.status.text(),'database disk image is malformed');self.assertEqual(h.log,['database disk image is malformed'])
    def test_a_failure_then_a_retry_keeps_the_scene_consistent(self):
        from mocap import person_merge as m
        h=self.host(project=self.path(1,0,120))
        with mock.patch.object(m,'_write_person_ids',side_effect=OSError('stopped')):self.W.continue_person_as(h,3,1)
        self.assertEqual(h.status.text(),'The change could not be finished: stopped. It will be finished when this scene is opened again.')
        self.assertEqual(h.people_state,self.ps.load_people(self.root));self.assertTrue((self.root/m.PENDING).is_file())
        self.W.continue_person_as(h,3,1)
        self.assertEqual(h.status.text(),'An earlier change of the people of this scene was finished. Check the list and choose again.')
        self.assertEqual(h.people_state,self.ps.load_people(self.root));self.assertEqual(self.loaded,[self.path(1,0,120)])
        self.assertFalse((self.root/m.PENDING).exists())
        self.ps.save_people(self.root,h.people_state)   # what activate_person does on the next click of a row
        state=self.ps.load_people(self.root)
        self.assertEqual(len(state['people']),2);self.assertEqual(m.absorbed(state['people'][0]),[(3,'Cover')])
    def test_recovery_before_a_new_request(self):
        from mocap import person_merge as m
        self.fail_later();h=self.host()
        with mock.patch.object(m,'continue_as',wraps=m.continue_as) as run:self.W.continue_person_as(h,2,1)
        run.assert_not_called();self.assertFalse((self.root/m.PENDING).exists())
        self.assertEqual(h.status.text(),'An earlier change of the people of this scene was finished. Check the list and choose again.')
        self.assertEqual([p['id'] for p in h.people_state['people']],[1,2]);self.assertEqual(h.people_state,self.ps.load_people(self.root))
    def test_a_refusal_after_a_change_on_disk_reloads_the_list(self):
        from mocap import person_merge as m
        h=self.host();m.continue_as(self.root,3,1,'gvhmr',30)   # changed behind the window's back
        self.W.continue_person_as(h,3,1)
        self.assertEqual(h.status.text(),'Choose another person of this scene.')
        self.assertEqual(h.people_state,self.ps.load_people(self.root));self.assertEqual(len(self.loaded),1)
    def test_an_out_of_date_joined_take_is_left(self):
        import json
        from test_take_joins import stamp
        root=self.root
        join=f'{self.tm.PREVIEW(1)}/takes/000000000_000000300_fill_v1';d=root/join;d.mkdir(parents=True)
        (d/'project.json').write_text(json.dumps(dict(people_root='../../../../..')),encoding='utf-8');(d/'hmr4d_results.pt').write_bytes(b'')
        state=self.ps.load_people(root);anna=state['people'][0]
        fill=dict(version=1,person=1,kind='gvhmr',first=0,end=300,gaps=[dict(first=120,end=166,method='smooth',params={})],
            sources=[dict(project=t['project'],first=t['first'],end=t['end'],files={'hmr4d_results.pt':stamp(root/t['project']/'hmr4d_results.pt')}) for t in anna['takes']])
        (d/'fill.json').write_text(json.dumps(fill),encoding='utf-8')
        anna['joins']=[dict(project=join,first=0,end=300,sources=[t['project'] for t in anna['takes']])];self.ps.save_people(root,state)
        self.assertEqual([t['project'] for t in self.ps.active_takes(root,anna,'gvhmr')],[join])   # the join is current before Continue as
        h=self.host(project=self.ps.safe_child(root,join));self.W.continue_person_as(h,3,1)
        sources=[self.path(1,0,120),self.path(1,166,300),self.path(7,121,136)]
        self.assertEqual(len(self.loaded),1);self.assertIn(self.loaded[0],sources)
    def test_the_overlap_limit_uses_the_scene_frame_rate(self):
        import json
        self.tm.add_person(self.root,4,'Late',9,range(100,140),[(100,140)])   # 20 frames shared with Anna: over 0.5 s at 30 fps, under it at 60
        h=self.host();h.info={'fps':30}
        self.W.continue_person_as(h,4,1)
        self.assertEqual(h.status.text(),'Late and Anna are both visible in 20 frames, so they cannot be the same person.')
        (self.root/'project.json').write_text(json.dumps(dict(fps=60)),encoding='utf-8')
        self.W.continue_person_as(h,4,1)
        self.assertEqual(h.status.text(),'Late now continues as Anna. Right-click Anna and choose Separate again to undo.')
    def test_the_playhead_keeps_its_time(self):
        cover=self.path(7,121,136);h=self.host(project=cover);h.info={'fps':30};h.source_offset=121/30;h.slider.position=5
        def load(path,keep_current_options=False):self.loaded.append(path);h.slider.position=0   # a reload starts at the first frame
        h.load_project=load
        self.W.continue_person_as(h,3,1)
        self.assertEqual(self.loaded,[cover]);self.assertEqual(h.slider.position,5);self.assertFalse(h.motion_view.auto_fit)
        self.assertEqual(self.sought,[5])   # the video and the 3D view show that frame, also when the timeline never left it
    def test_help_mentions_continue_as(self):
        from mocap.people_ui import PEOPLE_HELP
        self.assertIn('Right-click a row to rename it, or to join it with another person (Continue as) and undo that (Separate again).',PEOPLE_HELP)


if __name__=='__main__':unittest.main()
