import unittest
from pathlib import Path
from mocap.take_selection import TakeChoice

C=lambda number,total,joined=False,name='Anna':TakeChoice(1,name,number,total,0,313,Path('x'),None,None,joined)

class TakeLabelsTest(unittest.TestCase):
    def test_source_takes_keep_numbering(self):
        self.assertEqual(C(2,3).stem,'Anna_take_02');self.assertEqual(C(2,3).label,'Anna, Take 2 (frames 1–313)')
        self.assertEqual(C(1,1).stem,'Anna')
    def test_single_joined_take(self):
        self.assertEqual(C(1,1,True).stem,'Anna_filled')
        self.assertEqual(C(1,1,True).label,'Anna, Filled take (frames 1–313)')
    def test_several_joined_takes(self):
        self.assertEqual(C(2,2,True).stem,'Anna_filled_02');self.assertEqual(C(1,2,True).stem,'Anna_filled_01')
    def test_joined_stem_is_safe(self):
        self.assertEqual(C(1,1,True,'CON').stem,'_CON_filled')

if __name__=='__main__':unittest.main()
