"""Integration checks for shared static/live local curvature correction."""
import unittest
import numpy as np
from static_reconstruction import CurvatureSamplesReconstructor
from local_curvature_correction import LocalCurvatureDirectionCorrection
class LocalRotationTests(unittest.TestCase):
 def test_selected_components_preserve_magnitude(self):
  s=np.linspace(0,.9,40);k=np.linspace(1,3,40);a=np.zeros(40)
  kk,aa,m=LocalCurvatureDirectionCorrection(90,'end15').apply(k,a,s)
  np.testing.assert_array_equal(kk,k)
  np.testing.assert_array_equal(aa[~m['local_rot_mask']],a[~m['local_rot_mask']])
  np.testing.assert_allclose(aa[m['local_rot_mask']],np.pi/2)
 def test_integrated_path_changes_only_after_distal_region(self):
  s=np.linspace(0,.9,40);k=np.ones(40)*2;a=np.zeros(40)
  obj=CurvatureSamplesReconstructor(s,.9,frame_rule='ht',move_filter=1)
  original=obj.reconstruct(k,a)[0]
  obj.local_curvature_correction.angle=35
  corrected=obj.reconstruct(k,a)[0]
  np.testing.assert_allclose(original[:200],corrected[:200],atol=1e-7)
  self.assertGreater(np.linalg.norm(original[-1]-corrected[-1]),1e-3)
  obj.local_curvature_correction.angle=0
  np.testing.assert_array_equal(original,obj.reconstruct(k,a)[0])
if __name__=='__main__':unittest.main()
