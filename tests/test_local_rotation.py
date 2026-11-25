import unittest
from contextlib import redirect_stdout
import io
import numpy as np
from orthogonal_runtime import LocalCurvatureDirectionCorrection as Correction
from reconstructor import Reconstructor
from tools.check_orthogonal_rotation import make_probe
import test_orthogonal_json_runtime as fixtures

class LocalCurvatureTests(unittest.TestCase):
    def setUp(self):
        self.s=np.linspace(0,.4,401)

    def test_interval_rotates_only_selected_components(self):
        s=self.s; magnitude=np.ones_like(s)*3; angles=np.full_like(s,.2)
        m,a,info=Correction(30,'[end15,end5]').apply(magnitude,angles,s)
        mask=info['local_rot_mask']
        np.testing.assert_array_equal(mask,(s>=.25)&(s<.35))
        np.testing.assert_array_equal(m,magnitude)
        np.testing.assert_array_equal(a[~mask],angles[~mask])
        np.testing.assert_allclose(a[mask],angles[mask]+np.pi/6)
        np.testing.assert_allclose(np.linalg.norm(info['curvature_components_after_local'],axis=1),magnitude)
        np.testing.assert_array_equal(info['curvature_components_after_local'][~mask],info['curvature_components_before_local'][~mask])

    def test_end5_only_bending_is_unchanged_and_zero_is_stable(self):
        s=self.s; m=np.where(s>=.35,4.,0.); a=s*0
        magnitude,angles,_=Correction(25,'[end15,end5]').apply(m,a,s)
        np.testing.assert_array_equal(magnitude,m)
        np.testing.assert_array_equal(angles,a)
        for rule in ('ht','TNN','bishop','bishop_RK4'):
            r=Reconstructor.__new__(Reconstructor); r.frame_rule=rule
            before=r._reconstruct_from_interpolated_curvature(m,a,s,[1,0,0])
            after=r._reconstruct_from_interpolated_curvature(magnitude,angles,s,[1,0,0])
            for actual, expected in zip(after,before): np.testing.assert_array_equal(actual,expected)

    def test_end15_bending_changes_for_all_integrators(self):
        s=self.s; m=np.where(s>=.25,3.,0.); a=s*0
        mm,aa,_=Correction(20,'[end15,end5]').apply(m,a,s)
        for rule in ('ht','TNN','bishop','bishop_RK4'):
            r=Reconstructor.__new__(Reconstructor); r.frame_rule=rule
            base=r._reconstruct_from_interpolated_curvature(m,a,s,[1,0,0])[0]
            after=r._reconstruct_from_interpolated_curvature(mm,aa,s,[1,0,0])[0]
            self.assertGreater(np.max(np.abs(after[:,2]-base[:,2])),1e-4)
            # Safely upstream of the first changed grid cell.
            np.testing.assert_allclose(after[s<.24],base[s<.24],atol=1e-12)

    def test_parser_disabled_sign_and_wrap(self):
        s=self.s; m=np.ones_like(s); a=np.linspace(3.,4.,len(s))
        _,aa,info=Correction(-15,'[ end15cm, end5 ]').apply(m,a,s)
        np.testing.assert_allclose(aa[info['local_rot_mask']],a[info['local_rot_mask']]-np.pi/12)
        _,disabled,_=Correction(0).apply(m,a,s)
        np.testing.assert_array_equal(disabled,a)
        _,_,tail=Correction(1,'end5').apply(m,a,s)
        self.assertTrue(tail['local_rot_mask'][-1])
        for region in ('[end5,end15]','[end5,end5]','[end15]x','[end15,end5,end1]','bad','end0'):
            with self.assertRaises(ValueError): Correction(1,region)
        with self.assertRaises(ValueError): Correction(float('nan'))
        with self.assertRaises(ValueError): Correction(1,'end50').apply(m,a,s)

    def test_actual_vtk_record_udp_reintegrated_once_after_kalman(self):
        fixture=fixtures.JsonRuntimeTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        with redirect_stdout(io.StringIO()):
            for rule in ('ht','TNN'):
                for kalman in (False,True):
                    demo,state,step=make_probe(fixture.file,20.,rule,kalman)
                    baseline=step()
                    demo.recon.local_curvature_correction=Correction(15,'[end15,end5]')
                    for _ in range(demo.recon.move_filter+1): actual=step()
                    record=state['record']; info=record['intermediates']
                    np.testing.assert_allclose(actual,record['r_results'],atol=1e-10)
                    np.testing.assert_allclose(actual,np.asarray(state['packet']['data'])/1000,atol=1e-10)
                    self.assertGreater(np.max(np.linalg.norm(actual-baseline,axis=1)),1e-6)
                    diff=info['ang_interp']-info['ang_interp_before_local']
                    active=info['local_rot_mask']
                    np.testing.assert_allclose(diff[active],np.deg2rad(15),atol=1e-12)
                    np.testing.assert_array_equal(diff[~active],0)
                    np.testing.assert_allclose(np.asarray(record['end_pose']) @ np.asarray(record['end_pose']).T,np.eye(3),atol=1e-8)
