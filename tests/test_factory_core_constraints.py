import unittest
import numpy as np
from factory_core_constraints import validate_factory_geometry

class FactoryConstraintTests(unittest.TestCase):
    def setUp(self):
        self.limits=dict(ch1_outer_distances_um=[41.43,41.88,41.31],distance_relative_tolerance=.05,angle_tolerance_deg=5,outer_angle_deg=120)
        a=np.arange(3)*2*np.pi/3
        self.points=np.vstack(([0.,0.],np.array([41.43,41.88,41.31])[:,None]*np.column_stack((np.cos(a),np.sin(a)))))

    def test_translation_rotation_and_mirror_invariance(self):
        a=.7;q=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        for points in (self.points,self.points@q+103,self.points*np.array([1,-1])):
            validate_factory_geometry(points,self.limits)

    def test_legacy_bundle_distance_rejected(self):
        with self.assertRaisesRegex(ValueError,'distance'):validate_factory_geometry(self.points*3,self.limits)

    def test_angle_violation_rejected_even_with_correct_distances(self):
        p=self.points.copy();a=np.deg2rad(100);p[2]=41.88*np.array([np.cos(a),np.sin(a)])
        with self.assertRaisesRegex(ValueError,'angle'):validate_factory_geometry(p,self.limits)

    def test_nine_um_core_nonoverlap_and_radius_validation(self):
        limits=dict(self.limits,core_radius_um=9.)
        validate_factory_geometry(self.points,limits)
        # Broaden distance limits so rejection specifically tests disk overlap.
        limits['distance_relative_tolerance']=.9
        with self.assertRaisesRegex(ValueError,'overlap'):
            validate_factory_geometry(self.points*.3,limits)
        for invalid in (0,-1,float('nan')):
            with self.assertRaisesRegex(ValueError,'radius'):
                validate_factory_geometry(self.points,dict(limits,core_radius_um=invalid))
