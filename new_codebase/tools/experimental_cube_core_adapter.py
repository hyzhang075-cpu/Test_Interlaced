"""Opt-in experiment: channel-level core inversion with explicit WL imputation.

No file access, fitting, reference radii or ideal shapes in this forward adapter.
This does not enable the experiment in the production pipeline.
"""
from dataclasses import replace
import numpy as np
from shape_sensing.curvature.four_core import compute_four_core
from shape_sensing.curvature.core_regions import CoreDemoCurvature

class ExperimentalCoreAdapter:
    def __init__(self,service,design,parameters,*,fusion=False,allow_imputation=False):
        self.service=service;self.design=design;self.parameters=parameters
        self.fusion=fusion;self.allow_imputation=allow_imputation;self.regional={}
        block=parameters.parameters_by_region['full']
        self.imputed_ids=tuple(block.get('quality_imputation',{}).get('wl_ids',()))
        if self.imputed_ids and not allow_imputation:
            raise ValueError('Experimental candidate requires explicit WL-imputation opt-in')
    def channel_compute(self,frame,design,parameters,*,direct=0,**kwargs):
        block=parameters.parameters_by_region['full']
        calibration={**block['core_position_metadata'],'sections':block['core_position']}
        raw=compute_four_core(frame,design,calibration,None,1.)
        mapping=block['channel_weights'];weights=np.asarray(mapping.get(direct,mapping.get(str(direct))))
        expected=(len(frame.distance_m),len(self.service.config.thresholds)+1,4)
        if weights.shape!=expected or not np.isfinite(weights).all() or np.any(weights<=0):
            raise ValueError('Invalid experimental per-WL/channel/level gain array')
        mass=self.service.allocate(self.service.scores(raw.curvature_per_m))
        gain=np.einsum('lp,plc->pc',mass,weights)
        bias_mapping=block.get('signal_bias_pm',{})
        bias=bias_mapping.get(direct,bias_mapping.get(str(direct)))
        corrected=frame
        if bias is not None:
            bias=np.asarray(bias,dtype=float)
            if bias.shape!=frame.signals.shape or not np.isfinite(bias).all():
                raise ValueError('Invalid per-WL/channel input calibration offset')
            corrected=replace(frame,signals=frame.signals-bias*1e-6)
        return compute_four_core(corrected,design,calibration,gain,1.)
    def compute(self,frame,design,parameters,*,direct=0,**kwargs):
        if self.fusion:
            if direct not in self.regional:
                fusion=CoreDemoCurvature(self.service,design,parameters,('full','end15cm','end5cm'),direct)
                name,indices,local,_=fusion._regions[0]
                owner=self
                class Full:
                    def compute(self,frame,design,parameters,**kwargs):return owner.channel_compute(frame,design,parameters,**kwargs)
                fusion._regions[0]=(name,indices,local,Full());self.regional[direct]=fusion
            physical=self.regional[direct].compute(frame,parameters,direct=direct)
        else:physical=self.channel_compute(frame,design,parameters,direct=direct)
        if not self.imputed_ids:return physical
        wl_ids=np.asarray(physical.source_indices)+1
        if not set(self.imputed_ids).issubset(set(wl_ids.tolist())):
            raise ValueError('Imputation references a WL outside the physical grid')
        good=~np.isin(wl_ids,self.imputed_ids)
        if good.sum()<2:raise ValueError('At least two directly measured WLs must remain')
        vector=physical.curvature_per_m[:,None]*np.column_stack((np.cos(physical.direction_rad),np.sin(physical.direction_rad)))
        interpolated=np.column_stack([np.interp(physical.distance_m,physical.distance_m[good],vector[good,j]) for j in range(2)])
        return replace(physical,curvature_per_m=np.linalg.norm(interpolated,axis=1),direction_rad=np.arctan2(interpolated[:,1],interpolated[:,0]),diagnostics={**physical.diagnostics,'experimental_imputed_wl_ids':list(self.imputed_ids),'direct_measurement_mask':good.copy()})
