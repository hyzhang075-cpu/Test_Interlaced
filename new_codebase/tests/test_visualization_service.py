from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from shape_sensing.contracts import CalibrationDataset, DesignGeometry, ShapeResult, EvaluationResult
from shape_sensing.visualization.service import VisualizationInputs, VisualizationService


def inputs():
    design=DesignGeometry('visual-test',.0002,np.array([.1,.2,.3]),(1,3,4),{})
    dataset=CalibrationDataset({'circ':{0:{1000:np.zeros((3,3))}}},{0:{'circ':[1000]}},design,(1,3,4))
    return VisualizationInputs(dataset, segmentation=(3,))


def test_visualization_saves_original_mp4_with_explicit_ideal_overlay(monkeypatch,tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import shape_sensing.visualization.plots as plots
    monkeypatch.chdir(tmp_path)
    saved=[]
    class Animation:
        def __init__(self,figure,*args,**kwargs): self.figure=figure
        def save(self,path,writer,fps):
            saved.append((Path(path),writer,fps,[line.get_label() for line in self.figure.axes[0].lines]))
    monkeypatch.setattr(plots.animation,'FuncAnimation',Animation)
    curve=np.array([[0.,0.,0.],[.1,.05,0.],[.2,.1,.02]])
    original=curve.copy()
    service=VisualizationService(inputs())
    service.show_reconstruction(0,{'circ':{1000:curve}},np.array([0.,.1,.2]),ideals={'circ':{1000:curve*.9}})
    assert len(saved)==1 and saved[0][0].suffix=='.mp4'
    assert saved[0][0].parent.is_dir()
    assert saved[0][1:] == ('ffmpeg',10,['result-1000','ideal-1000'])
    np.testing.assert_array_equal(curve,original)
    with pytest.raises(ValueError,match='ideal curves'):
        service.show_reconstruction(0,{},np.array([0.,1.]))


def test_shape_plot_reads_explicit_evaluation_without_mutating_arrays():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    position=np.array([[0.,0.,0.],[1.,1.,1.]])
    result=ShapeResult((),'TNB',position_m=position)
    figure=VisualizationService().plot_shape(result,ideal_position_m=position*.9,evaluation=EvaluationResult({'error_m':.1}))
    assert len(figure.axes[0].lines)==2
    assert 'error_m' in figure.axes[0].get_title()
    np.testing.assert_array_equal(position,[[0.,0.,0.],[1.,1.,1.]])
    plt.close(figure)


def test_plot_segments_consume_explicit_configuration_only():
    from shape_sensing.visualization.segments import PlotSegments
    segmenter=PlotSegments(replace(inputs(),segmentation=(1,2),connect=True))
    distance,angle,segments=segmenter._segment_data(np.array([0.,.1,.2]))
    np.testing.assert_array_equal(distance,[.1,.2,.3])
    assert segments == [(0,1),(0,3)]
