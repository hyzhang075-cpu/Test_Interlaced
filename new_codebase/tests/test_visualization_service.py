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


@pytest.mark.parametrize('channels,signal_channels', [
    ((1, 2, 3, 4), (2, 3, 4)),
    ((4, 1, 3, 2), (4, 3, 2)),
    ((4, 1, 3), (4, 1, 3)),
    ((7,), (7,)),
])
@pytest.mark.parametrize('shape', ['circ', 'end'])
def test_init_plots_preserve_plot_columns_and_physical_labels(
        monkeypatch, tmp_path, channels, signal_channels, shape):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from shape_sensing.contracts import CalibrationSeed, InitializationResult

    # 绘图信号保留中心通道列，拟合信号则移除该列。
    plot_signals = (np.arange(3)[:, None] * 10 +
                    np.arange(len(channels))[None, :] + 1) * 1e-6
    signals = plot_signals[:, [channels.index(ch) for ch in signal_channels]]
    def tree(values):
        result = {name: {} for name in ('circ', 'S', 'end', 'end3', 'rot0', 'up', 'down')}
        result[shape] = {0: {1000: values if shape == 'circ' else {2000: values}}}
        return result
    design = DesignGeometry('plot-order', .0002, np.array([.1, .2, .3]), channels, {})
    dataset = CalibrationDataset(tree(signals), {0: {shape: [1000]}}, design,
                                 signal_channels, plot_data=tree(plot_signals))
    angles = {shape: {0: {1000: np.array([.1, .2, .3])}}}
    diagnostics = InitializationResult(CalibrationSeed(), {}, {}, angles, {}, {})
    service = VisualizationService(VisualizationInputs(dataset, diagnostics, segmentation=(3,)))
    saved = {}
    savefig = plt.savefig
    def capture(path, **kwargs):
        figure = plt.gcf()
        saved[Path(path).name] = [
            (ax.get_ylabel(), [(line.get_label(), line.get_xdata().copy(),
                               line.get_ydata().copy()) for line in ax.lines])
            for ax in figure.axes]
        kwargs['dpi'] = 30
        savefig(path, **kwargs)
    monkeypatch.setattr(plt, 'savefig', capture)
    monkeypatch.chdir(tmp_path)
    try:
        service.init_visualise(0)
        axes = saved[f'Strain_Data_{shape}.png']
        assert len(axes) == len(channels)
        for idx, (ylabel, lines) in enumerate(axes):
            assert ylabel == f'Ch {channels[idx]}'
            assert len(lines) == 1
            label, x, y = lines[0]
            assert label == ('1000' if shape == 'circ' else '1000-2000')
            np.testing.assert_array_equal(x, np.arange(3))
            expected = plot_signals[:, idx] / 1e-6 if shape == 'circ' else plot_signals[:, idx]
            np.testing.assert_array_equal(y, expected)
        assert 'All_Shapes_Unwrap_and_Fits.png' in saved
        assert len(list(tmp_path.rglob('*.png'))) == 2
        np.testing.assert_array_equal(dataset.plot_data[shape][0][1000]
                                      if shape == 'circ' else dataset.plot_data[shape][0][1000][2000],
                                      plot_signals)
    finally:
        plt.close('all')
