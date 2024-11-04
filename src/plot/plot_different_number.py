import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def example_data():
    # The following data is from the Denver Aerosol Sources and Health study.
    # See doi:10.1016/j.atmosenv.2008.12.017
    #
    # The data are pollution source profile estimates for five modeled
    # pollution sources (e.g., cars, wood-burning, etc) that emit 7-9 chemical
    # species. The radar charts are experimented with here to see if we can
    # nicely visualize how the modeled source profiles change across four
    # scenarios:
    #  1) No gas-phase species present, just seven particulate counts on
    #     Sulfate
    #     Nitrate
    #     Elemental Carbon (EC)
    #     Organic Carbon fraction 1 (OC)
    #     Organic Carbon fraction 2 (OC2)
    #     Organic Carbon fraction 3 (OC3)
    #     Pyrolyzed Organic Carbon (OP)
    #  2)Inclusion of gas-phase specie carbon monoxide (CO)
    #  3)Inclusion of gas-phase specie ozone (O3).
    #  4)Inclusion of both gas-phase species is present...aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    data = [
        ['Accuracy', 'Precision', 'F1', 'Recall', 'AUPR', 'AUC'],
        ('30%.', [
            [0.7468354430379747,0.5897435897435898,0.5348837209302325,0.48936170212765956,0.6834448319118277,0.7764999041594787],
            [0.8227848101265823,0.7878787878787878,0.65,0.5531914893617021,0.7450302858769923,0.7887674908951504]]),
        ('40%', [
            [0.7468354430379747,0.5813953488372093,0.5555555555555555,0.5319148936170213,0.6987626560191706,0.7653824036802761],
            [0.8291139240506329,0.8125,0.6582278481012658,0.5531914893617021,0.7629237656441089,0.8085106382978723]]),
        ('50%', [
            [0.7658227848101266,0.631578947368421,0.5647058823529411,0.5106382978723404,0.6944298138615088,0.7458309373202991],
            [0.8227848101265823,0.8064516129032258,0.641025641025641,0.5319148936170213,0.7390880789038841,0.7598236534406747]]),
        ('60%', [
            [0.8164556962025317,0.8,0.6233766233766233,0.5106382978723404,0.7477229536980582,0.7874257235959364],
            [0.8227848101265823,0.88,0.6111111111111112,0.46808510638297873,0.7493705109728649,0.7782250335441825]]),
        ('70%', [
            [0.8291139240506329,0.7941176470588235,0.6666666666666666,0.574468085106383,0.76749888891674,0.8090856814261069],
            [0.8291139240506329,0.8125,0.6582278481012658,0.5531914893617021,0.7702271211623257,0.8085106382978724]]),
        ('80%', [
            [0.8354430379746836,0.7837837837837838,0.6904761904761905,0.6170212765957447,0.7512054260815996,0.7964347326049452],
            [0.8291139240506329,0.7777777777777778,0.674698795180723,0.5957446808510638,0.7380912598775836,0.7791834387579069]]),
        ('90%', [
            [0.8481012658227848,0.8108108108108109,0.7142857142857143,0.6382978723404256,0.7633401329791348,0.8238451217174622],
            [0.8354430379746836,0.8,0.6829268292682927,0.5957446808510638,0.754636578236096,0.8085106382978723]]),
        ('100%', [
            [0.879746835443038,0.8181818181818182,0.7912087912087913,0.7659574468085106,0.8210202073902289,0.8915085298064022],
            [0.8481012658227848,0.8108108108108109,0.7142857142857143,0.6382978723404256,0.7975206773326327,0.8711903392754456]])
    ]
    return data


if __name__ == '__main__':
    N = 6
    fontsize = 7
    theta = radar_factory(N, frame='polygon')

    data = example_data()
    spoke_labels = data.pop(0)

    fig, axs = plt.subplots(figsize=(9, 8), nrows=2, ncols=4,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.6, hspace=0.2, top=0.9, bottom=0.45)
    # fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    # colors = ['b', 'r', 'g', 'm', 'y']
    colors = ['#E76A6A', '#00BFFF', 'r', 'g', 'm', 'y'] 
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axs.flat, data):
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.2),
        # ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center', fontsize=fontsize)
        for d, color in zip(case_data, colors):
            # ax.plot(theta, d, color=color)
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
        ax.set_varlabels(spoke_labels)
        for label in ax.get_xticklabels():
            label.set_fontsize(fontsize)
        ax.set_rlim(0.4, 1.0)

    # add legend relative to top-left plot
    labels = ('LKCN', 'Dense')
    legend = axs[0, 0].legend(labels, loc=(5.3, 1.5),
    # legend = axs[0, 0].legend(labels, loc=(0.9, .95),
                              labelspacing=0.1, fontsize=fontsize)

    # fig.text(0.5, 0.965, 'KCN and the Dense in different number sample',
    #          horizontalalignment='center', color='black', weight='bold',
    #          size='large')

    plt.savefig('different_number.png') 

    plt.show()

