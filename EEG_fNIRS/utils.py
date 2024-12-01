import inspect
import matplotlib.pyplot as plt
from IPython import display as ipy_display
import os


class HyperParameters:  #@save
    """
    The base class of hyperparameters.
    """
    def save_hyperparameters(self, ignore=[]):
        """
        Save function arguments into class attributes.
        """
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)


class ProgressBoard(HyperParameters):  # @save
    """
    The board that plots data points in animation.
    """
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()
        if fig is None and axes is None:
            self.fig, self.axes = plt.subplots(figsize=figsize)
        else:
            self.fig, self.axes = fig, axes
        self.xlabel, self.ylabel = xlabel, ylabel
        self.xlim, self.ylim = xlim, ylim
        self.xscale, self.yscale = xscale, yscale
        self.ls, self.colors = ls, colors
        self.display = display
        self.data = {}

    def draw(self, x, y, label, every_n=1, draw_online=True, img_path=None):
        """Draw data points in real-time animation."""
        if label not in self.data:
            # Initialize a new line with specified line style and color
            line_style = self.ls[len(self.data) % len(self.ls)]
            color = self.colors[len(self.data) % len(self.colors)]
            self.data[label] = {'x': [], 'y': [], 'line': None}
            self.data[label]['line'], = self.axes.plot([], [], linestyle=line_style, color=color, label=label)

        self.data[label]['x'].append(x)
        self.data[label]['y'].append(y)
        
        if len(self.data[label]['x']) % every_n == 0:
            # Update line data
            self.data[label]['line'].set_data(self.data[label]['x'], self.data[label]['y'])

            # Set plot limits if provided
            if self.xlim:
                self.axes.set_xlim(self.xlim)
            else:
                self.axes.set_xlim(0, max(self.data[label]['x'])*1.1)
            
            if self.ylim:
                self.axes.set_ylim(self.ylim)
            else:
                self.axes.set_ylim(0, max(self.data[label]['y'])*1.1)

            # Set labels and scales
            if self.xlabel:
                self.axes.set_xlabel(self.xlabel)
            if self.ylabel:
                self.axes.set_ylabel(self.ylabel)
            self.axes.set_xscale(self.xscale)
            self.axes.set_yscale(self.yscale)

            # Display legend
            self.axes.legend()

            # Display the updated figure
            if self.display:
                ipy_display.clear_output(wait=True)
                ipy_display.display(self.fig)

            if img_path != None:
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                self.fig.savefig(img_path)

