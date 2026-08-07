import matplotlib.pyplot as plt
import numpy as np
from MosaicAnalysis import Analysis
from pathlib import Path
import time
import matplotlib.image as mpimg
from matplotlib.patches import Circle

# -----------------------------
# Paths
# -----------------------------
global_path = str(Path.cwd())
parent_path = str(Path.cwd().parent) + '/'
saves_path = parent_path + "saves/"


# -----------------------------
# Global style settings
# -----------------------------
label_size = 18
tick_size = 12
legend_fontsize = 16
legend_size = 14
panel_label_size = 28
title_size = 16
line_width = 2.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans"],
    "font.size": tick_size,
    "axes.labelsize": label_size,
    "xtick.labelsize": tick_size,
    "ytick.labelsize": tick_size,
    "legend.fontsize": legend_fontsize,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
})

# -----------------------------
# Helper functions
# -----------------------------
def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", direction="out")


def add_panel_label(ax, letter, x=-0.12, y=1.08):
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        fontsize=panel_label_size,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False
    )


def add_panel_title(ax, title, color="black", pad=10):
    ax.set_title(
        title,
        fontsize=title_size,
        fontweight="bold",
        color=color,
        pad=pad
    )


def plot_model_schema(ax):
    schema_path = parent_path + "ruda_figures/" + "ruda_Fig_10A_v2.png"
    img = mpimg.imread(schema_path)
    ax.imshow(img)
    ax.axis("off")

    # Remove padding around the image axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0)

    # Keep the image centered in the available space
    ax.set_anchor("C")


# -----------------------------
# Data wrapper
# -----------------------------
class Saves:
    def __init__(self, save_names, saves_path=saves_path, x_range=15, step_size=0.01):
        self.input_noises = []
        self.center_surround_ratios = []
        self.n_neurons = []
        self.saves_path = saves_path
        self.dog_rfs = []
        self.save_names = save_names
        self.a = []
        self.zero_cross = []
        
        for save in save_names:
            an = Analysis(save, self.saves_path, None)
            print(save)
            an()

            dog_rf, x = an.DoG_median(x_range=x_range, step_size=step_size)
            

            self.input_noises.append(an.input_noise)
            self.n_neurons.append(an.n_neurons)
            self.center_surround_ratios.append(np.median(an.center_surround_ratio))
            self.x_rf = x
            self.dog_rfs.append(dog_rf)
            self.a.append(np.median(an.a))
            
            #---- Find zero crossing -------#
            half = len(dog_rf) // 2
            dog_second_half = dog_rf[half:]
            
            matches = np.where(dog_second_half < 0.01)[0]
            
            if len(matches) > 0:
                index = half + matches[0]   # index in original array
            else:
                index = None                # no value found
            
            self.zero_cross.append(x[index])
            
            time.sleep(0.1)

        change_noise = len(set(self.input_noises)) > 1
        change_neurons = len(set(self.n_neurons)) > 1

        assert change_noise ^ change_neurons
        if change_noise:
            self.changing_param = "input_noise"
        elif change_neurons:
            self.changing_param = "neurons"

        

    def plot_rfs(self, ax, model_type="nonlinear", legend_size = legend_size):
        n = len(self.save_names)

        if model_type == "nonlinear":
            colors = plt.cm.Blues(np.linspace(0.35, 0.95, n))
        elif model_type == "linear":
            colors = plt.cm.Oranges(np.linspace(0.35, 0.95, n))
        else:
            colors = plt.cm.Greys(np.linspace(0.35, 0.95, n))

        for i, dog_rf in enumerate(self.dog_rfs):
            if self.changing_param == "input_noise":
                label = r'$\sigma_{in}$ = ' + f"{self.input_noises[i]:.1f}"
            elif self.changing_param == "neurons":
                label = "n = " + str(self.n_neurons[i])
            else:
                label = str(i)

            ax.plot(
                self.x_rf,
                dog_rf,
                color=colors[i],
                linewidth=line_width,
                label=label
            )

        ax.set_xlabel("Distance from center")
        ax.set_ylabel("Weight")
        ax.legend(frameon=False, loc="upper right", fontsize = legend_size)
        style_axis(ax)
        ax.tick_params(axis='x', labelsize=legend_fontsize)
        ax.tick_params(axis='y', labelsize=legend_fontsize)
    
    def plot_center_surround_ratios(
        self,
        ax,
        label=None,
        color="tab:blue",
        marker="o",
        linestyle="-",
        legend_size=legend_size,
        ylims=(0, 1),
        equal_x_spacing=False,
        first_gap_scale=1.35
    ):
        if self.changing_param == "input_noise":
            x_values = np.asarray(self.input_noises, dtype=float)
            x_label = "Input noise"
        elif self.changing_param == "neurons":
            x_values = np.asarray(self.n_neurons, dtype=float)
            x_label = "Number of neurons"
        else:
            x_values = np.arange(len(self.save_names))
            x_label = "Run index"

        y_values = np.asarray(self.center_surround_ratios, dtype=float)
        # y_values = np.asarray(self.a, dtype=float)

        order = np.argsort(x_values)
        x_values = x_values[order]
        y_values = y_values[order]

        # For discrete input-noise conditions, use nearly equal spacing, but
        # enlarge the first gap so the wider "0.01" label does not crowd "0.1".
        if equal_x_spacing and self.changing_param == "input_noise":
            x_plot = np.arange(len(x_values), dtype=float)
            if len(x_plot) > 1:
                x_plot[1:] += first_gap_scale - 1.0
            ax.set_xticks(x_plot)
            ax.set_xticklabels([f"{x:g}" for x in x_values])
        else:
            x_plot = x_values
            ax.set_xticks(x_values)

        ax.plot(
            x_plot,
            y_values,
            color=color,
            linewidth=line_width,
            linestyle=linestyle,
            marker=marker,
            markersize=7,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label
        )

        ax.set_ylim(ylims[0], ylims[1])
        ax.set_xlabel(x_label)
        ax.set_ylabel("Surround-center ratio")
        ax.tick_params(axis="x", labelsize=legend_fontsize)
        ax.tick_params(axis="y", labelsize=legend_fontsize)

        if label is not None:
            ax.legend(frameon=False, loc="lower right", fontsize=legend_size)

        style_axis(ax)


    def plot_center_radius(
        self,
        ax,
        label=None,
        color="tab:blue",
        marker="o",
        linestyle="-",
        legend_size=legend_size,
        equal_x_spacing=False,
        first_gap_scale=1.35
    ):
        if self.changing_param == "input_noise":
            x_values = np.asarray(self.input_noises, dtype=float)
            x_label = "Input noise"
        elif self.changing_param == "neurons":
            x_values = np.asarray(self.n_neurons, dtype=float)
            x_label = "Number of neurons"
        else:
            x_values = np.arange(len(self.save_names))
            x_label = "Run index"

        # y_values = np.asarray(self.a, dtype=float)
        #y_values = np.asarray(self.zero_cross, dtype=float)
        
        y_values = np.asarray(self.zero_cross, dtype=float)

        order = np.argsort(x_values)
        x_values = x_values[order]
        y_values = y_values[order]

        # For discrete input-noise conditions, use nearly equal spacing, but
        # enlarge the first gap so the wider "0.01" label does not crowd "0.1".
        if equal_x_spacing and self.changing_param == "input_noise":
            x_plot = np.arange(len(x_values), dtype=float)
            if len(x_plot) > 1:
                x_plot[1:] += first_gap_scale - 1.0
            ax.set_xticks(x_plot)
            ax.set_xticklabels([f"{x:g}" for x in x_values])
        else:
            x_plot = x_values
            ax.set_xticks(x_values)

        ax.plot(
            x_plot,
            y_values,
            color=color,
            linewidth=line_width,
            linestyle=linestyle,
            marker=marker,
            markersize=7,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label
        )

        # ax.set_ylim(0, 1.0)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Center radius")
        ax.tick_params(axis="x", labelsize=legend_fontsize)
        ax.tick_params(axis="y", labelsize=legend_fontsize)

        if label is not None:
            ax.legend(frameon=False, loc="upper left", fontsize=legend_size)

        style_axis(ax)


def plot_cell_group(
    an,
    ax,
    color,
    marker='x',
    marker_size=18,
    alpha=0.2,
    edgecolor="none",
    xlim=None,
    ylim=None,
    value_text=None,
    value_loc=(0.98, 0.06),
    title = None
):
    """
    Plot one panel of circles centered at (x, y) with radius sigma.

    Parameters
    ----------
    ax : matplotlib axis
    x, y, sigma : 1D arrays of same length
    color : matplotlib color
    marker : None, 'o', or 'x'
        Small marker drawn at each circle center.
    marker_size : float
    alpha : float
    edgecolor : str
    xlim, ylim : tuple or None
    value_text : str or None
        Text shown in lower-right corner of panel.
    value_loc : tuple
        Axes coordinates for value_text.
    """
    
    select_neurons = np.array(an.pathway) == 'ON'
    x = np.asarray(an.kernel_centers[select_neurons,0], dtype = float)
    y = np.asarray(an.kernel_centers[select_neurons,1], dtype = float)
    an.half_crossings(0.35)
    sigma = an.half_cross
    print(x.shape, y.shape, sigma.shape)
    for xi, yi, si in zip(x, y, sigma):
        circ = Circle((xi, yi), si, facecolor=color, edgecolor=edgecolor, alpha=alpha, lw=0.8)
        ax.add_patch(circ)

    if marker is not None:
        if marker == "o":
            ax.scatter(x, y, s=marker_size, marker="o", facecolors="none",
                       edgecolors=color, linewidths=1.5, zorder=3)
        else:
            ax.scatter(x, y, s=marker_size, marker=marker, c=color,
                       linewidths=1.4, zorder=3)

    if xlim is None:
        pad = np.max(sigma) * 1.2 if len(sigma) else 1.0
        xlim = (x.min() - pad, x.max() + pad)
    if ylim is None:
        pad = np.max(sigma) * 1.2 if len(sigma) else 1.0
        ylim = (y.min() - pad, y.max() + pad)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("0.35")

    if value_text is not None:
        ax.text(
            value_loc[0], value_loc[1], str(value_text),
            transform=ax.transAxes,
            ha="right", va="center",
            fontsize=12,
            color="0.15"
        )
# -----------------------------
# Main figure
# -----------------------------
def fig10(save_path=None, a_label_y=100):
    fig, axes = plt.subplot_mosaic(
        [
            ["A", "A", "B"],
            ["C", "D", "E"],
            ["F", "G", "H"],
        ],
        figsize=(13, 14),
        constrained_layout=True
    )
    # -------------------------
    # A: model schema
    # -------------------------
    A_ax = axes["A"]
    plot_model_schema(A_ax)
    #add_panel_title(A_ax, "Model schematic", color="black", pad=8)
    
    #--------------------------
    # B: Mosaics for a nice model. 
    #--------------------------

    an_B = Analysis('260602-080910_ruda', saves_path, None) #Best at 1M
    
    an_B()
    B_ax = axes["B"]
    plot_cell_group(an_B, B_ax, color = 'tab:blue', title = "Example mosaic")
    add_panel_title(B_ax, "Example mosaic", color="black", pad=8)
    
    # -------------------------
    # C: non-linear model, varying input noise
    #Version with 1M steps, no jittering and centering. 0.0001 learning rate. 
    # -------------------------
    
    #Version with 100 neurons
    saveC_names = ['260602-011344_ruda', '260602-022255_ruda', '260602-033212_ruda', '260602-044121_ruda', '260602-055029_ruda', '260602-065959_ruda', '260602-080910_ruda', '260602-091816_ruda', '260602-102724_ruda']
    
    savesC_half = Saves(saveC_names[2::2], x_range = 7)
    savesC = Saves(saveC_names)
    print(savesC.n_neurons, 'see see see!')
    C_ax = axes["C"]
    savesC_half.plot_rfs(C_ax, model_type="nonlinear", legend_size = legend_size)
    add_panel_title(C_ax, "Non-linear model", color="tab:blue")
    C_ax.set_ylim(-0.02,0.09)
    # -------------------------
    # D: linear model, varying input noise
    #Version with 1M steps, no jittering and centering. 0.0001 learning rate. 
    # -------------------------
    #Version with 50 neurons
    saveD_names = ['260806-145331_ruda', '260806-160446_ruda', '260806-171545_ruda', '260806-182757_ruda', '260806-193910_ruda', '260806-205227_ruda', '260806-220557_ruda', '260806-231627_ruda', '260807-002658_ruda']
    
    savesD_half = Saves(saveD_names[2::2], x_range = 7)
    savesD = Saves(saveD_names)
    D_ax = axes["D"]
    savesD_half.plot_rfs(D_ax, model_type="linear", legend_size = legend_size)
    add_panel_title(D_ax, "Linear model", color="tab:orange")
    D_ax.set_ylim(-0.02,0.09)

    # -------------------------
    # E: center-surround ratio comparison
    # -------------------------
    E_ax = axes["E"]
    savesC.plot_center_surround_ratios(
        E_ax,
        label="Non-linear",
        color="tab:blue",
        marker="o",
        linestyle="-",
        legend_size=18,
        equal_x_spacing=True,
        first_gap_scale=1.35
    )
    savesD.plot_center_surround_ratios(
        E_ax,
        label="Linear",
        color="tab:orange",
        marker="s",
        linestyle="--",
        legend_size=18,
        equal_x_spacing=True,
        first_gap_scale=1.35
    )
    add_panel_title(E_ax, "Model comparison", color="black")
    
    E_ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    E_ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", ""])
    # Remove 1.0 so it does not overlap with the E panel label.

    # -------------------------
    # F: non-linear model, varying number of neurons
    # -------------------------
    saveF_names = ['260530-210245_ruda',
                   '260531-065740_ruda',
                   '260601-024935_ruda',
                   '260601-124901_ruda',
                   '260601-225825_ruda',
                   '260602-091816_ruda'] #100 neurons
    savesF = Saves(saveF_names, x_range = 20)
    F_ax = axes["F"]
    print(savesF.input_noises, 'hello!')
    savesF.plot_rfs(F_ax, model_type="nonlinear", legend_size = legend_size)
    add_panel_title(F_ax, "Non-linear model", color="tab:blue")

    # -------------------------
    # G: Interaction between input noise and number of neurons on surround-center ratios
    # H: Interaction between input noise and number of neurons on center radius 
    # -------------------------
    
    savesG_names = [['260530-132126_ruda', '260530-142656_ruda', '260530-153253_ruda', '260530-163853_ruda', '260530-174452_ruda', '260530-185054_ruda', '260530-195652_ruda', '260530-210245_ruda', '260530-220846_ruda'],
                    ['260530-231451_ruda', '260531-002055_ruda', '260531-012659_ruda', '260531-023303_ruda', '260531-033910_ruda', '260531-044523_ruda', '260531-055133_ruda', '260531-065740_ruda', '260531-080353_ruda'],
                    ['260531-091001_ruda', '260531-101609_ruda', '260531-112219_ruda', '260531-122831_ruda', '260531-133441_ruda', '260531-144052_ruda', '260531-154703_ruda', '260531-165318_ruda', '260531-175931_ruda'],
                    ['260531-190543_ruda', '260531-201201_ruda', '260531-211817_ruda', '260531-222431_ruda', '260531-233044_ruda', '260601-003658_ruda', '260601-014318_ruda', '260601-024935_ruda', '260601-035555_ruda'],
                    ['260601-050213_ruda', '260601-060858_ruda', '260601-071539_ruda', '260601-082218_ruda', '260601-092856_ruda', '260601-103538_ruda', '260601-114221_ruda', '260601-124901_ruda', '260601-135545_ruda'],
                    ['260601-150223_ruda', '260601-160953_ruda', '260601-171728_ruda', '260601-182459_ruda', '260601-193402_ruda', '260601-204252_ruda', '260601-215043_ruda', '260601-225825_ruda', '260602-000605_ruda'],
                    ['260602-011344_ruda', '260602-022255_ruda', '260602-033212_ruda', '260602-044121_ruda', '260602-055029_ruda', '260602-065959_ruda', '260602-080910_ruda', '260602-091816_ruda', '260602-102724_ruda']]

    
    
    G_ax = axes["G"]
    H_ax = axes["H"]
    neuron_conditions_num = len(savesG_names)
    
    colors_blues = plt.cm.Blues(np.linspace(0.35, 0.95, neuron_conditions_num))
    colors_greens = plt.cm.Greens(np.linspace(0.6, 0.95, neuron_conditions_num))
    for i, saveG_names, color_blue, color_green in zip(range(neuron_conditions_num), savesG_names, colors_blues, colors_greens):
        savesG = Saves(saveG_names)
        
        
        
        savesG.plot_center_surround_ratios(
            G_ax,
            label = "n = " + str(savesG.n_neurons[0]),
            color = color_blue,
            marker = "o",
            linestyle="--",
            legend_size=legend_size - 1,
            ylims=(0, 0.8),
            equal_x_spacing=True,
            first_gap_scale=1.35
        )
        
        color_H = color_blue
            
        savesG.plot_center_radius(
            H_ax,
            label = "n = " + str(savesG.n_neurons[0]),
            color = color_H,
            marker = "o",
            linestyle="--",
            legend_size=legend_size,
            equal_x_spacing=True,
            first_gap_scale=1.35
        )

    # Hide selected x-axis labels while keeping their tick marks and data points.
    hidden_x_labels = {"0.2", "0.4", "0.6", "0.8"}
    for ax in (E_ax, G_ax, H_ax):
        current_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        ax.set_xticklabels([
            "" if label in hidden_x_labels else label
            for label in current_labels
        ])
    
    G_ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    G_ax.set_yticklabels(["0.1", "0.2", "0.3", "0.4", "0.5","0.6", "0.7", ""])

    # ------------------------
    # Align y-axis labels
    #-------------------------
    fig.align_ylabels([axes["C"], axes["F"]])
    fig.align_ylabels([axes["D"], axes["G"]])
    fig.align_ylabels([axes["E"], axes["H"]])
    
    # -------------------------
    # Panel labels
    # -------------------------
    # Draw once so constrained_layout has finalized the axes positions
    fig.canvas.draw()
    
    # Put A at the same vertical height as C/D
    bbox_A = axes["A"].get_position()
    bbox_C = axes["C"].get_position()

    
    add_panel_label(A_ax, "A", x=-0.05, y=1.01)
    add_panel_label(B_ax, "B")
    add_panel_label(C_ax, "C")
    add_panel_label(D_ax, "D")
    add_panel_label(E_ax, "E")
    add_panel_label(F_ax, "F")
    add_panel_label(G_ax, "G")
    add_panel_label(H_ax, "H")

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, axes


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    fig, axes = fig10(
        save_path=parent_path + "ruda_figures/Fig10_publication.pdf",
        a_label_y=1.14
    )

    fig.savefig(
        parent_path + "ruda_figures/Fig10_publication.svg",
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()