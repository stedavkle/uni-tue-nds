from ipywidgets import IntSlider, FloatRangeSlider, Checkbox, Layout, interact
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from IPython.display import display, Image, clear_output

class Visualization:
    def __init__(self, data: dict):
        self.t = data["t"]
        self.dff = data["dff"]
        self.stim_table = data["stim_table"]
        self.roi_masks = data["roi_masks"]
        self.max_projection = data["max_projection"]
        self.running_speed = np.nan_to_num(data["running_speed"][0], nan=0)
        self.stim_epoch_table = data["stim_epoch_table"]
        self.stim_table["start"] = self.stim_table["start"].astype(int)
        self.stim_table["end"] = self.stim_table["end"].astype(int)
        self.fs = 1 / np.mean(np.diff(self.t))
        self.dt = 1 / self.fs
        self.directions = self.stim_table["orientation"].unique().tolist()
        self.directions.sort()

    ### input functions (slider, dropdown, etc.)
    def time_interval_slider(self, value: list[float, float]=None, update: bool=False) -> FloatRangeSlider:
        """
        Create a FloatRangeSlider for selecting a time interval.
        :param value: Initial value [a, b]
        :param update: Update plot only on releasing the slider
        """
        if value is None:
            value = [0, self.t[-1]]
        else:
            if value[0] < 0:
                value[0] = 0
            if value[1] > self.t[-1]:
                value[1] = self.t[-1]
        return FloatRangeSlider(
            value=value,  # Initial value [a, b]
            min=0.0,  # Minimum value A
            max=self.t[-1],  # Maximum value B
            step=self.dt,  # Step size
            description="Interval [s]:",  # Slider label
            continuous_update=update,  # Update plot only on releasing the slider
            layout=Layout(width="99%"),  # Adjust the layout width
        )

    def cell_index_slider(self, value: int=0, update: bool=False) -> IntSlider:
        """
        Create an IntSlider for selecting a cell index.
        :param value: Initial value
        :param update: Update plot only on releasing the slider
        """
        if value < 0:
            value = 0
        if value >= self.dff.shape[0]:
            value = self.dff.shape[0] - 1
        return IntSlider(
            value=value,  # Initial value
            min=0,  # Minimum value
            max=self.dff.shape[0] - 1,  # Maximum value based on number of cells
            step=1,  # Step size
            description="Cell Index:",  # Slider label
            continuous_update=update,  # Update plot only on releasing the slider
            layout=Layout(width="99%"),  # Adjust the layout width
        )

    def checkbox(self, value: bool=False, description: str="PLACEHOLDER", update: bool=False) -> Checkbox:
        """
        Create a Checkbox.
        :param value: Initial value
        :param update: Update plot only on releasing the slider
        """
        return Checkbox(
            value=value,  # Initial value
            description=description,  # Checkbox label
            indent=True, 
            layout=Layout(width="99%"),  # Adjust the layout width
        )

    ### plot update functions
    def update_raw_activity_traces_plot(
        self, 
        cellIdx: int, 
        sample_range: list[float, float], 
        show_epochs: bool=False
    ) -> None:
        """
        Plots the activity traces of a cell and the running speed of the mouse.

        Parameters
        ----------
        cellIdx : int
            Index of the cell to plot
        sample_range : list[float, float]
            Range of samples to plot [a, b] in seconds
        show_epochs : bool, optional
            Whether to show the epochs where the stimulus was shown, by default False
        """
        start = max(int(sample_range[0] * self.fs), 0)
        end = min(int(sample_range[1] * self.fs), len(self.t) - 1)
        fig, axs = plt.subplots(2, 1, figsize=(10, 4), height_ratios=[2, 1])
        axs[0].plot(self.t[start:end], self.dff[cellIdx, start:end], color="blue")
        axs[0].set_title(f"Raw Activity Trace of Cell {cellIdx}")
        axs[0].set_ylabel("Activity")
        axs[0].set_ylim([np.min(self.dff[cellIdx, :] - 0.2), np.max(self.dff[cellIdx, :]) + 0.2])
        axs[0].set_xlim([self.t[start] - 10, self.t[end] + 10])

        axs[1].plot(self.t[start:end], self.running_speed[start:end], color="orange")
        axs[1].set_title("Running Speed")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_ylabel("Speed [cm/s]")
        axs[1].set_ylim([-15, np.max(self.running_speed) + 5])
        axs[1].set_xlim([self.t[start] - 10, self.t[end] + 10])

        if show_epochs:
            epochs = self.get_epochs_stimulus_shown(start, end)
            legend = True
            for idx, row in epochs.iterrows():
                if legend:
                    axs[0].axvspan(self.t[row["start"]], self.t[row["end"]], color="gray", alpha=0.2, label="Stimulus Shown")
                    legend = False
                else:
                    axs[0].axvspan(self.t[row["start"]], self.t[row["end"]], color="gray", alpha=0.2)
                axs[1].axvspan(self.t[row["start"]], self.t[row["end"]], color="gray", alpha=0.2)
            axs[0].legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    def update_filter_traces_plot(
        self, 
        cellIdx: int, 
        sample_range: list[float, float], 
        dff_butter: np.array, 
        dff_wiener: np.array, 
        dff_both: np.array
    ) -> None:
        """
        Update the plot with the filtering stages of the cell activity traces.
        :param cellIdx: Cell index
        :param sample_range: Sample range [a, b]
        :param dff_butter: Butterworth filtered activity traces; provide fixed values
        :param dff_wiener: Wiener filtered activity traces; provide fixed values
        :param dff_both: Butterworth and Wiener filtered activity traces; provide fixed values 
        """
        start = max(int(sample_range[0] * self.fs), 0)
        end = min(int(sample_range[1] * self.fs), len(self.t) - 1)
        fig, axs = plt.subplots(4, 1, figsize=(10, 10), tight_layout=True)
        axs[0].plot(self.t[start:end], self.dff[cellIdx, start:end], "b")
        axs[0].set_title("Raw Activity Traces")
        #axs[0].set_ylabel("Activity")
        axs[0].set_xlim([self.t[start], self.t[end]])
        axs[0].set_ylim([np.min(self.dff[cellIdx, :]) - 0.5, np.max(self.dff[cellIdx, :]) + 0.5])

        axs[1].plot(self.t[start:end], dff_butter[cellIdx, start:end], "b")
        axs[1].set_title("Butterworth Filter")
        #axs[1].set_ylabel("Activity")
        axs[1].set_xlim([self.t[start], self.t[end]])
        axs[1].set_ylim(
            [np.min(dff_butter[cellIdx, :]) - 0.5, np.max(dff_butter[cellIdx, :]) + 0.5]
        )

        axs[2].plot(self.t[start:end], dff_wiener[cellIdx, start:end], "b")
        axs[2].set_title("Wiener Filter")
        #axs[2].set_ylabel("Activity")
        axs[2].set_xlim([self.t[start], self.t[end]])
        axs[2].set_ylim(
            [np.min(dff_wiener[cellIdx, :]) - 0.5, np.max(dff_wiener[cellIdx, :]) + 0.5]
        )

        axs[3].plot(self.t[start:end], dff_both[cellIdx, start:end], "b")
        axs[3].set_title("Butterworth and Wiener Filter")
        #axs[3].set_ylabel("Activity")
        axs[3].set_xlabel("Time [s]")
        axs[3].set_xlim([self.t[start], self.t[end]])
        axs[3].set_ylim(
            [np.min(dff_both[cellIdx, :]) - 0.5, np.max(dff_both[cellIdx, :]) + 0.5]
        )
        # y label
        fig.text(0.04, 0.5, "Activity", va="center", rotation="vertical")
        fig.suptitle(f"Filtering Stages of Cell {cellIdx} Activity Traces")
        plt.show()

    def update_inferred_spikes_plot(
        self,
        cellIdx: int,
        sample_range: list[float, float],
        show_oopsi: bool,
        show_oasis: bool,
        oopsi_spikes,
        oasis_spikes,
        input_trace: np.array,
    ) -> None:
        start = max(int(sample_range[0] * self.fs), 0)
        end = min(int(sample_range[1] * self.fs), len(self.t) - 1)
        fig, axs = plt.subplots(4, 1, figsize=(15, 10), height_ratios=[3, 3, 3, 1])
        min_ylim = []
        max_ylim = []
        min_subs = [0.1, 0.1, 0, 0]
        alpha = 0.5 if show_oopsi and show_oasis else 1.0
        for i, inferred_spikes in enumerate([oopsi_spikes, oasis_spikes]):
            if i == 0 and not show_oopsi:
                continue
            if i == 1 and not show_oasis:
                continue
            color = 'blue' if i == 0 else 'orange'
            label = 'OOPSI' if i == 0 else 'OASIS'
            if i == 0:
                axs[0].plot(self.t[start:end], input_trace[cellIdx, start:end], color=color, label=label, alpha=alpha)
            else:
                axs[0].plot(self.t[start:end], self.dff[cellIdx, start:end], color=color, label=label, alpha=alpha)
            axs[1].plot(self.t[start:end], inferred_spikes["deconv"][cellIdx][start:end], color=color, alpha=alpha)
            axs[2].plot(self.t[start:end], inferred_spikes["spikes"][cellIdx][start:end], color=color, alpha=alpha)
            axs[3].plot(self.t[start:end], inferred_spikes["binspikes"][cellIdx][start:end], color=color, alpha=alpha)
            min_ylim.append(np.min(input_trace[cellIdx, start:end]))
            max_ylim.append(np.max(input_trace[cellIdx, start:end]))
            min_ylim.append(np.min(inferred_spikes["deconv"][cellIdx, start:end]))
            max_ylim.append(np.max(inferred_spikes["deconv"][cellIdx, start:end]))
            min_ylim.append(np.min(inferred_spikes["spikes"][cellIdx, start:end]))
            max_ylim.append(np.max(inferred_spikes["spikes"][cellIdx, start:end]))
            min_ylim.append(np.min(inferred_spikes["binspikes"][cellIdx, start:end]))
            max_ylim.append(np.max(inferred_spikes["binspikes"][cellIdx, start:end]))
        if show_oasis or show_oopsi:
            if len(min_ylim) == 4 and len(max_ylim) == 4:
                for i in range(4):
                    axs[i].set_ylim([min_ylim[i] - min_subs[i], max_ylim[i] + 0.1])
            else:
                for i in range(4):
                    axs[i].set_ylim([min(min_ylim[i], min_ylim[i+4]) - min_subs[i], max(max_ylim[i], max_ylim[i+4]) + 0.1])

        axs[0].set_title("Input Signal")
        axs[0].set_ylabel("Activity")
        axs[1].set_title(f"Deconvolved Activity Trace")
        axs[1].set_ylabel("Activity")
        axs[2].set_title(f"Inferred Spikes")
        axs[2].set_ylabel("Spike Intensity")
    
        axs[3].set_title(f"Inferred Binary Spikes")
        axs[3].set_xlabel("Time [s]")

        for ax in axs:
            ax.set_xlim([self.t[start], self.t[end]])

        fig.suptitle(f"Cell {cellIdx} - Deconvolution and Spike Inference")
        if show_oopsi and show_oasis:
            axs[0].legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    ### static plot functions
    def color_roi(self, data: np.array, title: str, is_binary: bool = False) -> None:
        if is_binary:
            altered_roi_masks = np.zeros(self.roi_masks.shape, dtype="int")
            for c in range(len(data)):
                cell_roi = self.roi_masks[c, :, :].astype("int")
                if data[c] == 0:
                    cell_roi *= -1
                altered_roi_masks[c, :, :] = cell_roi
            mask = np.sum(altered_roi_masks, axis=0)
            mask[mask > 1] = 1
            mask[mask < -1] = -1
            mask = (mask + 1) / 2
            colors = ["blue", "lightgray", "red"]
            bounds = [0, 0.25, 0.75, 1]
            legend_patches = [
                mpatches.Patch(color="blue", label="Simple Cell"),
                mpatches.Patch(color="red", label="Complex Cell"),
            ]
        else:
            paired_cmap = plt.get_cmap("Paired")
            paired_idx = [0, 4, 6, 8, 1, 5, 7, 9]
            colors = ["lightgray"]
            for i in paired_idx:
                colors.append(paired_cmap(i))
            bounds = [-1.5, -0.5, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]
            mask = np.ones(self.roi_masks[0].shape, dtype="int") * -1
            for c in range(len(data)):
                cell_roi = self.roi_masks[c, :, :].astype("int")
                cell_roi *= data[c]
                # apply corresponding value of cell_roi to mask where cell_roi is not 0.
                mask[cell_roi != 0] = cell_roi[cell_roi != 0]
            legend_patches = [
                mpatches.Patch(color=paired_cmap(i), label=f"{self.directions[i]}°")
                for i in range(len(self.directions))
            ]

        cmap = ListedColormap(colors)
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(mask, cmap=cmap, norm=norm, interpolation="none")
        ax.set_title(title)
        ax.axis("off")
        plt.legend(handles=legend_patches, loc="upper center")
        plt.show()

    def filtered_running_speed(self, start, end, running_periods: np.array=None, running_smooth: np.array=None) -> None:
        """
        Plot the filtered running speed.
        :param start: Start time in seconds
        :param end: End time in seconds
        :param running_periods: Binary running periods
        :param running_smooth: Smoothed running speed
        """
        start = max(int(start * self.fs), 0)
        end = min(int(end * self.fs), len(self.t) - 1)
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        axs.plot(
            self.t[start:end],
            self.running_speed[start:end],
            color="blue",
            label="Measured Speed",
            alpha=0.5,
        )
        if running_periods is not None:
            axs.plot(
                self.t[start:end],
                running_periods[start:end] * 10,
                color="red",
                label="Binary Running Periods",
            )
        if running_smooth is not None:
            axs.plot(
                self.t[start:end],
                running_smooth[start:end],
                color="orange",
                label="Smoothed Running Speed",
            )
        axs.set_title("Running Speed Binary")
        axs.set_xlim([self.t[start], self.t[end]])
        axs.set_xlabel("Time [s]")
        axs.set_ylabel("Speed [cm/s]")
        plt.legend()
        plt.show()

    ### Helper functions
    def get_epochs_stimulus_shown(self, start: int, end: int) -> pd.DataFrame:
        if start < 0:
            start = 0
        if end > len(self.t)-1:
            end = len(self.t)-1
        grating_epochs = self.stim_epoch_table[self.stim_epoch_table["stimulus"] == "drifting_gratings"]
        # pop out all rows where either start and end are outside the interval, 
        # start is inside the interval or end is inside the interval
        grating_epochs = grating_epochs[~((grating_epochs["start"] > end) | (grating_epochs["end"] < start))]
        # check if it is empty and return
        if grating_epochs.empty:
            return grating_epochs
        # check if start is outside the first epoch interval
        if grating_epochs.iloc[0]["start"] < start:
            grating_epochs.iloc[0]["start"] = start
        # check if end is outside the last epoch interval
        if grating_epochs.iloc[-1]["end"] > end:
            grating_epochs.iloc[-1]["end"] = end
        return grating_epochs


# load data
def load_data(path="../data"):
    def array2df(d, key, cols):
        d[key] = pd.DataFrame(d[key], columns=cols)

    data = np.load(path + "/dff_data_dsi.npz", allow_pickle=True)
    data = dict(data)
    print("Data keys: ", data.keys())
    array2df(
        data,
        "stim_table",
        ["temporal_frequency", "orientation", "blank_sweep", "start", "end"],
    )
    array2df(data, "stim_epoch_table", ["stimulus", "start", "end"])

    return data

if __name__ == "__main__":
    data = load_data()
    vis = Visualization(data)
    #print(vis.get_epochs_stimulus_shown(780, 95000))
    vis.color_roi(np.array([1, 0, 1, 0, 1]), "Test", is_binary=True)
