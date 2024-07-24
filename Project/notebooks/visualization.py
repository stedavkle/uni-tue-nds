from ipywidgets import IntSlider, FloatRangeSlider, Checkbox, Layout, interact, Dropdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from IPython.display import display, Image, clear_output
import seaborn as sns
import warnings

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
        self.directions = self.stim_table["orientation"].dropna().unique().tolist()
        self.directions.sort()
        self.frequencies = self.stim_table["temporal_frequency"].dropna().unique().tolist()
        self.frequencies.sort()
        self.inferred_spikes = None
        self.keys_str = ["1", "2", "4", "8", "15", "-1"] # access keys for dataframe columns

    def set_inferred_spikes(self, inferred_spikes: dict) -> None:
        """ 
        Set the processed spikes data, used in functions:
            - update_stimulus_spike_times
        """
        self.inferred_spikes = inferred_spikes

    ################################################################################################
    ##### input functions (slider, dropdown, etc.)
    ################################################################################################

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

    def frequency_dropdown(self, value: float=0.0) -> Dropdown:
        """
        Create a Dropdown for selecting a frequency.
        """
        frequencies = [('All Frequencies', 0.0)]
        frequencies.extend([(f"{f} Hz", f) for f in self.frequencies])
        return Dropdown(
            options=frequencies,
            value=value,
            description="Temporal Frequency:",
            disabled=False,
            layout=Layout(width="99%"),
        )

    ################################################################################################
    ##### plot update functions
    ################################################################################################


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
        axs[0].set_xlim([self.t[start], self.t[end]])

        axs[1].plot(self.t[start:end], self.running_speed[start:end], color="orange")
        axs[1].set_title("Running Speed")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_ylabel("Speed [cm/s]")
        axs[1].set_ylim([-15, np.max(self.running_speed) + 5])
        axs[1].set_xlim([self.t[start], self.t[end]])

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

    def update_stimulus_spike_times(self, cellIdx: int, frequency: int, histogram: bool=False) -> None:
        """
        Plot with the spike times of a cell per orientation and trial.
        """
        if frequency == 0.0:
            stimulus_epochs = self.stim_table[self.stim_table["blank_sweep"] == 0.0].copy()
        else:
            stimulus_epochs = self.stim_table[
                self.stim_table["temporal_frequency"] == frequency
            ].copy()
        stimulus_epochs["diff"] = stimulus_epochs["end"] - stimulus_epochs["start"]
        stimulus_epochs = stimulus_epochs.sort_values(by="orientation")
        x_range = stimulus_epochs["diff"].max()
        #t_range = self.t[x_range] - self.t[0]
        y_range = stimulus_epochs["orientation"].value_counts().max() + 1
        stimulus_times = np.arange(x_range) * self.dt

        freq_string = (
            "Independent of Temporal Frequency"
            if frequency == 0.0
            else f"Temporal Frequency {frequency} Hz"
        )

        fig, axs = plt.subplots(1, 1, figsize=(10, 6))
        for i, ori in enumerate(self.directions):
            cell_spike_times = self.get_cell_stimulus_spikes_to_one_range(
                cellIdx, stimulus_epochs[stimulus_epochs["orientation"] == ori]
            )
            if histogram:
                spike_bins = np.zeros(x_range)
                for trial in cell_spike_times:
                    bin_idx = trial[:, 0].astype(int)
                    spike_bins[bin_idx] += 1
                axs.bar(
                    np.arange(x_range),
                    spike_bins,
                    width=1,
                    align="edge",
                    color='k',
                    bottom=i * y_range
                )
                axs.set_xlim(0, x_range-1)
                axs.set_xticks(np.linspace(0, len(stimulus_times), 9))
                axs.set_xticklabels(np.arange(0.0, 2.25, step=0.25))
                axs.set_title(
                    f"Spike Density of Cell {cellIdx} per Orientation - {freq_string}"
                )
            else:
                for t, trial in enumerate(cell_spike_times):
                    axs.scatter(
                        stimulus_times[trial[:, 0].astype(int)],
                        np.zeros_like(trial[:, 1]) + (i * y_range) + t + 0.5,
                        c="k",
                        s=2,
                        marker="|",
                    )
                # x-axis
                axs.set_xlim(-0.01, 2.0)
                axs.set_title(
                    f"Spike Times of Cell {cellIdx} per Orientation and Trial - {freq_string}"
                )

        axs.set_xlabel("Time [s]")

        # y-axis
        axs.set_ylim(0, len(self.directions) * y_range)
        axs.set_yticks(np.arange(len(self.directions) * y_range, step=y_range))
        axs.set_yticklabels(self.directions)
        axs.set_ylabel("Direction [°]")
        for tick_label in axs.get_yticklabels():
            tick_label.set_transform(
                tick_label.get_transform()
                + transforms.ScaledTranslation(0, 0.3, fig.dpi_scale_trans)
            )
        axs.grid(axis="y", alpha=0.5)
        plt.show()

    def update_spikes_per_frequency(self, cellIdx: int, sample_range: list[float, float]) -> None:
        start = max(int(sample_range[0] * self.fs), 0)
        end = min(int(sample_range[1] * self.fs), len(self.t) - 1)
        y_step = len(self.frequencies) + 1
        # create color array of y_step distinguishable colors
        colors = plt.cm.tab10(np.arange(0, y_step))
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))
        for i, ori in enumerate(orientations):
            for j, freq in enumerate(self.frequencies):
                cell_spike_times = self.get_spike_times_cell(
                    cellIdx, ori, freq
                )
                cell_spike_times_in_range = cell_spike_times[
                    (cell_spike_times[:, 0] >= start) & (cell_spike_times[:, 0] <= end)
                ]
                axs.scatter(
                    self.t[cell_spike_times_in_range[:, 0].astype(int)],
                    np.zeros_like(cell_spike_times_in_range[:, 1]) + (i * y_step) + j + 0.5,
                    color=colors[j],
                    s=2,
                    marker="|",
                    label=f"{freq} Hz" if i == 0 else "",
                )

        # x-axis
        axs.set_xlim(self.t[start], self.t[end])
        axs.set_xlabel("Time [s]")

        # y-axis
        axs.set_ylim(0, len(self.directions) * y_step)
        axs.set_yticks(np.arange(len(self.directions) * y_step, step=y_step))
        axs.set_yticklabels(self.directions)
        axs.set_ylabel("Direction [°]")
        for tick_label in axs.get_yticklabels():
            tick_label.set_transform(
                tick_label.get_transform()
                + transforms.ScaledTranslation(0, 0.18, fig.dpi_scale_trans)
            )

        axs.set_title(f"Spike Times of Cell {cellIdx} for Different Orientations")
        axs.grid(axis="y", alpha=0.5)
        # place legend outside on the right of the plot
        plt.legend(
            title=f" Temporal\n Frequency",
            title_fontsize="medium",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
        plt.show()

    def update_temporal_tuning_curve(self, cellIdx: int, temporal_tunings: np.array) -> None:
        x = np.arange(temporal_tunings.shape[2])

        temporal_tuning_mean = temporal_tunings[0, cellIdx, :]
        temporal_tuning_sd = temporal_tunings[1, cellIdx, :]

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.errorbar(
            x,
            temporal_tuning_mean,
            yerr=temporal_tuning_sd,
            fmt="o",
            capsize=5,
            color="blue",
        )
        # TODO x und y lim
        ax.set_title(f"Temporal Tuning of Cell {cellIdx}")
        ax.set_xlabel("Temporal Frequency [Hz]")
        ax.set_ylabel("Mean Value")  # TODO passt dieses label?
        ax.set_xticks(ticks=x, labels=self.frequencies)
        plt.show()

    def update_directional_tuning_curve(self, cellIdx: int, tuning_curve_fit: dict) -> None:
        fitted_curves = tuning_curve_fit[cellIdx]["fitted_curves"]
        mean_spike_counts = tuning_curve_fit[cellIdx]["mean_spike_counts"]
        std_counts = tuning_curve_fit[cellIdx]["std_spike_counts"]
        fig, ax = plt.subplots(figsize=(7, 5))
        # Plot average spike count per direction
        ax.plot(
            self.directions,
            mean_spike_counts,
            label="Mean Spike Count",
            color="orange",
            linewidth=2,
            alpha=0.7,
        )
        # Plot the standard deviation
        ax.fill_between(
            self.directions,
            mean_spike_counts - std_counts,
            mean_spike_counts + std_counts,
            alpha=0.1,
            color="orange",
        )
        # Plot fitted tuning curves for each temporal frequency
        [
            ax.plot(
                self.directions,
                fitted_curves[i, :],
                label=f"T {self.frequencies[i]} [Hz]",
                linestyle="--",
            )
            for i in range(len(self.frequencies))
        ]
        # get the two directions with the maximum at all fitted curves
        max_curve_idx_for_all_directions = np.argmax(fitted_curves, axis=0)
        max_direction_idx = np.argmax(np.max(fitted_curves, axis=0))
        ax.plot(
            self.directions[max_direction_idx],
            fitted_curves[
                max_curve_idx_for_all_directions[max_direction_idx], max_direction_idx
            ],
            "ro",
            label="Preferred Orientation",
        )
        # remove the maximum direction from the list and get the second maximum
        fitted_curves[max_curve_idx_for_all_directions[max_direction_idx]][max_direction_idx] = 0
        max_direction_idx = np.argmax(np.max(fitted_curves[max_curve_idx_for_all_directions], axis=0))
        ax.plot(
            self.directions[max_direction_idx],
            fitted_curves[
                max_curve_idx_for_all_directions[max_direction_idx], max_direction_idx
            ],
            "ro",
        )
        ax.set_xticks(np.arange(0, 316, 45))
        ax.set_xlim(-1, 316)
        ax.set_ylim(0, np.max(mean_spike_counts + std_counts) + 3)
        ax.set_xlabel("Direction [°]")
        ax.set_ylabel("Spike Count")
        ax.set_title("Tuning Curve Fit of Cell {}".format(cellIdx))
        ax.legend()
        plt.show()

    ################################################################################################
    ##### static plot functions
    ################################################################################################
    def color_roi(self, data: np.array, title: str, is_binary: bool = False) -> None:
        """
        Plot the colored ROI masks.

        Parameters
        ----------
        data: np.array -> (n_cells,) or (2, n_cells) if is_binary else (n_cells,)
            The data to color the ROI masks.
            If is_binary is True, the data should be binary values (0 or 1) for each cell.
                data[0, :] significant cells (0: not significant, 1: significant)
                data[1, :] single and complex cells (0: simple, 1: complex)
            If is_binary is False, the data should be the orientation of the drifting grating stimulus in degrees.
        title: str
            The title of the plot.
        is_binary: bool, optional
            If True, the data is binary values (0 or 1) for each cell, by default False.
        """
        tab_cmap = plt.get_cmap("tab20")
        def get_color(i):
            if i == 15:
                return "white"
            return tab_cmap(i)

        if is_binary:
            mask = np.zeros(self.roi_masks[0].shape, dtype="int")
            # just show single cells and complex cells
            if data.ndim == 1:
                for c in range(data.size):
                    cell_roi = self.roi_masks[c, :, :].astype("int")
                    # set roi cells to 1 or -1 depending on type
                    if data[c] == 0:
                        cell_roi *= -1
                    mask = np.where(mask == 0, cell_roi, mask)
                tab_idx = [0, 15, 6]
                colors = [get_color(i) for i in tab_idx]
                bounds = [-1.5, -0.5, 0.5, 1.5]
                legend_patches = [
                    mpatches.Patch(color=colors[0], label="Simple Cell"),
                    mpatches.Patch(color=colors[2], label="Complex Cell"),
                ]
            # show significance and type of cells
            else:
                for c in range(data.shape[1]):
                    cell_roi = self.roi_masks[c, :, :].astype("int")
                    # set roi cells to 1 or -1 depending on significance
                    if data[0, c] == 0:
                        cell_roi *= -1
                    # set roi cells to (-)2 if cell is complex
                    if data[1, c] == 1:
                        cell_roi *= 2
                    mask = np.where(mask == 0, cell_roi, mask)
                    
                tab_idx = [0, 1, 15, 7, 6]
                colors = [get_color(i) for i in tab_idx]
                bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
                legend_patches = [
                    mpatches.Patch(color=colors[0], label="Not Significant - Complex Cell"),
                    mpatches.Patch(color=colors[1], label="Not Significant - Simple Cell"),
                    mpatches.Patch(color=colors[3], label="Significant - Simple Cell"),
                    mpatches.Patch(color=colors[4], label="Significant - Complex Cell"),
                ]
        else:
            tab_idx = [15, 1, 3, 7, 9, 0, 2, 6, 8]
            colors = [get_color(i) for i in tab_idx]
            bounds = [-1.5, -0.5, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]
            mask = np.ones(self.roi_masks[0].shape, dtype="float") * -1
            for c in range(len(data)):
                cell_roi = self.roi_masks[c, :, :].astype("float")
                # set all zeros to -1
                cell_roi = np.where(cell_roi == 0, -1, cell_roi)
                # set all ones to data[c]
                cell_roi = np.where(cell_roi == 1, float(data[c]), cell_roi)
                # apply corresponding value of cell_roi to mask where cell_roi is not 0.
                mask = np.where(mask == -1.0, cell_roi, mask)
            legend_patches = [
                mpatches.Patch(color=tab_cmap(tab_idx[i + 1]), label=f"{self.directions[i]}°")
                for i in range(len(self.directions))
            ]

        cmap = ListedColormap(colors)
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(mask, cmap=cmap, norm=norm, interpolation="none")
        ax.set_title(title)
        # set x and yticks off
        ax.set_xticks([])
        ax.set_yticks([])
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

    def plot_tuning_orientation_test(
        self, cellIdx: int, qp_results: dict, qdistr: np.ndarray, direction: bool = False
    ) -> None:
        freq_strings = self.frequencies.copy()
        freq_strings.append(-1)
        fig, axs = plt.subplots(2, 3, figsize=(10, 6))

        for i, ax in enumerate(axs.flatten()):
            tf = freq_strings[i]
            q = qp_results[cellIdx][tf]["q"]

            ax.hist(
                qdistr[cellIdx, i, :],
                bins=30,
                color="skyblue",
                edgecolor="black",
                alpha=0.7,
            )
            ax.axvline(q, color="red", linestyle="--", label=f"Observed |q| = {q:.2f}")
            if tf == -1:
                ax.set_title("All Temporal Frequencies")
            else:
                ax.set_title(f"Temporal Frequency: {tf} Hz")
            ax.set_xlim(left=0)
            ax.legend(loc="upper right")
        for i in range(2):
            axs[i, 0].set_ylabel("Frequency")
        for i in range(3):
            axs[1, i].set_xlabel("|q| Values")
        oridir_string = "Direction" if direction else "Orientation"
        plt.suptitle(f"Permutation Test {oridir_string} Tuning of Neuron {cellIdx}")
        plt.tight_layout()
        plt.show()

    def plot_p_distribution(self, df_or: pd.DataFrame, df_dir: pd.DataFrame, hist=True) -> None:
        """
        Plot either the histogram or scatter plot of the p-values of the orientation and direction tuning. 
        
        Parameters
        ----------
        df_dir: pd.DataFrame
            DataFrame containing the p-values of the direction tuned neurons.
        df_or: pd.DataFrame
            DataFrame containing the p-values of the orientation tuned neurons.
        """
        data = [df_or, df_dir]
        columns = ["p_val_1", "p_val_2", "p_val_4", "p_val_8", "p_val_15", "p_val_-1"]
        freqs = ["1 Hz", "2 Hz", "4 Hz", "8 Hz", "15 Hz", "All Frequencies"]

        fig = plt.figure(figsize=(15, 5), constrained_layout=True)
        subfigs = fig.subfigures(2, 1)
        for i, subfig in enumerate(subfigs):
            subfig.suptitle(
                "Orientation Tuned Neurons" if i == 0 else "Direction Tuned Neurons"
            )
            df = data[i]
            axs = subfig.subplots(1, 6)
            for col, ax in enumerate(axs):
                if hist:
                    ax.hist(df[columns[col]], bins=30, color="blue", alpha=0.7)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 80)
                    if i == 1:
                        ax.set_xlabel("p-value")
                        ax.set_ylim(0, 60)
                    if col == 0:
                        ax.set_ylabel("Frequency")
                else:
                    ax.scatter(
                        np.arange(len(df[columns[col]])),
                        df[columns[col]],
                        color="blue",
                        alpha=0.7,
                        s=3,
                    )
                    if i == 1:
                        ax.set_xlabel("Cell")
                    if col == 0:
                        ax.set_ylabel("p-value")
                ax.set_title(f"{freqs[col]}")
        plt.suptitle("Distribution of p-values", fontsize=16)
        plt.show()

    def plot_num_significant_cells(self, comp_or: np.array, noncomp_or: np.array, comp_dir: np.array, noncomp_dir: np.array) -> None:
        """
        Plot the number of significant neurons for orientation and direction tuning per temporal frequency.

        Parameters
        ----------
        comp_or: np.array
            Number of complex neurons for orientation tuning.
        noncomp_or: np.array
            Number of simple neurons for orientation tuning.
        comp_dir: np.array
            Number of complex neurons for direction tuning.
        noncomp_dir: np.array
            Number of simple neurons for direction tuning.
        """
        keys_str = ["1", "2", "4", "8", "15", "All"]
        x_axis = np.arange(len(keys_str))
        data = [comp_or, comp_dir, noncomp_or, noncomp_dir]
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for i, ax in enumerate(axs):
            ax.scatter(
                x_axis - 0.05,
                data[i].values,
                color="blue",
                label="Complex Neurons",
            )
            ax.scatter(
                x_axis + 0.05,
                data[i + 2].values,
                color="lightblue",
                label="Simple Neurons",
            )
            ax.set_xticks(x_axis)
            ax.set_xticklabels(keys_str)
            ax.set_xlim(-0.3, len(keys_str) - 0.7)
            ax.set_ylim(0, 40)
            ax.set_xlabel("Temporal Frequency [Hz]")
            ax.set_ylabel("Sum of Significant Neurons")
            ax.set_title("Orientation Tuned" if i == 0 else "Direction Tuned")
            ax.legend(loc="lower left")

        plt.suptitle("Number of Significant Neurons for Orientation and Direction Tuning")
        plt.show()

    def plot_significance_heatmap(self, data: pd.DataFrame, p_thresh: float, direction=True):
        """ 
        Plot a heatmap of the significance of the tuned neurons.
        The heatmap shows the significance of the tuned neurons for each temporal frequency.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the significance values of the tuned neurons.
        p_thresh : float
            The significance threshold.
        direction : bool, optional
            If True, the heatmap is for direction tuned neurons, else for orientation tuned neurons, by default True.
        """
        colors = ["white", "black"]
        cmap = ListedColormap(colors)
        bounds = [-0.5, 0.5, 1.5]
        norm = BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(15, 2))
        ax = sns.heatmap(
            data[self.keys_str].T,
            cmap=cmap,
            norm=norm,
            cbar=False,
            xticklabels=True,
            yticklabels=True,
            linewidths=0.1,
            linecolor="gray",
            square=False,
        )

        xticks = ax.get_xticks()
        ax.set_xticks(xticks[::5])
        ax.set_xticklabels([f"{int(tick)}" for tick in xticks[::5]], rotation=0)
        ax.set_yticklabels(["1", "2", "4", "8", "15", "All"], rotation=0)

        legend_elements = [
            mpatches.Patch(facecolor="white", edgecolor="gray", label="0"),
            mpatches.Patch(facecolor="black", edgecolor="gray", label="1"),
        ]
        plt.legend(
            handles=legend_elements,
            title="Significance",
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            borderaxespad=0.0,
            frameon=False,
        )

        plt.ylabel("Temporal Frequency [Hz]")
        plt.xlabel("Neurons")
        ordir_string = "Direction" if direction else "Orientation"
        plt.title(
            f"Significant {ordir_string} Tuned Neurons (at Significance Level {p_thresh})"
        )
        plt.show()

    ################################################################################################
    ##### Helper functions
    ################################################################################################
    
    def get_epochs_stimulus_shown(self, start: int, end: int) -> pd.DataFrame:
        if start < 0:
            start = 0
        if end > len(self.t)-1:
            end = len(self.t)-1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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

    def get_cell_stimulus_spikes_to_one_range(
        self, cellIdx: int, stimulus_epochs: pd.DataFrame
    ) -> list[np.array]:
        """
        Get the spike times for a specific cell and stimulus mapped to a single range.

        Parameters
        ----------
        cellIdx: int
            The index of the cell.
        stimulus_epochs: pd.DataFrame
            Preprocessed stimulus epochs table. It assumes that the table is filtered in terms of orientation and frequency.

        Returns
        -------
        spike_times: list[np.array]
            List of spike times for each stimulus epoch (trial).
            np.array, (n_spikes, 2)
            spike_times[:, 0] contains the time index of the spikes
            spike_times[:, 1] contains the intensity of the spikes
        """
        if self.inferred_spikes is None:
            raise ValueError("Inferred spikes are not set. Use set_inferred_spikes method.")
        
        # time frame the spikes are mapped to
        time_range = max(stimulus_epochs["end"] - stimulus_epochs["start"])
        time_frame = np.arange(time_range)

        # iterate over the epochs the stimulus is shown
        spike_times = []
        for trial, epoch in enumerate(stimulus_epochs.iterrows()):
            start = int(epoch[1]["start"])
            end = int(epoch[1]["end"])
            cell_spikes = self.inferred_spikes["spikes"][cellIdx, start:end]
            idx_cell_spikes = np.where(cell_spikes > 0)[0]
            cell_spike_times = np.zeros((len(idx_cell_spikes), 2))
            cell_spike_times[:, 0] = idx_cell_spikes #time_frame[idx_cell_spikes]
            cell_spike_times[:, 1] = cell_spikes[idx_cell_spikes]
            spike_times.append(cell_spike_times)
        return spike_times

    def get_spike_times_cell(
        self,
        cell: int,
        orientation: float = 0.0,
        frequency: float = 1.0,
        blank_sweep=False
    ) -> np.array:
        """
        Get the spike times for a given orientation and frequency for a specific cell.

        Parameters
        ----------
        cell: int
            The index of the cell.
        stim_table: pd.DataFrame
            The table containing the start and end time index of each stimulus epoch.
        orientation: float
            The orientation of the drifting grating stimulus in degrees.
        frequency: float
            The frequency of the drifting grating stimulus in Hz.
        blank_sweep: bool
            If True, return the spike times for the blank sweep periods.

        Returns
        -------
        spike_times: np.array, (n_spikes, 2)
            spike_times[:, 0] contains the time index of the spikes
            spike_times[:, 1] contains the intensity of the spikes
        """
        # get epochs the stimulus is shown
        if not blank_sweep:
            stimulus_epochs = self.stim_table[
                (self.stim_table["orientation"] == orientation)
                & (self.stim_table["temporal_frequency"] == frequency)
            ]
        else:
            stimulus_epochs = self.stim_table[self.stim_table["blank_sweep"] == 1.0]

        # create binary mask for the entire experiment time with 1s for the stimulus epochs
        epochs_mask = np.zeros(self.inferred_spikes["spikes"].shape[1], dtype=int)
        for idx, epoch in stimulus_epochs.iterrows():
            epochs_mask[int(epoch["start"]) : int(epoch["end"])] = 1

        # match the epochs mask with the cells spike train by mutliplying them elementwise
        cell_spikes = self.inferred_spikes["spikes"][cell] * epochs_mask

        # store indices where cell_spikes is not zero using argwhere
        spike_indices = np.argwhere(cell_spikes > 0).flatten().astype(int)

        # remove zero entries from cell_spikes
        cell_spikes = cell_spikes[cell_spikes > 0]

        # check if cell_spikes has the same length as spike_indices
        assert len(cell_spikes) == len(spike_indices)

        # create the spike times array
        spike_times = np.zeros((len(cell_spikes), 2))
        spike_times[:, 0] = spike_indices
        spike_times[:, 1] = cell_spikes
        return spike_times


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
