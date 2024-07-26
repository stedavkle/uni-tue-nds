import oopsi
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr, stats
from scipy.optimize import curve_fit
import statsmodels.stats.multitest as smm
from scipy import signal, ndimage
import scipy.special as sp
from tqdm import tqdm
import pickle
from oasis.functions import deconvolve
from oasis.oasis_methods import oasisAR1, oasisAR2

### helper functions for accessing data

def get_stimulus(t_idx: int, data: dict) -> np.array:
    """
    Provides the stimulus frame shown at time index.
    Parameters
    ----------
    t_idx: int 
        time index (of t from the data)
    data: (dict)
        dictionary containing at least 
            "stim" (np.array): the shown stimulus frames with shape (n_frames, height, width)
            "stim_table" (pandas dataframe): the table containing the start and end time of each stimulus frame

    Returns
    -------
    stimulus: np.array, (height, width)
        The stimulus frame shown at time index. 
        Returns None if there is no stimulus at time t.
    """

    stim_table = data["stim_table"]  # pandas dataframe
    stim = data["stim"]

    # get the index of the stimulus at time t
    idx = stim_table[(stim_table["start"] <= t_idx) & (stim_table["end"] > t_idx)]["frame"]

    stimulus = stim[idx]

    # if there is no stimulus at time t
    if len(stimulus) == 0:
        return None
    return stimulus

def get_spike_times(spikes: np.array, stim_table: pd.DataFrame, orientation: float = 0.0, frequency: float = 1.0, blank_sweep=False) -> np.array:
    """
    Get the spike times for a given orientation and frequency for all cells.

    Parameters
    ----------
    spikes: np.array, (n_cells, n_samples)
        The binary spike times for all cells.
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
    spike_times: np.array, (n_cells, n_samples)
        The spike times for all cells for the given orientation and frequency as a binary array.
    """
    # get epochs the stimulus is shown
    if not blank_sweep:
        stimulus_epochs = stim_table[(stim_table["orientation"] == orientation) & (stim_table["temporal_frequency"] == frequency)]
    else:
        stimulus_epochs = stim_table[stim_table["blank_sweep"] == 1.0]

    # get the spike times for the stimulus epochs
    spike_times = np.zeros(spikes.shape)
    for idx, epoch in stimulus_epochs.iterrows():
        spike_times[:, int(epoch["start"]):int(epoch["end"])] = spikes[:, int(epoch["start"]):int(epoch["end"])]
    return spike_times

def get_spike_times_cell(
    spikes: np.array,
    cell: int,
    stim_table: pd.DataFrame,
    orientation: float = 0.0,
    frequency: float = 1.0,
    blank_sweep=False,
) -> np.array:
    """
    Get the spike times for a given orientation and frequency for a specific cell.

    Parameters
    ----------
    spikes: np.array, (n_cells, n_samples)
        The spike counts for all cells.
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
        stimulus_epochs = stim_table[
            (stim_table["orientation"] == orientation)
            & (stim_table["temporal_frequency"] == frequency)
        ]
    else:
        stimulus_epochs = stim_table[stim_table["blank_sweep"] == 1.0]

    # create binary mask for the entire experiment time with 1s for the stimulus epochs
    epochs_mask = np.zeros(spikes.shape[1], dtype=int)
    for idx, epoch in stimulus_epochs.iterrows():
        epochs_mask[int(epoch["start"]) : int(epoch["end"])] = 1

    # match the epochs mask with the cells spike train by mutliplying them elementwise
    cell_spikes = spikes[cell] * epochs_mask
    
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



def load_inferred_spikes(file_path: str) -> dict:
    """
    Load the inferred spikes from a file.
    
    Parameters
    ----------
    file_path: str
        The file path to the pickled file containing the inferred spikes.

    Returns
    -------
    spikes: dict
        {
            "spikes": np.array, (n_cells, n_samples)    # inferred spikes by oopsi
            "deconv": np.array, (n_cells, n_samples)    # deconvolved signal
            "binspikes": np.array, (n_cells, n_samples) # thresholded inferred spikes with {0, 1}
        }
        Dictionary containing the inferred spikes for each cell.
    """
    spikes = {}

    with open(file_path, "rb") as f:
        spikes = pickle.load(f)

    # convert the lists to numpy arrays
    spikes["spikes"] = np.array(spikes["spikes"])
    spikes["deconv"] = np.array(spikes["deconv"])
    spikes["binspikes"] = np.array(spikes["binspikes"])

    return spikes


### functions for processing
def window_rms(a, window_size):
    """
    Apply RMS convolution to a window of the (running speed) signal
    
    Parameters
    ----------
    x: np.array, (n_samples )
        Each column in x is one cell.
    window: float
        size of the rms window for convolution

    Returns
    -------
    y: np.ndarray, (n_samples, n_cells)
        The smoothed x.
    """
    a2 = np.power(a, 2)  # signal quadrieren
    window = np.ones(window_size) / float(
        window_size
    )  # Gewichte für die Convolution festlegen
    return np.sqrt(np.convolve(a2, window, "same"))

def butter_filter_signal(
    x: np.array, fs: float, low: float, high: float, order: int = 3
) -> np.array:
    """
    Filter raw signal x using a Butterworth filter.
    
    Parameters
    ----------
    x: np.array, (n_samples, n_cells)
        Each column in x is one cell.
    fs: float
        Sampling frequency.
    low, high: float, float
        Passband in Hz for the butterworth filter.
    order: int
        The order of the Butterworth filter. Default 3

    Returns
    -------
    y: np.array, (n_samples, n_cells)
    The filtered x. The filter delay is compensated in the output y.
    """

    y = np.apply_along_axis(
        lambda col: signal.sosfiltfilt( # apply the filter to all columns
            signal.butter(              # apply the filter to a column
                N=order,
                Wn=[low, high],         # frequency thresholds (normalized)
                btype="band",           # filter type
                analog=False,
                output="sos",           # second-order sections
            ),
            col,
        ),
        axis=1,
        arr=x,
    )

    return y

def wiener_filter_signal(x: np.array, window: float) -> np.array:
    """
    Apply Wiener Filter to raw signal x.
    
    Parameters
    ----------
    x: np.array, (n_samples, n_cells)
        Each column in x is one cell.
    window: float
        size of the wiener filter

    Returns
    -------
    y: np.ndarray, (n_samples, n_cells)
        The filtered x. There is no filter delay as to my knowledge # TODO clarify!
    """

    y = np.apply_along_axis(
        lambda col: signal.wiener(  # apply the filter to a column
            col,
            mysize=window,          # window of the wiener filter
        ),
        axis=1,
        arr=x,
    )

    return y

def oopsi_inference(dff: np.array, dt: float, thresh: int = 0.035, to_file: bool = False) -> dict:
    """
    Perform spike inference using the OOPSi algorithm.
    
    Parameters
    ----------
    dff: np.array, (n_cells, n_samples)
        Filtered signal from cells.
    dt: float
        Time step between samples.
    thresh: float
        Threshold for spike inference. Default 0.035.

    Returns
    -------
    spikes: dict
        {
            "spikes": np.array, (n_cells, n_samples)    # inferred spikes by oopsi
            "deconv": np.array, (n_cells, n_samples)    # deconvolved signal
            "binspikes": np.array, (n_cells, n_samples) # thresholded inferred spikes with {0, 1}
        }
        Dictionary containing the inferred spikes for each cell.
    """

    spikes = {
        "spikes": np.zeros(dff.shape),
        "deconv": np.zeros(dff.shape),
        "binspikes": np.zeros(dff.shape)
    }
    for idxCell in tqdm(range(dff.shape[0])):
        oopsi_inf = oopsi.fast(dff[idxCell], dt=dt)
        spike_train = [1 if value > thresh else 0 for value in oopsi_inf[0]]

        # store the results
        spikes["spikes"][idxCell, :] = oopsi_inf[0]
        spikes["deconv"][idxCell, :] = oopsi_inf[1]
        spikes["binspikes"][idxCell, :] = spike_train

    if to_file:
        with open("../data/inference_oopsi.pkl", "wb") as f:
            pickle.dump(spikes, f)

    return spikes

def oasis_inference(dff: np.array, optimize_g: int = 3, penalty: int = 0, to_file: bool = False) -> dict:
    """
    Perform spike inference using the OASIS algorithm.
    
    Parameters
    ----------
    dff: np.array, (n_cells, n_samples)
        Filtered signal from cells.
    optimize_g: int
        Number of large, isolated events to consider for optimizing g.
        If optimize_g=0 the estimated g is not further optimized.
    penalty: int
        Sparsity penalty. 1: min |dff|_1  0: min |dff|_0
    to_file: bool
        If True, save the inferred spikes to a file.

    Returns
    -------
    spikes: dict
        {
            "spikes": np.array, (n_cells, n_samples)    # inferred spikes by oopsi
            "deconv": np.array, (n_cells, n_samples)    # deconvolved signal
            "binspikes": np.array, (n_cells, n_samples) # thresholded inferred spikes with {0, 1}
        }
        Dictionary containing the inferred spikes for each cell.
    """
    inferred_spikes = {
        "spikes": np.zeros(dff.shape),
        "deconv": np.zeros(dff.shape),
        "binspikes": np.zeros(dff.shape)
    }
    for idxCell in tqdm(range(dff.shape[0])):
        deconv_signal, spikes, b, g, lam = deconvolve(dff[idxCell], penalty=1, optimize_g=optimize_g)
        binary_spikes = [1 if value > 0 else 0 for value in spikes]
        
        inferred_spikes["spikes"][idxCell, :] = spikes
        inferred_spikes["deconv"][idxCell, :] = deconv_signal
        inferred_spikes["binspikes"][idxCell, :] = binary_spikes
    
    if to_file:
        with open("../data/inference_oasis.pkl", "wb") as f:
            pickle.dump(inferred_spikes, f)

    return inferred_spikes

# TODO add the other inference methods

### functions for running speed processing
def filter_running_speed(
    running_speed: np.array, s_kernel: int = 34, c_kernel: int = 100, o_kernel: int = 34, noise_thresh: float = 4.0
) -> np.array:
    """
    Extract running periods from the running speed [cm/s] signal using uniform filtering.
    
    Parameters
    ----------
    running_speed: np.array, (n_samples,)
        The running speed signal.
    s_kernel: int
        The size of the moving average window. Default 34, which corresponds to ~1s.
    c_kernel: int
        Max gap size in running periods to be filled. Default 100, which corresponds to ~3s.
    o_kernel: int
        Max peak width in running speed data to be not considered as running period. Default 34, which corresponds to ~1s.
    noise_thresh: float
        The threshold to remove noise from the running speed signal. Default 4.0

    Returns
    -------
    running_periods: np.array, (n_samples,)
        A binary array indicating the running periods.
    """

    # smooth the running speed signal
    running_speed_smooth = ndimage.uniform_filter1d(running_speed, s_kernel)

    # make binary array for running periods
    running_speed_binary = (running_speed_smooth > noise_thresh).astype(int)

    # fill gaps in running periods
    running_periods = ndimage.morphology.binary_closing(running_speed_binary, structure=np.ones(c_kernel))

    # remove peaks in running speed data
    running_periods = ndimage.morphology.binary_opening(running_periods, structure=np.ones(o_kernel))

    return running_periods

def get_running_periods_table(running_periods: np.array) -> pd.DataFrame:
    """
    Get the running periods table from the binary array of running periods.
    
    Parameters
    ----------
    running_periods: np.array, (n_samples,)
        A binary array indicating the running periods.

    Returns
    -------
    running_periods_table: pd.DataFrame
        A table containing the start and end time index of each running period.
        Note: The end index points on the last 1 of the running period.
        Note: The stimulus column is set to "running" and has no meaning. 
            It is added for compatibility with epochs_in_range().
    """
    
    # extract all start and end indices of running periods
    diff = np.diff(running_periods.astype(int))
    start_indices = np.where(diff == 1)[0] + 1
    stop_indices = np.where(diff == -1)[0]

    # Handle edge cases
    if running_periods[0] == 1:
        start_indices = np.insert(start_indices, 0, 0)
    if running_periods[-1] == 1:
        stop_indices = np.append(stop_indices, len(running_periods) - 1)

    # write to table with columns "start" and "end"
    running_periods_table = pd.DataFrame({"stimulus": ["running"] * len(start_indices), "start": start_indices, "end": stop_indices})
    return running_periods_table

# TODO die funktion ist glaub überflüssig
# def analyze_spike_running_correlation(spiketrains, running_period):
#     n_cells, n_measurements = spiketrains.shape

#     # Ensure running_period is boolean
#     running_period = running_period.astype(bool)

#     # Initialize lists to store results
#     p_values = []

#     # Iterate over each cell to perform the statistical test
#     for cell in range(n_cells):
#         # Get spikes for running and non-running periods
#         spikes_running = spiketrains[cell, running_period]
#         spikes_non_running = spiketrains[cell, ~running_period]

#         # Calculate the average spike rate during running and non-running periods
#         rate_running = np.mean(spikes_running)
#         rate_non_running = np.mean(spikes_non_running)

#         # Perform unpaired statistical test
#         if np.var(spikes_running) == 0 or np.var(spikes_non_running) == 0:
#             # If there's no variation, set p-value to 1.0
#             p_value = 1.0
#         else:
#             # Use independent t-testa
#             t_stat, p_value = ttest_ind(
#                 spikes_running, spikes_non_running, equal_var=False
#             )

#         p_values.append(p_value)

#     # Apply multiple comparisons correction (Bonferroni)
#     corrected_p_values = smm.multipletests(p_values, method="bonferroni")[1]

#     # Identify significant cells
#     significant_cells = np.where(corrected_p_values < 0.05)[0]

#     return significant_cells, corrected_p_values

def get_running_correlation_max(roi_masks, inferred_spikes, running_speed):
    """
    Calculate the correlation between the running speed and the activity of each cell.
    The correlation is then applied to the roi masks of the cells.
    The top 10 absolute correlation values are returned.

    Parameters
    ----------
    roi_masks: np.array
        The roi masks of the cells.
    inferred_spikes: dict
        The inferred spikes of the cells.
    running_speed: np.array
        The running speed of the mouse.

    Returns
    -------
    roi_masks_corr_sum: np.array
        The sum of the roi masks with the correlation values applied.
    cell_corr: np.array
        The correlation values for each cell.
    top_10: np.array
        The top 10 absolute correlation values.
    """
    roi_masks_copy = roi_masks.copy()
    roi_masks_corr = np.zeros_like(roi_masks_copy, dtype=np.float64)
    cell_corr = np.zeros(roi_masks_copy.shape[0])
    for cell in range(roi_masks_copy.shape[0]):
        corr, p = pearsonr(inferred_spikes["binspikes"][cell], running_speed)
        roi_masks_corr[cell, :, :] = np.where(
            roi_masks_copy[cell, :, :].astype(np.float64) < 1,
            roi_masks_copy[cell, :, :],
            corr,
        )
        cell_corr[cell] = corr

    roi_masks_sum = np.sum(roi_masks_copy, axis=0)
    roi_masks_sum = roi_masks_sum / np.max(roi_masks_sum)

    roi_masks_corr_sum = np.sum(roi_masks_corr, axis=0)
    roi_masks_corr_sum = roi_masks_corr_sum / np.max(roi_masks_corr_sum)

    # get the top 10 correlation values
    top_10 = np.argsort(np.abs(cell_corr))[-10:]
    return roi_masks_corr_sum, cell_corr, top_10

### Visualization Helper
def get_epochs_in_range(stim_epoch_table: pd.DataFrame, start: int = 0, end: int = 105967) -> pd.DataFrame:
    """
    Get the epochs which don't show the drifting_gratings stimulus in a given range.
    
    Parameters
    ----------
    stim_epoch_table: pd.DataFrame
        A table containing the start and end time index of each stimulus epoch.
    start: int
        The start of the range of interest. Default 0.
    end: int
        The end of the range of interest. Default 105967 (The last index of t).

    Returns
    -------
    epochs_in_range: pd.DataFrame
        A table containing the epochs that are within the range [start, end].
        Note: Epochs that are partially in the range are included and cut to start or end.
    """
    # extract all epochs that don't show the locally sparse noise stimulus
    # mark the epochs where the locally sparse noise stimulus is not shown
    other_stimuli_epochs = stim_epoch_table[
        stim_epoch_table["stimulus"] != "drifting_gratings"
    ].copy()

    # if start is smaller than the first epoch start, add interval from start to epoch start
    if start < stim_epoch_table["start"].min():
        other_stimuli_epochs = pd.concat([other_stimuli_epochs, pd.DataFrame([
            {"stimulus": "no_measurement", 
            "start": start, 
            "end": stim_epoch_table.iloc[0]["start"] - 1}])],
            ignore_index=True,
        )

    # if end is larger than the last epoch end, add interval from last epoch end to end
    if end > stim_epoch_table["end"].max():
        other_stimuli_epochs = pd.concat([other_stimuli_epochs, pd.DataFrame([
            {"stimulus": "no_measurement", 
            "start": stim_epoch_table.iloc[-1]["end"] + 1, 
            "end": end}])],
            ignore_index=True,
        )

    # find epochs that overlap "start" and set their start to "start"
    other_stimuli_epochs.loc[
        (other_stimuli_epochs["start"] < start) & (other_stimuli_epochs["end"] >= start), "start"
    ] = start

    # find epochs that overlap "end" and set their end to "end"
    other_stimuli_epochs.loc[
        (other_stimuli_epochs["start"] <= end) & (other_stimuli_epochs["end"] > end), "end"
    ] = end

    # filter epochs that are within the range [start, end]
    other_stimuli_epochs = other_stimuli_epochs[
        (other_stimuli_epochs["start"] >= start) & (other_stimuli_epochs["end"] <= end)
    ]

    return other_stimuli_epochs

### ML Estimation of RF

def negloglike_lnp(
    w: np.array, c: np.array, s: np.array, dt: float = 0.1, R: float = 50
) -> float:
    """Implements the negative (!) log-likelihood of the LNP model

    Parameters
    ----------

    w: np.array, (Dx * Dy, )
      current receptive field

    c: np.array, (nT, )
      spike counts

    s: np.array, (Dx * Dy, nT)
      stimulus matrix


    Returns
    -------

    f: float
      function value of the negative log likelihood at w

    """

    # ------------------------------------------------
    # Implement the negative log-likelihood of the LNP
    # ------------------------------------------------

    # compute dot product of w and s
    ws = np.dot(w, s)
    # compute the mean rate in time bins
    r = np.exp(ws) * dt * R

    # compute the negative log likelihood
    f = -np.sum(c * np.log(r) - np.log(sp.factorial(c)) - r)

    return f


def deriv_negloglike_lnp(
    w: np.array, c: np.array, s: np.array, dt: float = 0.1, R: float = 50
) -> np.array:
    """Implements the gradient of the negative log-likelihood of the LNP model

    Parameters
    ----------

    see negloglike_lnp

    Returns
    -------

    df: np.array, (Dx * Dy, )
      gradient of the negative log likelihood with respect to w

    """

    # --------------------------------------------------------------
    # Implement the gradient with respect to the receptive field `w`
    # --------------------------------------------------------------

    # compute dot product of w and s
    ws = np.dot(w, s)
    # compute the mean rate in time bins
    r = np.exp(ws) * dt * R
    # compute the gradient
    df = np.dot(s, (r - c))  # switched r and c to match negative log likelihood

    return df

def vonMises(theta: np.ndarray, alpha: float, kappa: float, mu: float, phi: float) -> np.ndarray:
    """Evaluate the parametric von Mises tuning curve with parameters p at locations theta.

    Parameters
    ----------

    theta: np.array, shape=(N, )
        Locations. The input unit is degree.

    alpha, kappa, mu, phi : float
        Function parameters

    Return
    ------
    f: np.array, shape=(N, )
        Tuning curve.
    """

    # Convert theta to radians
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)

    f = np.exp(
        alpha
        + kappa * (np.cos(2 * (theta_rad - phi_rad)) - 1)
        + mu * (np.cos(theta_rad - phi_rad) - 1)
    )
    return f

def fitTemporalTuningCurve(
    inferred_spikes: np.ndarray,
    stim_table=pd.DataFrame(),
) -> dict:
    """Fit a von Mises tuning curve to the spike counts in count with direction dir using a least-squares fit.
    Parameters
    ----------
    inferred_spikes: np.array, shape=(n_neurons, n_timepoints)

    stim_table: pd.DataFrame
        DataFrame containing the stimulus information

    neuron: int, default=0

    Return
    ------
    fitted_curves: np.array, shape=(n_temporal_frequencies, n_directions)
        Fitted tuning curves for each temporal frequency

    mean_spike_counts: np.array, shape=(n_directions,)
        Mean spike counts per direction

    std_spike_counts: np.array, shape=(n_directions,)
        Standard deviation of spike counts per direction
    """
    result = {}

    starts = stim_table["start"].astype(int).to_numpy()
    ends = stim_table["end"].astype(int).to_numpy()

    dirs = stim_table["orientation"].dropna()
    unique_directions = np.unique(dirs).astype(int)
    unique_directions.sort()

    temporal_frequencies = stim_table["temporal_frequency"].dropna()
    unique_temporal_frequencies = np.unique(temporal_frequencies).astype(int)
    unique_temporal_frequencies.sort()

    for neuron in tqdm(range(inferred_spikes["binspikes"].shape[0])):
        # Vectorized calculation of spike counts
        spike_counts = np.array(
            [
                np.sum(inferred_spikes["binspikes"][neuron, start:end])
                for start, end in zip(starts, ends)
            ]
        )
        spike_counts = spike_counts[stim_table["orientation"].dropna().index]
        initial_guess = [np.mean(spike_counts), 1, 1, np.median(dirs)]

        curves_temporal_frequencies = np.zeros(
            (len(unique_temporal_frequencies), len(initial_guess))
        )
        fitted_curves = np.zeros(
            (len(unique_temporal_frequencies), len(unique_directions))
        )
        mean_spike_counts = np.array(
            [np.mean(spike_counts[dirs == d]) for d in unique_directions]
        )
        std_spike_counts = np.array(
            [np.std(spike_counts[dirs == d]) for d in unique_directions]
        )

        for i, temporal_frequency in enumerate(unique_temporal_frequencies):
            # Perform the non-linear least squares fit
            bounds = (
                [-np.inf, -np.inf, -np.inf, -np.inf],
                [np.inf, np.inf, np.inf, 360],
            )
            popt, _ = curve_fit(
                vonMises,
                dirs[temporal_frequencies == temporal_frequency],
                spike_counts[temporal_frequencies == temporal_frequency],
                p0=initial_guess,
                bounds=bounds,
                maxfev=5000,
            )
            curves_temporal_frequencies[i, :] = popt

            fitted_curves[i, :] = vonMises(unique_directions, *popt)

        result[neuron] = {
            "fitted_curves": fitted_curves,
            "mean_spike_counts": mean_spike_counts,
            "std_spike_counts": std_spike_counts,
        }
    return result

def getMaxOfTemporalTuningCurves(
    tuning_curve_fit: dict,
    stim_table,
):
    unique_dirs = np.unique(stim_table["orientation"].dropna())
    unique_dirs.sort()
    unique_temps = np.unique(stim_table["temporal_frequency"].dropna())
    unique_temps.sort()
    results = {}
    for neuron in tuning_curve_fit.keys():
        fitted_curves = tuning_curve_fit[neuron]["fitted_curves"].copy()
        max_curve_idx_for_all_directions = np.argmax(fitted_curves, axis=0)
        max_direction_idx = np.argmax(np.max(fitted_curves, axis=0))

        # print(
        #     max_curve_idx_for_all_directions,
        #     np.max(fitted_curves, axis=0),
        #     max_direction_idx,
        # )
        # print(
        #     tuning_curve_fit[neuron]["fitted_curves"][
        #         max_curve_idx_for_all_directions[max_direction_idx]
        #     ]
        # )

        # remove the maximum direction from the list and get the second maximum
        fitted_curves[
            max_curve_idx_for_all_directions[max_direction_idx], max_direction_idx
        ] = 0
        # print(
        #     tuning_curve_fit[neuron]["fitted_curves"][
        #         max_curve_idx_for_all_directions[max_direction_idx]
        #     ]
        # )
        max_direction_idx2 = np.argmax(
            fitted_curves[
                max_curve_idx_for_all_directions[max_direction_idx]
            ]
        )

        is_orientational = (
            1
            if (unique_dirs[max_direction_idx] + 180) % 360
            == unique_dirs[max_direction_idx2]
            else 0
        )
        results[neuron] = {
            "max_direction": unique_dirs[max_direction_idx],
            "max_direction2": unique_dirs[max_direction_idx2],
            "is_orientationnal": is_orientational,
        }
        
    return results

def testTuningFunction_opt(
    inferred_spikes: np.ndarray,
    stim_table=pd.DataFrame(),
    psi: int = 1,
    niters: int = 1000,
    random_seed: int = 2046,
    to_file: bool = True,
) -> [dict, np.ndarray]:
    starts = stim_table["start"].astype(int).to_numpy()
    ends = stim_table["end"].astype(int).to_numpy()

    dirs = stim_table["orientation"].dropna()
    unique_directions = np.unique(dirs).astype(int)
    unique_directions.sort()
    theta_k = np.deg2rad(np.unique(unique_directions))

    temporal_frequencies = stim_table["temporal_frequency"].dropna()
    unique_temporal_frequencies = np.unique(temporal_frequencies).astype(int)
    unique_temporal_frequencies.sort()

    qdistr = np.zeros((inferred_spikes["binspikes"].shape[0], len(unique_temporal_frequencies)+1, niters))
    result = {}
    for neuron in tqdm(range(inferred_spikes["binspikes"].shape[0])):
        result[neuron] = {}
        # Vectorized calculation of spike counts
        # TODO ist das nicht die bin_spike_counts Funktion?
        spike_counts = np.array(
            [
                np.sum(inferred_spikes["binspikes"][neuron, start:end])
                for start, end in zip(starts, ends)
            ]
        )
        spike_counts = spike_counts[stim_table["orientation"].dropna().index]

        for tf, temporal_frequency in enumerate(
            np.concatenate((unique_temporal_frequencies, [-1]))
        ):
            m_k = np.array(
                [
                    np.mean(
                        spike_counts[temporal_frequencies == temporal_frequency][
                            dirs[temporal_frequencies == temporal_frequency] == d
                        ]
                        if temporal_frequency != -1
                        else spike_counts[dirs == d]
                    )
                    for d in unique_directions
                ]
            )
            v_k = np.exp(psi * 1j * theta_k)
            q = np.abs(np.dot(m_k, v_k))

            rng = np.random.default_rng(random_seed)

            for i in range(niters):
                shuffled_counts = rng.permutation(spike_counts)
                shuffled_m_k = np.array(
                    [np.mean(shuffled_counts[dirs == d]) for d in unique_directions]
                )
                qdistr[neuron, tf, i] = np.abs(np.dot(shuffled_m_k, v_k))

            p = np.sum(qdistr[neuron, tf, :] >= q) / niters
            result[neuron][temporal_frequency] = {
                "p": p,
                "q": q,
            }
    if to_file:
        file_string = "or" if psi == 2 else "dir"
        with open(f"../data/qp_tuning_test_{file_string}.pkl", "wb") as f:
            pickle.dump(result, f)
        with open(f"../data/qdistr_tuning_test_{file_string}.pkl", "wb") as f:
            pickle.dump(qdistr, f)
    return result, qdistr

def getTemporalTunings(
    inferred_spikes: np.ndarray,
    stim_table=pd.DataFrame(),):
    """
    Calculates mean and standard deviation of spike counts for each neuron and temporal frequency.

    Args:
        stim_table (DataFrame): Table containing stimulus information, including 'temporal_frequency'.
        inferred_spikes (DataFrame): Spike data with a 'binspikes' column for each neuron.

    Returns:
        np.ndarray: Array with shape (2, neurons, unique_temporal_frequencies),
                    where the first dimension represents mean (0) and std dev (1).
    """
    unique_dirs = np.unique(stim_table["temporal_frequency"].dropna())
    neurons = inferred_spikes["binspikes"].shape[0]
    temporal_tunings = np.zeros((2, neurons, len(unique_dirs)))
    stim_table_tf_nona = stim_table["temporal_frequency"].dropna()

    for neuron in range(neurons):
        spike_count = bin_spike_counts(
            stim_table, inferred_spikes, neuron=neuron
        )[stim_table["temporal_frequency"].notna()]

        temporal_tunings[:, neuron, :] = np.array(
            [
                (
                    np.mean(spike_count[stim_table_tf_nona == d]),
                    np.std(spike_count[stim_table_tf_nona == d]),
                )
                for d in unique_dirs
            ]
        ).T  # Transpose to match desired shape

    return temporal_tunings

def load_tuning_test_results(orientation: bool = True):
    file_string = "or" if orientation else "dir"
    with open(f"../data/qp_tuning_test_{file_string}.pkl", "rb") as f:
        result = pickle.load(f)
    with open(f"../data/qdistr_tuning_test_{file_string}.pkl", "rb") as f:
        qdistr = pickle.load(f)
    return result, np.array(qdistr)

# asses temporal frequency:
def bin_spike_counts(stim_table, spikes, neuron):
    spike_count = np.zeros(len(stim_table["start"]))
    for i in range(len(stim_table)):
        start = stim_table["start"][i].astype(int)
        end = stim_table["end"][i].astype(int)
        spike_count[i] = np.sum(spikes["binspikes"][neuron, start:end])

    return spike_count

def process_tuning_results(
    testTuningFunctionResultsDir: dict,
    stim_table: pd.DataFrame,
    inferred_spikes: dict,
    max_of_temporal_tuning_curve: dict,
    keys: list,
    p_thresh: float = 0.0001
) -> pd.DataFrame:
    """
    Processes the results of temporal frequency tuning tests, creating a DataFrame
    with relevant statistics for further analysis.

    Args:
        testTuningFunctionResultsDir: Dictionary containing test results per neuron.
        p_thresh: Significance threshold for p-values.
        keys: List of temporal frequencies used in the tests.

    Returns:
        DataFrame containing processed results.
    """
    spike_count = np.array(
    [
        bin_spike_counts(stim_table, inferred_spikes, neuron=neuron)
        for neuron in range(inferred_spikes["binspikes"].shape[0])
    ]
    )
    
    rows = []

    for neuron, temporal_resolutions in testTuningFunctionResultsDir.items():
        # Create a row with the neuron index and its p-values
        row = {"Neuron": neuron}
        for resolution, values in temporal_resolutions.items():
            if resolution == -1:
                row["all freq."] = 1 if values["p"] <= p_thresh else 0
            else:
                row[resolution] = (
                    1 if values["p"] <= p_thresh else 0
                )  # Extract the p value
            row[f"p_val_{resolution}"] = values["p"]
            row[f"q_{resolution}"] = values["q"]  # Extract q value
        rows.append(row)

    # Create DataFrame from rows
    df_dir = pd.DataFrame(rows)
    df_dir.columns = df_dir.columns.astype(str)
    df_dir.set_index("Neuron", inplace=True)
    for freq in keys[0:-1]:
        df_dir[f"LI_{freq}"] = df_dir[f"q_{freq}"] / np.mean(
            spike_count[:, stim_table["temporal_frequency"] == freq], axis=1
        )
    # Calculate our linearity Index:
    df_dir["LI_all"] = df_dir["q_-1"] / np.mean(spike_count, axis=1)

    filtered_columns = [col for col in df_dir.columns if col.startswith("LI")]
    df_dir["complex"] = df_dir[filtered_columns].lt(1).all(axis=1)
    df_dir["complex_flag"] = df_dir["complex"].apply(lambda x: 1 if x else 0)
    cols_to_remove = [col for col in df_dir.columns if col.startswith(("LI", "q"))]
    df_dir = df_dir.drop(columns=cols_to_remove)
    # Convert the dictionary to a DataFrame
    max_df = pd.DataFrame.from_dict(max_of_temporal_tuning_curve, orient="index")

    # Concatenate the new DataFrame with the existing one
    df_dir = pd.concat([df_dir, max_df], axis=1)

    return df_dir

def process_tuning_data(
    testTuningFunctionResultsOr: dict,
    max_of_temporal_tuning_curve: dict,
    df_dir: pd.DataFrame,
    keys_str: list,
    p_thresh: float = 0.0001,  # Added default p-value threshold
):
    """
    Processes temporal frequency tuning data, constructs DataFrames,
    and calculates summary statistics for complex and non-complex neurons.

    Args:
        testTuningFunctionResultsOr: Dictionary containing test results per neuron.
        max_of_temporal_tuning_curve: Dictionary with maximum values for the tuning curve.
        keys_str: List of string keys representing temporal frequencies.
        p_thresh: (Optional) P-value threshold for determining significance (default: 0.0001).

    Returns:
        df_or: DataFrame with processed results.
        df_complex: DataFrame for complex neurons (excluding 'complex' column).
        df_non_complex: DataFrame for non-complex neurons (excluding 'complex' column).
        comp_or: Sum of values across `keys_str` columns for complex neurons in `df_or`.
        noncomp_or: Sum of values across `keys_str` columns for non-complex neurons in `df_or`.
        comp_dir: Sum of values across `keys_str` columns for complex neurons in `df_dir`.
        noncomp_dir: Sum of values across `keys_str` columns for non-complex neurons in `df_dir`.
    """

    # Create rows for DataFrame (with dictionary comprehension)
    rows = [
        {
            "Neuron": neuron,
            **{
                resolution: 1 if values["p"] <= p_thresh else 0
                for resolution, values in temporal_resolutions.items()
                if resolution != -1
            },
            "all freq.": (
                1 if temporal_resolutions[-1]["p"] <= p_thresh else 0
            ),  # Calculate "all freq." value
            **{
                f"p_val_{resolution}": values["p"]
                for resolution, values in temporal_resolutions.items()
            },
        }
        for neuron, temporal_resolutions in testTuningFunctionResultsOr.items()
    ]

    # Create DataFrame from rows and set 'Neuron' as index
    df_or = pd.DataFrame(rows)
    df_or.columns = df_or.columns.astype(str)
    # Set the Neuron column as the index
    df_or.set_index("Neuron", inplace=True)
    df_or["complex"] = df_dir["complex"]
    max_df = pd.DataFrame.from_dict(max_of_temporal_tuning_curve, orient="index")

    # Concatenate the new DataFrame with the existing one
    df_or = pd.concat([df_or, max_df], axis=1)

    ## Save some data for the plots below:
    df_complex = df_or[df_or["complex"]].drop("complex", axis=1)
    df_non_complex = df_or[~df_or["complex"]].drop("complex", axis=1)
    comp_or = df_complex[keys_str].sum()
    noncomp_or = df_non_complex[keys_str].sum()

    df_complex = df_dir[df_dir["complex"]].drop("complex", axis=1)
    df_non_complex = df_dir[~df_dir["complex"]].drop("complex", axis=1)
    comp_dir = df_complex[keys_str].sum()
    noncomp_dir = df_non_complex[keys_str].sum()

    return df_or, df_complex, df_non_complex, comp_or, noncomp_or, comp_dir, noncomp_dir

def kolomogrovTest(df:pd.DataFrame(),
                   df2:pd.DataFrame() = None,
                   columns: list = ["p_val_1", "p_val_2", "p_val_4", "p_val_8", "p_val_15", "p_val_-1"],
                   base_column: str = "p_val_-1") -> pd.DataFrame():
    # Perform the Kolmogorov-Smirnov test for each pair of distributions
    if df2 is not None:
        ks_statistic, p_value = stats.ks_2samp(df[base_column], df2[base_column])
        print(f"KS Statistic: {ks_statistic}")
        print(f"P-value: {p_value}")
        return
    
    results = []
    for col in df[columns].columns:
        if col != base_column:
            ks_statistic, p_value = stats.ks_2samp(df[base_column], df[col])
            freq = col.split("_")[-1] + "_hz"
            results.append(
                {
                    "Base Distribution": "all_freq",
                    "Compared Distribution": freq,
                    "KS Statistic": ks_statistic,
                    "P-value": p_value,
                }
            )
    results_df = pd.DataFrame(results)
    print("----------------------------------------------------")
    print("Significance Testing for Orientational Tuned Neurons")
    print("----------------------------------------------------")
    print(results_df)
    print("----------------------------------------------------")

    return results_df