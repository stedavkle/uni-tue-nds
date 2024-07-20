import oopsi
import pandas as pd
import numpy as np
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
            "infspikes": np.array, (n_cells, n_samples) # thresholded inferred spikes with {0, 1}
        }
        Dictionary containing the inferred spikes for each cell.
    """
    spikes = {}

    with open(file_path, "rb") as f:
        spikes = pickle.load(f)

    # convert the lists to numpy arrays
    spikes["spikes"] = np.array(spikes["spikes"])
    spikes["deconv"] = np.array(spikes["deconv"])
    spikes["infspikes"] = np.array(spikes["infspikes"])

    return spikes


### functions for processing

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
            "infspikes": np.array, (n_cells, n_samples) # thresholded inferred spikes with {0, 1}
        }
        Dictionary containing the inferred spikes for each cell.
    """

    spikes = {
        "spikes": np.zeros(dff.shape),
        "deconv": np.zeros(dff.shape),
        "infspikes": np.zeros(dff.shape)
    }
    for idxCell in tqdm(range(dff.shape[0])):
        oopsi_inf = oopsi.fast(dff[idxCell], dt=dt)
        spike_train = [1 if value > thresh else 0 for value in oopsi_inf[0]]

        # store the results
        spikes["spikes"][idxCell, :] = oopsi_inf[0]
        spikes["deconv"][idxCell, :] = oopsi_inf[1]
        spikes["infspikes"][idxCell, :] = spike_train

    if to_file:
        with open("../data/inference_oopsi.pkl", "wb") as f:
            pickle.dump(spikes, f)

    return spikes

def oasis_inference(dff: np.array, dt: float, thresh: int = 0.035, ar_order: int = 1, to_file: bool = False) -> dict:
    """
    Perform spike inference using the OASIS algorithm.
    
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
            "infspikes": np.array, (n_cells, n_samples) # thresholded inferred spikes with {0, 1}
        }
        Dictionary containing the inferred spikes for each cell.
    """
    spikes = {
        "spikes": np.zeros(dff.shape),
        "deconv": np.zeros(dff.shape),
        "infspikes": np.zeros(dff.shape)
    }
    for idxCell in tqdm(range(dff.shape[0])):
        if ar_order == 1:
            c, s, b, g, lam = deconvolve(dff[idxCell], penalty=1, optimize_g=5, max_iter=5)
            c, s = oasisAR1(dff[idxCell].astype(np.float64), g=g, s_min=0.)
        else:
            c, s, b, g, lam = deconvolve(dff[idxCell], g=(None, None), penalty=1, optimize_g=5, max_iter=5)
            c, s = oasisAR2(dff[idxCell].astype(np.float64), g1=g[0], g2=g[1], s_min=0.)
        spike_train = [1 if value > thresh else 0 for value in c]

        spikes["spikes"][idxCell, :] = s
        spikes["deconv"][idxCell, :] = c
        spikes["infspikes"][idxCell, :] = spike_train
    
    if to_file:
        with open("../data/inference_oasis.pkl", "wb") as f:
            pickle.dump(spikes, f)

    return spikes

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


### Visualization Helper
def get_epochs_in_range(stim_epoch_table: pd.DataFrame, start: int = 0, end: int = 105967) -> pd.DataFrame:
    """
    Get the epochs which don't show the locally sparse noise stimulus in a given range.
    
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
        stim_epoch_table["stimulus"] != "locally_sparse_noise"
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


### spike binning

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

