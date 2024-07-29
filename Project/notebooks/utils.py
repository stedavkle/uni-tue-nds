import oopsi
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, pearsonr, stats, f_oneway
from scipy.optimize import curve_fit
import statsmodels.stats.multitest as smm
from scipy import signal, ndimage
import scipy.special as sp
from tqdm import tqdm
import pickle
from oasis.functions import deconvolve
from joblib import Parallel, delayed
import copy

### helper functions for accessing data
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
    a2 = np.power(a, 2)
    window = np.ones(window_size) / float(
        window_size
    )
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
        The filtered x.
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

def get_running_correlation_max(roi_masks: np.array, inferred_spikes: np.array, running_speed: np.array):
    """
    Calculate the correlation between the running speed and the activity of each cell.
    The correlation is then applied to the roi masks of the cells.
    The top 10 absolute correlation values are returned.

    Parameters
    ----------
    roi_masks: np.array
        The roi masks of the cells.
    inferred_spikes: np.array
        The inferred spikes of the cells.
    running_speed: np.array
        The running speed of the mouse.

    Returns
    -------
    mask: np.array
        The mask with the correlation values applied.
    cell_corr: np.array
        The correlation values for each cell.
    top_10: np.array
        The indices of the top 10 absolute correlation values.
    """
    roi_masks_copy = roi_masks.copy()
    cell_corr = np.zeros(roi_masks_copy.shape[0])
    mask = np.zeros(roi_masks_copy[0].shape, dtype=np.float64)
    for cell in range(roi_masks_copy.shape[0]):
        corr, p = pearsonr(inferred_spikes[cell], running_speed)
        cell_roi = roi_masks_copy[cell].astype(np.float64)
        cell_roi *= corr
        mask = np.where(mask == 0.0, cell_roi, mask)
        cell_corr[cell] = corr
    # get the top 10 correlation values
    top_10 = np.argsort(-np.abs(cell_corr))[:10]
    return mask, cell_corr, top_10

def vonMises(theta: np.ndarray, alpha: float, kappa: float, mu: float, phi: float) -> np.ndarray:
    """
    Evaluate the parametric von Mises tuning curve with parameters p at locations theta.

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

def fitTemporalTuningCurve(inferred_spikes: np.ndarray, stim_table=pd.DataFrame()) -> dict:
    """
    Fit a von Mises tuning curve to the spike counts in count with direction dir using a least-squares fit.
    
    Parameters
    ----------
    inferred_spikes: np.array, shape=(n_neurons, n_timepoints)
        The inferred spike counts for each neuron.
    stim_table: pd.DataFrame
        DataFrame containing the stimulus information

    Return
    ------
    result: dict
        Dictionary containing the fitted curves, mean spike counts and std spike counts for each neuron.
    """
    result = {}

    starts = stim_table["start"].astype(int).to_numpy()
    ends = stim_table["end"].astype(int).to_numpy()

    dirs = stim_table["orientation"].copy().dropna()
    unique_directions = np.unique(dirs).astype(int)
    unique_directions.sort()

    temporal_frequencies = stim_table["temporal_frequency"].copy().dropna()
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
        bounds = (
                [-np.inf, -np.inf, -np.inf, -np.inf],
                [np.inf, np.inf, np.inf, 360],
            )
        def process_frequency(temporal_frequency):
            popt, _ = curve_fit(
                vonMises,
                dirs[temporal_frequencies == temporal_frequency],
                spike_counts[temporal_frequencies == temporal_frequency],
                p0=initial_guess,
                bounds=bounds,
                maxfev=5000,
            )
            fitted_curve = vonMises(unique_directions, *popt)
            return temporal_frequency, popt, fitted_curve

        results = Parallel(n_jobs=32)(
            delayed(process_frequency)(freq) for freq in unique_temporal_frequencies
        )
        # Process results from parallel execution
        for temporal_frequency, popt, fitted_curve in results:
            i = np.where(unique_temporal_frequencies == temporal_frequency)[0][0]
            curves_temporal_frequencies[i, :] = popt
            fitted_curves[i, :] = fitted_curve

        result[neuron] = {
            "fitted_curves": fitted_curves,
            "mean_spike_counts": mean_spike_counts,
            "std_spike_counts": std_spike_counts,
        }
    return result

def getMaxOfTemporalTuningCurves(tuning_curve_fit: dict, stim_table) -> dict:
    """
    Get the maximum direction for each neuron from the tuning curve fit.

    Parameters
    ----------
    tuning_curve_fit: dict
        Dictionary containing the fitted curves, mean spike counts and std spike counts for each neuron.
    stim_table: pd.DataFrame
        DataFrame containing the stimulus information

    Returns
    -------
    results: dict
        Dictionary containing the maximum direction, second maximum direction and whether the neuron is orientation selective or not.
    """
    stim_table = stim_table.copy()
    unique_dirs = np.unique(stim_table["orientation"].dropna())
    unique_dirs.sort()
    unique_temps = np.unique(stim_table["temporal_frequency"].dropna())
    unique_temps.sort()
    results = {}
    for neuron in tuning_curve_fit.keys():
        fitted_curves = copy.deepcopy(tuning_curve_fit[neuron]["fitted_curves"])
        max_curve_idx_for_all_directions = np.argmax(fitted_curves, axis=0)
        max_direction_idx = np.argmax(np.max(fitted_curves, axis=0))

        # remove the maximum direction from the list and get the second maximum
        fitted_curves[
            max_curve_idx_for_all_directions[max_direction_idx], max_direction_idx
        ] = 0
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
    """
    Test the tuning function of the neurons using a permutation test.

    Parameters
    ----------
    inferred_spikes: np.ndarray
        The inferred spikes for each neuron.
    stim_table: pd.DataFrame
        DataFrame containing the stimulus information
    psi: int
        The tuning function to test. 1 for direction, 2 for orientation.
    niters: int
        The number of iterations to run the permutation test.
    random_seed: int
        The random seed for reproducibility.
    to_file: bool
        If True, save the results to a file.

    Returns
    -------
    result: dict
        Dictionary containing the p-value and q-value for each neuron.
    qdistr: np.ndarray
        The distribution of q-values for each neuron.
    """
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
        spike_counts = np.array(
            [
                np.sum(inferred_spikes["binspikes"][neuron, start:end])
                for start, end in zip(starts, ends)
            ]
        )
        spike_counts = spike_counts[stim_table["orientation"].dropna().index]

        for tf, temporal_frequency in enumerate(np.concatenate((unique_temporal_frequencies, [-1]))):
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
            def process_iteration(i, random_seed):  # Pass random_seed to each iteration
                rng = np.random.default_rng(random_seed + i)  # Create separate RNG
                shuffled_counts = rng.permutation(spike_counts)
                shuffled_m_k = np.array(
                    [np.mean(shuffled_counts[dirs == d]) for d in unique_directions]
                )
                return np.abs(np.dot(shuffled_m_k, v_k))

            qdistr_values = Parallel(n_jobs=-1, prefer="processes", batch_size=5)(
                delayed(process_iteration)(i, random_seed) for i in range(niters)
            )
            qdistr[neuron, tf, :] = np.array(qdistr_values)

            p = np.sum(qdistr_values >= q) / len(qdistr_values)
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

def getTemporalTunings(inferred_spikes: np.ndarray, stim_table=pd.DataFrame()) -> np.ndarray:
    """
    Calculates mean and standard deviation of spike counts for each neuron and temporal frequency.

    Parameters
    ----------
    inferred_spikes: np.ndarray
        The inferred spikes for each neuron.
    stim_table: pd.DataFrame
        DataFrame containing the stimulus information

    Returns
    -------
    temporal_tunings: np.ndarray
        The mean and standard deviation of spike counts for each neuron and temporal frequency.
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

def load_tuning_test_results(orientation: bool = True) -> [dict, np.ndarray]:
    """
    Load the tuning test results from a file.

    Parameters
    ----------
    orientation: bool
        If True, load orientation tuning test results. Otherwise, load direction tuning test results.

    Returns
    -------
    result: dict
        Dictionary containing the p-value and q-value for each neuron.
    qdistr: np.ndarray
        The distribution of q-values for each neuron.
    """
    file_string = "or" if orientation else "dir"
    with open(f"../data/qp_tuning_test_{file_string}.pkl", "rb") as f:
        result = pickle.load(f)
    with open(f"../data/qdistr_tuning_test_{file_string}.pkl", "rb") as f:
        qdistr = pickle.load(f)
    return result, np.array(qdistr)

# asses temporal frequency:
def bin_spike_counts(stim_table: pd.DataFrame, spikes: dict, neuron: int) -> np.array:
    """
    Calculate the spike counts for a given neuron and stimulus table.

    Parameters
    ----------
    stim_table: pd.DataFrame
        DataFrame containing the stimulus information
    spikes: dict
        Dictionary containing the inferred spikes for each neuron.
    neuron: int
        The neuron index.

    Returns
    -------
    spike_count: np.array
        The spike counts for the given neuron.
    """
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

    Parameters
    ----------
    testTuningFunctionResultsDir: dict
        Dictionary containing test results per neuron.
    stim_table: pd.DataFrame
        DataFrame containing the stimulus information
    inferred_spikes: dict
        Dictionary containing the inferred spikes for each neuron.
    max_of_temporal_tuning_curve: dict
        Dictionary with maximum values for the tuning curve.
    keys: list
        List of keys representing temporal frequencies.
    p_thresh: float
        P-value threshold for determining significance (default: 0.0001).

    Returns
    -------
    df_dir: pd.DataFrame
        DataFrame with processed results.
    """
    spike_count = np.array(
    [
        bin_spike_counts(stim_table, inferred_spikes, neuron=neuron)
        for neuron in range(inferred_spikes["binspikes"].shape[0])
    ])
    rows = []

    for neuron, temporal_resolutions in testTuningFunctionResultsDir.items():
        # Create a row with the neuron index and its p-values
        row = {"Neuron": neuron}
        for resolution, values in temporal_resolutions.items():
            if resolution == -1:
                row["-1"] = 1 if values["p"] <= p_thresh else 0
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
    p_thresh: float = 0.0001,  # default p-value threshold
) -> pd.DataFrame:
    """
    Processes temporal frequency tuning data, constructs DataFrames,
    and calculates summary statistics for complex and non-complex neurons.

    Parameters
    ----------
    testTuningFunctionResultsOr: dict
        Dictionary containing test results per neuron.
    max_of_temporal_tuning_curve: dict
        Dictionary with maximum values for the tuning curve.
    df_dir: pd.DataFrame
        DataFrame with processed results.
    keys_str: list
        List of keys representing temporal frequencies.
    p_thresh: float
        P-value threshold for determining significance (default: 0.0001).

    Returns
    -------
    df_or: pd.DataFrame
        DataFrame with processed results.
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
            "-1": (
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

def kolmogorovTest(
    df:pd.DataFrame(),
    df2:pd.DataFrame() = None,
    columns: list = ["p_val_1", "p_val_2", "p_val_4", "p_val_8", "p_val_15", "p_val_-1"],
    base_column: str = "p_val_-1",
    direction: bool = False
) -> pd.DataFrame():
    """
    Perform the Kolmogorov-Smirnov test for each pair of distributions.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the p-values for each neuron.
    df2: pd.DataFrame
        DataFrame containing the p-values for each neuron for a second condition.
    columns: list
        List of columns to compare.
    base_column: str
        The base column to compare against.

    Returns
    -------
    results_df: pd.DataFrame
        DataFrame containing the results of the Kolmogorov-Smirnov test.
    """
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
    ordir_string = "Direction" if direction else "Orientation"
    print("----------------------------------------------------")
    print(f"Significance Testing for {ordir_string} Tuned Neurons")
    print("----------------------------------------------------")
    print(results_df)
    print("----------------------------------------------------")
    return results_df
    
def process_permutation(
    i: int, 
    stim_table: pd.DataFrame, 
    inferred_spikes: dict, 
    neuron: int
) -> float:
    """
    Process a single permutation for the permutation test.

    Parameters
    ----------
    i: int
        The permutation index.
    stim_table: pd.DataFrame
        DataFrame containing the stimulus information.
    inferred_spikes: dict
        Dictionary containing the inferred spikes for each neuron.
    neuron: int
        The neuron index.

    Returns
    -------
    float
        The test statistic for the permutation.
    """
    permuted_frequencies = np.random.permutation(
        stim_table["temporal_frequency"].dropna()
    )
    spike_count_by_freq = get_spike_count_by_freq(
        stim_table, permuted_frequencies, inferred_spikes, neuron
    )
    return get_test(spike_count_by_freq)

def get_p_values_permutation_test_helper(
    inferred_spikes: dict, 
    stim_table: pd.DataFrame, 
    n_permutations: int, 
    n_jobs: int = -1
) -> [dict, dict]:
    """
    Helper function to get the p-values for the permutation test.

    Parameters
    ----------
    inferred_spikes: dict
        Dictionary containing the inferred spikes for each neuron.
    stim_table: pd.DataFrame
        DataFrame containing the stimulus information.
    n_permutations: int
        The number of permutations to run.
    n_jobs: int
        The number of jobs to run in parallel.
        
    Returns
    -------
    neuron_stats: dict
        Dictionary containing the test statistic for each neuron.
    permuted_stats: dict
        Dictionary containing the test statistic for each permutation.
    """
    neuron_stats = {}
    permuted_stats = {}
    for neuron in tqdm(range(inferred_spikes["binspikes"].shape[0])):
        spike_count_by_freq = get_spike_count_by_freq(
            stim_table,
            stim_table["temporal_frequency"].dropna().values,
            inferred_spikes,
            neuron,
        )
        neuron_stats[neuron] = get_test(spike_count_by_freq)

        results = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=5)(
            delayed(process_permutation)(i, stim_table, inferred_spikes, neuron)
            for i in range(n_permutations)
        )
        permuted_stats[neuron] = np.array(results)  # Store as numpy array
    return neuron_stats, permuted_stats

def get_spike_count_by_freq(
    stim_table: pd.DataFrame, 
    frequencies: np.array, 
    inferred_spikes: dict, 
    neuron: int = 0
) -> np.array:
    """
    Get the spike count by frequency for a given neuron.

    Parameters
    ----------
    stim_table: pd.DataFrame
        DataFrame containing the stimulus information.
    frequencies: np.array
        The temporal frequencies.
    inferred_spikes: dict
        Dictionary containing the inferred spikes for each neuron.
    neuron: int
        The neuron index.

    Returns
    -------
    spike_count_by_freq: np.array
        The spike count by frequency for the given neuron.
    """
    spike_count = bin_spike_counts(stim_table, inferred_spikes, neuron=neuron)
    spike_count = spike_count[
        stim_table["temporal_frequency"].dropna().index
    ]  # shorten spike count
    unique_frequencies = np.unique(frequencies)
    spike_count_by_freq = np.zeros((len(unique_frequencies), 120))

    for freq in unique_frequencies:
        if len(spike_count[np.where(frequencies == freq)]) == 120:
            spike_count_by_freq[np.where(unique_frequencies == freq)] = spike_count[
                np.where(frequencies == freq)
            ]
        else:
            spike_count_by_freq[np.where(unique_frequencies == freq)] = np.append(
                spike_count[np.where(frequencies == freq)],
                0,  # just add a zero when we are missing a condition...
            )
    return spike_count_by_freq

def get_test(spike_count_by_freq: np.array) -> float:
    """
    Get the test statistic for the permutation test.

    Parameters
    ----------
    spike_count_by_freq: np.array
        The spike count by frequency for a neuron.

    Returns
    -------
    float
        The test statistic for the permutation.
    """
    # test statistic for our permutation test is a one way anove (variation explained by temporal frequency)
    # variance explained by frequency (test the frequency distributions of each neuron against eachother)
    return f_oneway(*[spike_count_by_freq[i, :] for i in range(spike_count_by_freq.shape[0])]).statistic