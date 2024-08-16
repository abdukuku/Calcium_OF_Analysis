# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2
# from curses import tparm
from pathlib import Path
import sys
workspace_dir = Path().resolve()
src_dir = workspace_dir / 'src'
# Recursively add all subfolders under src_dir to sys.path
for subfolder in src_dir.rglob('*'):
    if subfolder.is_dir() and str(subfolder) not in sys.path:
        sys.path.append(str(subfolder))
from scipy.ndimage import label
from re import A
import numpy as np
import pandas as pd
import pynapple as nap
import copy
import sys
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from datetime import date
import warnings
import matplotlib.pyplot as plt
import traceback
# Ignore future warnings from pandas
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("error", "invalid value encountered in divide")

# Your pandas code here

current_file_path = Path(__file__).resolve()
module_dir = str(current_file_path.parents[1])
if module_dir not in sys.path:
    sys.path.append(module_dir)

#from neural_tuning import spearmanr_nan, rotate_tmaps, find_max_coordinates


def classify_place_cell(bits_per_event, calcium_events, bootstrap_distribution):
    """
    Classifies a neuron as a place cell based on specific criteria.
    
    Parameters:
    - bits_per_event (float): The bits per event of the neuron.
    - calcium_events (int): The number of calcium events observed for the neuron.
    - bootstrap_distribution (list or np.ndarray): The bootstrapped distribution of information content.

    Returns:
    - bool: True if the neuron is classified as a place cell, False otherwise.
    """
    # Criterion 1: Bits per event must exceed 0.2
    if bits_per_event <= 0.2:
        return False

    # Criterion 2: Must exhibit more than five calcium events
    if calcium_events <= 5:
        return False

    # Criterion 3: Information content must exceed 95% of the bootstrapped distribution
    threshold = np.percentile(bootstrap_distribution, 95)
    if bits_per_event <= threshold:
        return False

    return True


def sparsity(tcurves2d_dict, occupancy_map):
    """
    Compute sparsity of a rate map, The sparsity  measure is an adaptation
    to space. The adaptation measures the fraction of the environment  in which
    a cell is  active. A sparsity of, 0.1 means that the place field of the
    cell occupies 1/10 of the area the subject traverses [2]_

    Parameters
    ----------
    rate_map : numpy.ndarray
        A firing rate map, any number of dimensions.

    Returns
    -------
    out : float
        sparsity

    References
    ----------
    .. [2] Skaggs, W. E., McNaughton, B. L., Wilson, M., & Barnes, C. (1996).
       Theta phase precession in hippocampal neuronal populations and the
       compression of temporal sequences. Hippocampus, 6, 149-172.
    """
    """
    Originally from https://github.com/MattNolanLab/gridcells
    LICENSE: GPLv3
    """

    sparsity_dict = {}
    for matrix_key, tmp_rate_map in tcurves2d_dict.items():
        tmp_rate_map[np.isnan(tmp_rate_map)] = 0
        avg_rate = np.sum(np.ravel(tmp_rate_map * occupancy_map))
        avg_sqr_rate = np.sum(np.ravel(tmp_rate_map**2 * occupancy_map))
        sparsity_dict[matrix_key] = avg_rate**2 / avg_sqr_rate

    return sparsity_dict


def selectivity(tcurves2d_dict, occupancy_map):
    """
    "The selectivity measure max(rate)/mean(rate)  of the cell. The more
    tightly concentrated  the cell's activity, the higher the selectivity.
    A cell with no spatial tuning at all will  have a  selectivity of 1" [2]_.

    Parameters
    ----------
    rate_map : numpy.ndarray
        A firing rate map, any number of dimensions.

    Returns
    -------
    out : float
        selectivity
    """

    """
    Originally from https://github.com/MattNolanLab/gridcells
    LICENSE: GPLv3
    """
    selectivity_dict = {}
    for matrix_key, tmp_rate_map in tcurves2d_dict.items():
        tmp_rate_map[np.isnan(tmp_rate_map)] = 0
        avg_rate = np.sum(np.ravel(tmp_rate_map * occupancy_map))
        max_rate = np.max(np.ravel(tmp_rate_map))
        selectivity_dict[matrix_key] = max_rate / avg_rate

    return selectivity_dict


def smooth_tunningcurve(tcurves2d_dict, sigma=1.5):
    """
    Find the coordinates of the maximum value for each matrix within a dictionary
    after applying Gaussian filtering.

    Parameters:
        matrix_dict (dict): A dictionary of matrices.
        sigma (float): The standard deviation of the Gaussian filter.

    Returns:
        dict: A dictionary where keys are matrix names and values are tuples containing
              the row (i) and column (j) indices of the maximum value.
    """
    smooth_tcurves2d_dict = {}
    small_constant = 1e-6
    for cell_id, tcurve in tcurves2d_dict.items():
        tcurve[tcurve == 0] = small_constant
        smooth_tcurve = gaussian_filter(tcurve, sigma=sigma)
        smooth_tcurves2d_dict[cell_id] = smooth_tcurve
    return smooth_tcurves2d_dict


def computeCentroidDispersion(Spikes, Position, ep=None):
    """
    This function calculates the centroid and dispersion of of each neuron

    Args:
        Spikes (npz): A tsd array containing spike waveforms.
        Position (npz): A tsd array containing the spike's spatial location information.
        epochs (list): A list of epochs to analyze.
    Returns:
        coordinates (list): A list of the spike's spatial coordinates.
        centroid (list): The centroid of the spike's spatial distribution.
        mean_dispersion (float): The mean dispersion of the spike's spatial distribution.
    """
    if isinstance(ep, nap.IntervalSet):
        position = position.restrict(ep)
        spikes = spikes.restrict(ep)
    Center_Data = {}
    for n in Spikes.columns:
        t = Spikes.t[Spikes[n].values > 0]
        ts = nap.Ts(t=t, time_units="s")
        coordinates = ts.value_from(Position)
        centroid = [np.median(coordinates[:, 0]), np.median(coordinates[:, 1])]
        dispersion = []
        for x, y in zip(coordinates[:, 0], coordinates[:, 1]):
            dispersion.append(np.sqrt((x - centroid[0]) ** 2 + (y - centroid[1]) ** 2))
        mean_dispersion = np.mean(dispersion)
        Center_Data[n] = {
            "coordinates": coordinates,
            "centroid": centroid,
            "mean_dispersion": mean_dispersion,
        }

    return Center_Data

def compute_prerievent(df_results, Spikes, Trace, intervals, suffix):
    """
    Computes perievent data for spike and trace signals around specified intervals
    and updates the input DataFrame with these computed values.
    """

    Prevent_Spikes = nap.compute_perievent_continuous(Spikes, intervals, minmax=(-15, 15), time_unit="s")
    Prevent_Trace = nap.compute_perievent_continuous(Trace, intervals, minmax=(-15, 15), time_unit="s")
    df_results["Prievent_Trace_" + suffix] = "NaN"
    df_results["Prievent_Spike_" + suffix] = "NaN"
    for i, ID in enumerate(Spikes.columns):
        df_results.at[ID, "Prievent_Trace_" + suffix] = Prevent_Trace.as_array()[:, :, i]
        df_results.at[ID, "Prievent_Spike_" + suffix] = Prevent_Spikes.as_array()[:, :, i]
    return df_results


def tcurve_threshold(tcurve, n_std=1.2):
    """
    Calculate the threshold for a tuning curve based on the mean and standard deviation of the curve.

    Parameters:
    tcurve (array-like): The tuning curve values.
    n_std (float): The number of standard deviations to add to the mean.

    Returns:
    float: The threshold value.

    """
    mean_rate = np.mean(tcurve)
    std_rate = np.std(tcurve)
    # Determine bins exceeding the mean rate by >1.2 SD
    threshold = mean_rate + n_std * std_rate

    return tcurve > threshold


def compute_circle_shift(input_spikes):
    """Efficient circular shift without deep copying the entire array."""
    timestamps = input_spikes.shape[0]
    for n in range(input_spikes.shape[1]):
        random_ts = np.random.randint(timestamps)
        input_spikes[:, n] = np.roll(input_spikes[:, n], random_ts)
    return input_spikes


def compute_firing_rate(spikes, epochs=None):
    """Calculates firing rate."""
    if epochs:
        spike_restrict = spikes.restrict(epochs)
        return (spike_restrict.as_dataframe() > 0).sum() / spike_restrict.time_support.tot_length()
    return (spikes.as_dataframe() > 0).sum() / spikes.time_support.tot_length()


def compute_null_distribution(spikes, epochs, n_shuffles):
    """
    Compute the null distribution for a given set of spikes and epochs.

    Parameters:
    - spikes: Spike train data
    - epochs: Epochs of interest
    - n_shuffles: Number of iterations for computing the null distribution

    Returns:
    - normalized_null_distribution: Normalized null distribution
    """
    null_distribution = spikes.restrict(epochs).as_dataframe().sum() / spikes.restrict(epochs).time_support.tot_length()
    null_distribution = null_distribution.to_frame()

    distribution = {}
    for shuf in tqdm(range(n_shuffles)):
        distribution[f"Shuffle_{shuf+1}"] = (
            compute_circle_shift(spikes).restrict(epochs).as_dataframe().sum() / spikes.restrict(epochs).time_support.tot_length()
        )
    null_distribution = pd.concat([null_distribution, pd.DataFrame(distribution)], axis=1)
    null_distribution["Count"] = null_distribution.apply(lambda row: (row[1:] > row[0]).sum(), axis=1)

    return null_distribution["Count"] / n_shuffles


def load_project_data(project, mouse_id, day, experiment_type, Trace_type="Ca_Events"):
    """
    Load project data based on the experiment type for a specific mouse and day.

    Parameters:
    - project: The project object containing the data.
    - mouse_id: The ID of the mouse.
    - day: The day of the experiment.
    - experiment_type: The type of experiment ("CFC", "OpenField", or "OLM").
    - Trace_type: The type of trace data to load (default is "Ca_Events").
    """
    match experiment_type:
        case "CFC":
            spikes = project[f"Calcium_Trace_{mouse_id}_{day}_Ca_Events"]
            spikes_trace = project[f"Calcium_Trace_{mouse_id}_{day}_zscored"]
            freeze_interval = project[f"FreezeEpoch_{day}"]
            return spikes, spikes_trace, freeze_interval
        case "OpenField":
            spikes = project[f"Calcium_Trace_{mouse_id}_{day}_{Trace_type}"]
            spikes_trace = project[f"Calcium_Trace_{mouse_id}_{day}_zscored"]
            neuroninfo = pd.read_csv(Path(project.path) / f"Calcium_{mouse_id}_{day}_cell_metrics.csv")
            position = project[f"Position_{mouse_id}_{day}"]
            return spikes, spikes_trace, position,neuroninfo
        case "OLM":
            spikes = project[f"Calcium_Trace_{mouse_id}_{day}_Ca_Events"]
            spikes_trace = project[f"Calcium_Trace_{mouse_id}_{day}_zscored"]
            exploration_intervals = {"object_1": project[f"Object1_{day}"], "object_2": project[f"Object2_{day}"]}
            return spikes, spikes_trace, exploration_intervals


def compute_velocity(position):
    """
    Compute velocity from position data.
    """
    sr = np.round(len(position.t) / position.t[-1], 0)  # sampling rate

    # Calculate the differences in x and y positions
    x = position.values[:, 0]
    y = position.values[:, 1]
    # Compute the distances between consecutive positions
    distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    # Compute the speeds by dividing the distances by the corresponding time intervals
    speeds = distances / np.diff(position.t)
    speeds = np.convolve(speeds, np.ones(2 * int(sr)) / (2 * sr), mode="same")
    speeds = np.insert(speeds, 0, 0)
    Speed = nap.Tsd(t=position.t, d=speeds, time_units="s")
    return Speed

def create_results_dataframe(mouse_id, day, mouse_info, columns):
    """Create an initial DataFrame for storing results."""
    df = pd.DataFrame(index=columns)
    df["MouseID"] = mouse_id
    df["Day"] = day
    df["Sex"] = mouse_info["Sex"]
    df["Genotype"] = mouse_info["Genotype"]
    df["CellID"] = df.index
    return df


def populate_spikes(df, spikes, field="Spikes"):
    """Populate the DataFrame with spike data."""
    df[field] = "NaN"
    for col in spikes.columns:
        df.at[col, field] = spikes[col].as_series()
    return df

def compute_SI_2D_shuffles(df_results, spikes, position, epochs, n_x_bins, n_y_bins, n_shuffles=0):
    """
    Compute shuffled Spatial Information (SI) values for 2D data.
    
    """
    
    si = df_results["PF_SI"].copy(deep=True).to_frame()  # Spatial Information dictionary
    si_shuffle = {}
    for ii in tqdm(range(n_shuffles)):
        spikes_shuffle = compute_circle_shift(spikes)
        tc_shuffle, _ = nap.compute_2d_tuning_curves_continuous(spikes_shuffle, position, (n_x_bins, n_y_bins), epochs)
        tc_shuffle = smooth_tunningcurve(tc_shuffle, sigma=1.5)
        si_shuffle[f"shuf_SI{ii}"] = nap.compute_2d_mutual_info(tc_shuffle, position)["SI"]
    si = pd.concat([si, pd.DataFrame(si_shuffle)], axis=1)
    si["Count"] = si.apply(lambda row: (row.iloc[1:] > row["PF_SI"]).sum(), axis=1)
    df_results["PF_pvalue"] = si["Count"] / n_shuffles


def compute_SI_1D_shuffles(df_results, spikes, velocity, n_shuffles=0):
    """
    Compute shuffled Spatial Information (SI) values for 1D data.
    """
    si = df_results["Speed_SI"].copy(deep=True).to_frame()  # Spatial Information dictionary
    si_shuffle = {}
    for ii in tqdm(range(n_shuffles)):
        spikes_shuffle = compute_circle_shift(spikes)
        tc_shuffle = nap.compute_1d_tuning_curves_continuous(spikes_shuffle, velocity, 20)
        si_shuffle[f"shuf_SI{ii}"] = nap.compute_1d_mutual_info(tc_shuffle, velocity)["SI"]
    si = pd.concat([si, pd.DataFrame(si_shuffle)], axis=1)
    si["Count"] = si.apply(lambda row: (row.iloc[1:] > row["Speed_SI"]).sum(), axis=1)
    df_results["Speed_pvalue"] = si["Count"] / n_shuffles


def correlate_turning(row, key1, key2):
    """
    Calculate the correlation between 'Turning' values of two specified keys in the given row.

    Parameters:
    - row (dict): The input dictionary representing a data row.
    - key1 (str): The first key indicating the first set of data.
    - key2 (str): The second key indicating the second set of data.
    """
    if any(key not in row["Day"].values for key in [key1, key2]):
        return row

    A_index = row.query("Day==@key1").index
    B_index = row.query("Day==@key2").index
    A = row.loc[A_index]["Turning"].values[0]
    B = row.loc[B_index]["Turning"].values[0]
    A_threshold = tcurve_threshold(A)
    B_threshold = tcurve_threshold(B)

    Corr_Pv = {}
    for rot in [0, 90, 180, 270]:
        Corr_Pv[rot] = spearmanr_nan(A_threshold, rotate_tmaps(B_threshold, rot))
    row[key1 + "_" + key2] = {A_index[0]: Corr_Pv, B_index[0]: Corr_Pv}

    return row

def correlate_PF(row, key1, key2):
    """
    Correlates the PF (Place Field) values for two given keys in a row.
    
    Parameters:
    row (pandas.Series): The row containing the data.
    key1 (str): The first key to correlate.
    key2 (str): The second key to correlate.
    
    """
    A_index = row[row["Day"] == key1].index
    B_index = row[row["Day"] == key2].index
    A = row.loc[A_index]["PF_pvalue"].values
    B = row.loc[B_index]["PF_pvalue"].values
    if len(A) > 0 and len(B) > 0:  # Check if A and B are not empty
        if A < 0.05 and B < 0.05:
            row["PF_" + key1 + "_" + key2] = {A_index[0]: True, B_index[0]: True}
        return row
    else:
        return row


def computeOccupancy(position, epochs, nb_bins=(20, 20)):
    """
    Compute the occupancy of a given position in a specified epoch.

    Parameters:
    position (array-like): The position data.
    epochs (array-like): The epochs data.
    nb_bins (tuple, optional): The number of bins for x and y axes. Defaults to (20, 20).

    Returns:
    array-like: The occupancy matrix.

    """
    xpos = position.restrict(epochs).values[:, 0]
    ypos = position.restrict(epochs).values[:, 1]
    xbins = np.linspace(xpos.min(), xpos.max() + 1e-6, nb_bins[0] + 1)
    ybins = np.linspace(ypos.min(), ypos.max() + 1e-6, nb_bins[1] + 1)
    occupancy, _, _ = np.histogram2d(ypos, xpos, [ybins, xbins])
    occupancy = occupancy / np.sum(occupancy)
    return occupancy


def place_field_size(tcurves2d_dict, plot_pfsize=False):
    """
    Compute the size of the place field for each cell.

    Parameters:
    - tcurves2d_dict (dict): A dictionary of 2D tuning curves.

    Returns:
    - dict: A dictionary where keys are cell IDs and values are the size of the place field.
    """
    place_field_size_dict = {}
    place_field_mask_dict = {}
    for cell_id, tcurve in tcurves2d_dict.items():
        # Identify the peak rate bin
        peak_rate_bin = np.unravel_index(np.argmax(tcurve), tcurve.shape)
        peak_rate = tcurve[peak_rate_bin]
        # Calculate mean and standard deviation of the rate map values
        above_threshold = tcurve_threshold(tcurve)
        # Step 4: Identify contiguous bins including the peak rate bin
        labeled_array, num_features = label(above_threshold)
        peak_label = labeled_array[peak_rate_bin]
        place_field_mask = labeled_array == peak_label
        place_field_size = np.sum(place_field_mask)
        place_field_size_dict[cell_id] = place_field_size
        place_field_mask_dict[cell_id] = place_field_mask
        if plot_pfsize:
            # Plotting the place field
            plt.figure(figsize=(8, 6))
            plt.imshow(tcurve, cmap="viridis", origin="lower")
            plt.colorbar(label="Firing Rate")
            plt.scatter(peak_rate_bin[1], peak_rate_bin[0], color="red", label="Peak Rate Bin")

            # Highlight the place field
            place_field_mask = labeled_array == peak_label
            plt.contour(place_field_mask, levels=[0.5], colors="white", linewidths=1.5)

            plt.title(f"Place Field (Size: {place_field_size} bins)")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.legend()
            plt.show()

    return place_field_size_dict,place_field_mask_dict


def Speed_Score(df_results, spikes, velocity, n_shuffles=0):

    tcurve = nap.compute_1d_tuning_curves_continuous(spikes, velocity, 20)
    df_results["Speed_SI"] = nap.compute_1d_mutual_info(tcurve, velocity)
    df_results["Speed_tcurve"] = tcurve.to_dict()
    if n_shuffles == 0:
        return df_results
    else:
        compute_SI_1D_shuffles(df_results, spikes, velocity, n_shuffles)
        return df_results
        # #from matplotlib import pyplot as plt
    # order = tcurve.idxmax().sort_values().index.values
    # plt.figure(figsize=(12, 10))
    # gs = plt.GridSpec(len(order), 1)
    # for i, n in enumerate(order):
    #     plt.subplot(gs[i, 0])
    #     plt.fill_between(tcurve.index.values, np.zeros(len(tcurve)), tcurve[n].values)
    #     plt.yticks([])
    # plt.xlabel("Speed (cm/s)")
    # plt.show()

    return df_results


def PlaceField_2D_analysis(df_results, spikes, position, epochs, n_x_bins, n_y_bins, n_shuffles=0):
    tcurves2d_dict, binsxy = nap.compute_2d_tuning_curves_continuous(spikes, position, (n_x_bins, n_y_bins), epochs)
    tcurves2d_dict = smooth_tunningcurve(tcurves2d_dict, sigma=1.5)
    occupancy_map = computeOccupancy(position, epochs, (n_x_bins, n_y_bins))
    occupancy_map = gaussian_filter(occupancy_map, sigma=1.5)

    df_results["Turning"] = tcurves2d_dict
    #df_results["MaxValue"] = find_max_coordinates(tcurves2d_dict)
    df_results["Sparsity"] = sparsity(tcurves2d_dict, occupancy_map)
    df_results["Selectivity"] = selectivity(tcurves2d_dict, occupancy_map)
    df_results["PF_SI"] = nap.compute_2d_mutual_info(tcurves2d_dict, position)
    df_results["PFsize"],df_results["PFmask"] = place_field_size(tcurves2d_dict, plot_pfsize=False)
    df_results["CenterFiring"] = computeCentroidDispersion(spikes, position)
    # df_results["Frate_Sig"] = compute_null_distribution(spikes=spikes, epochs=epochs, n_shuffles=n_shuffles)

    if n_shuffles == 0:
        return df_results
    else:
        compute_SI_2D_shuffles(df_results, spikes, position, epochs, n_x_bins, n_y_bins, n_shuffles)
        return df_results

def analyze_spike_data(project_path, mouse_info, day, n_shuffles=1000,Trace_type=None):
    """
    Main analysis function to load data, compute metrics, and save results.
    """
    mouse_id = mouse_info["MouseID"]
    project = nap.load_folder(str(project_path))
    spikes,spikes_trace, position,neuroninfo = load_project_data(project, mouse_id, day, experiment_type="OpenField",Trace_type=Trace_type)
    neuroninfo.set_index("cellName", inplace=True)
    position = position.dropna()
    velocity = compute_velocity(position)
    threshold = 0.05 # arbitrary threshold for movement to fit with ratinbox
    epochs = velocity.threshold(threshold).time_support.drop_short_intervals(2)
    rest_epochs = velocity.threshold(threshold, method="below").time_support.drop_short_intervals(2)

    df_results = create_results_dataframe(mouse_id, day, mouse_info, spikes.columns)
    df_results = populate_spikes(df_results, spikes, field="Spikes")
    df_results = populate_spikes(df_results, spikes_trace, field="RawTrace")
    df_results = pd.concat([df_results, neuroninfo], axis=1)

    # Example of how to integrate additional analyses
    n_x_bins, n_y_bins = 20, 20
    df_results = PlaceField_2D_analysis(df_results, spikes_trace, position, epochs, n_x_bins, n_y_bins, n_shuffles)
    df_results = Speed_Score(df_results, spikes_trace, velocity, n_shuffles)
    df_results["Frate"] = compute_firing_rate(spikes)
    df_results["Frate_moving"] = compute_firing_rate(spikes.restrict(epochs))
    df_results["Frate_rest"] = compute_firing_rate(spikes.restrict(rest_epochs))
    rest_start = nap.Ts(t=rest_epochs.start, time_units="s")
    #df_results = compute_prerievent(df_results, spikes, spikes_trace, rest_start, suffix="Reset")
    velocity.save(f"{project_path}\\velocity_{day}")

    return df_results

def log_error(log_file, error_message):
    with open(log_file, "a") as log:
        log.write(error_message + "\n")

def run_for_multiple_mice(base_path, MiceInfo, days, n_shuffles=1000,Trace_type="Ca_Events"):
    """
    Run analysis for multiple mice across multiple days.
    """
    to_date = date.today().strftime("%d-%m-%Y")
    log_file = base_path.parent.parent / f"OpenField{to_date}_{Trace_type}.txt"
    for _, mouse_info in tqdm(MiceInfo.iterrows()):
        result_df = pd.DataFrame()
        project_path = base_path / f"E8KO_{mouse_info['MouseID']}" / "Results"
        for day in days:
            try:
                df_results = analyze_spike_data(project_path, mouse_info, day, n_shuffles=n_shuffles,Trace_type=Trace_type)
                result_df = pd.concat([result_df, df_results], ignore_index=True)
            except Exception as e:
                error_message = f"An error occurred while running OpenField for {mouse_info['MouseID']} day {day}: {e}\n"
                error_message += traceback.format_exc()
                log_error(log_file, error_message)
                print(error_message)
                continue
        try:
            # result_df = result_df.groupby(["CellID"]).apply(correlate_turning, key1="HD1", key2="HD2").reset_index(drop=True)
            # result_df = result_df.groupby(["CellID"]).apply(correlate_turning, key1="HD1", key2="HD3").reset_index(drop=True)
            # result_df = result_df.groupby(["CellID"]).apply(correlate_turning, key1="HD2", key2="HD3").reset_index(drop=True)
            # result_df = result_df.groupby(["CellID"]).apply(correlate_PF, key1="HD1", key2="HD2").reset_index(drop=True)
            # result_df = result_df.groupby(["CellID"]).apply(correlate_PF, key1="HD1", key2="HD3").reset_index(drop=True)
            # result_df = result_df.groupby(["CellID"]).apply(correlate_PF, key1="HD2", key2="HD3").reset_index(drop=True)
            result_df.to_pickle(project_path / f"Results_OLM_{Trace_type}.pkl")
            print(f"Successfully completed analysis for mouse {mouse_info['MouseID']}")
        except Exception as e:
            error_message = f"An error occurred while saving results for {mouse_info['MouseID']}: {e}\n"
            error_message += traceback.format_exc()
            log_error(log_file, error_message)
            print(error_message)
            continue

if __name__ == "__main__":
    print("Starting OpenField analysis")
    base_path = Path.cwd()
    MiceInfo = pd.read_excel(
       base_path / "AnalysisProgress.xlsx",
        dtype={"MouseID": "category", "Sex": "category", "Genotype": "category"},
    ).astype({"MouseID": "int"})
    MiceInfo = MiceInfo[["MouseID", "Sex", "Genotype"]]
    days = ["HD1", "HD2", "HD3"]
    n_shuffles = 500
    Trace_type = "Ca_Events"
    run_for_multiple_mice(base_path, MiceInfo, days, n_shuffles,Trace_type=Trace_type)
    print(f"Successfully completed analysis for all mice for {Trace_type}")
