import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import resample_poly, butter, sosfiltfilt
from sklearn.preprocessing import StandardScaler
from fractions import Fraction
import matplotlib.pyplot as plt
import cv2

import sys
import os
import yaml
import pickle
import h5py




def load_DLC_df(
    filepath,
    pupil_fps
):
    """
    Load DeepLabCut dataframe and collapse multi-level header.
    """
    DLC_df = pd.read_csv(filepath, header=[0,1,2])

    # Flatten multi-level header into single-level names, skipping the first column (frame numbers)
    new_columns = [f"{bp}_{coord}" for i, (scorer, bp, coord) in enumerate(DLC_df.columns) if i != 0]
    # Select only the data columns (skip the first column) and assign flattened column names
    DLC_df = DLC_df.iloc[:, 1:].copy()
    DLC_df.columns = new_columns

    print(f"Raw DeepLabCut output: {len(DLC_df)} frames (or {len(DLC_df) / pupil_fps} seconds).")
    return DLC_df



def filter_high_likelihood_frames(
    DLC_df,
    pupil_fps
):
    """
    Filter for frames with high likelihood (>0.98 in all 6 DeepLabCut bodypart points) as proxy for laser-ON frames.
    """
    # Select only likelihood columns
    likelihood_cols = [col for col in DLC_df.columns if 'likelihood' in col]
    # Set mask
    mask = (DLC_df[likelihood_cols] > 0.98).sum(axis=1) >= 6
    # Find start of the recording
    num_consecutive_frames = 3
    for i in range(len(mask) - num_consecutive_frames + 1):
        # If the consecutive frame is also True
        if mask.iloc[i : i+num_consecutive_frames].all():
            start = mask.index[i]
            break
    # Find end
    end = mask[mask].index[-1]
    laser_on_df = DLC_df.loc[start:end]

    print(f"Laser on from frame {start} to {end} for a duration of {end - start + 1} frames (or {(end - start + 1) / pupil_fps} seconds).")
    return laser_on_df



def add_stimuli_label(
    laser_on_df,
    stimuli_blocks,
    pupil_fps
):
    """
    Add stimulus labels.
    """

    # Initialize all labels as None
    labels = np.full(len(laser_on_df), None)
    # Assign labels sequentially
    i = 0
    for label, duration in stimuli_blocks.items():
        labels[i : i + duration] = label
        i += duration
    # Add labels to dataframe
    laser_on_df = laser_on_df.copy()
    laser_on_df['stimulus'] = labels

    num_stimuli_frames = len(laser_on_df[laser_on_df['stimulus'].notna()])
    print(f"Stimuli (black/grey/movie): {num_stimuli_frames} frames (or {num_stimuli_frames / pupil_fps} seconds).")
    return laser_on_df



def filter_and_interpolate_areas(
    areas
):
    """
    Filter frames with drastic pupil area changes that correspond to blinking activity, then interpolate.
    """
    # Take derivatives
    dx = np.diff(areas, prepend=areas[0])
    ddx = np.diff(dx, prepend=dx[0])

    # Compute median absolute deviation (MAD) for the derivatives
    mad_dx = np.nanmedian(np.abs(dx - np.nanmedian(dx)))
    mad_ddx = np.nanmedian(np.abs(ddx - np.nanmedian(ddx)))

    # Set MAD threshold for the derivatives
    dx_thresh  = np.nanmedian(np.abs(dx))  + 6 * mad_dx
    ddx_thresh = np.nanmedian(np.abs(ddx)) + 6 * mad_ddx
    mask = (np.abs(dx) > dx_thresh) & (np.abs(ddx) > ddx_thresh)

    # Mask areas that pass the threshold
    areas_filtered = areas.copy()
    areas_filtered[mask] = np.nan

    # Interpolate
    to_interp = np.where(np.isnan(areas_filtered))[0]
    known_x = np.where(~np.isnan(areas_filtered))[0]
    known_y = areas_filtered[known_x]
    areas_interp = areas_filtered.copy()
    areas_interp[to_interp] = np.interp(to_interp, known_x, known_y)

    return areas_interp




def find_norm_pupil_areas(
    laser_on_df
):
    """
    Compute normalised pupil areas.
    """
    areas = []
    for _, row in laser_on_df.iterrows():
        # Build point array: (N, 2)
        pts = np.array(
            [[row[f'b{n+1}_x'], row[f'b{n+1}_y']] for n in range(6)], dtype=np.float32
        )
        # Fit ellipse: ((cx, cy), (major, minor), angle)
        (center, axes, angle) = cv2.fitEllipse(pts)
        # axes = (width, height) → semi-axes = /2
        a = axes[0] / 2
        b = axes[1] / 2
        # Ellipse area
        area = np.pi * a * b
        areas.append(area)

    # Filter drastic changes due to blinking
    areas = np.array(areas)
    areas = filter_and_interpolate_areas(areas)
    # Find max area and normalise
    max_area = np.nanmax(areas)
    norm_areas = areas / max_area

    return areas, norm_areas



def get_pupil_data(
    filepath,
    stimuli_blocks,
    pupil_fps,
    plot: bool = True
):
    """
    Get pupil data.
    """
    # Load data
    DLC_df = load_DLC_df(filepath, pupil_fps)
    # Crop for laser-ON frames only
    laser_on_df = filter_high_likelihood_frames(DLC_df, pupil_fps)
    # Label frames with stimuli
    laser_on_df = add_stimuli_label(laser_on_df, stimuli_blocks, pupil_fps)
    # Compute normalised pupil areas
    areas, norm_areas = find_norm_pupil_areas(laser_on_df)
    # Output
    pupil_df = pd.DataFrame({
        'frames': laser_on_df.index,
        'stimulus': laser_on_df['stimulus'],
        'area': areas,
        'norm_area': norm_areas,
    })
    pupil_df.reset_index(drop=True, inplace=True)

    # Plot
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(pupil_df.index, pupil_df['norm_area'], label='Normalized pupil area')

        stimulus = pupil_df['stimulus'].values
        index = pupil_df.index.values  # actual row indices

        start_idx = 0
        current_label = stimulus[0]

        for i in range(1, len(stimulus) + 1):  # +1 to handle last block
            if i == len(stimulus) or stimulus[i] != current_label:
                # Shade only black or grey
                if current_label is not None:
                    if 'black' in current_label:
                        plt.axvspan(index[start_idx], index[i - 1], color='gray', alpha=0.6)
                    elif 'grey' in current_label:
                        plt.axvspan(index[start_idx], index[i - 1], color='gray', alpha=0.2)
                # Update start and label
                if i < len(stimulus):
                    start_idx = i
                    current_label = stimulus[i]

        plt.xlabel('Frame (Laser ON)')
        plt.ylabel('Normalized pupil area')
        plt.ylim(0,1.1)

        filename = os.path.basename(filepath)
        rec = filename.split('DLC_mobnet')[0]
        plt.title(f'Normalised pupil area in Recording/Trial {rec}')
        plt.tight_layout()
        plt.show()

    return pupil_df



def load_and_resample_deltaf(
    filepath,
    deltaf_fps,
    pupil_fps,
    plot: bool = True
):
    """
    Load and downsample delta F.
    """
    # Load data
    calcium_data = sio.loadmat(filepath)
    deltaf = calcium_data['deltaF']
    # Remove first 2 "neurons" which are background and neuropil
    deltaf = deltaf[2:]

    frac = Fraction(pupil_fps, deltaf_fps) # 30/40 -> 3/4
    up = frac.numerator
    down = frac.denominator
    # Downsample by neuron
    deltaf_resampled = np.array([resample_poly(neuron, up, down) for neuron in deltaf])

    print(f'Original shape: {deltaf.shape}, Resampled shape: {deltaf_resampled.shape}')


    # Randomly select 6 neurons for plots
    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharey=True)
        axes = axes.flatten()  # Flatten to easily iterate
        max_frames = 1000  # Only plot the first 1000 frames
        neuron_indices = np.random.choice(deltaf.shape[0], 6, replace=False)

        for i, neuron_idx in enumerate(neuron_indices):
            original_frames = np.arange(min(max_frames, deltaf.shape[1]))
            resampled_frames = np.linspace(0, deltaf.shape[1]-1, deltaf_resampled.shape[1])
            resampled_frames = resampled_frames[resampled_frames < max_frames]
            
            axes[i].plot(original_frames, deltaf[neuron_idx, :len(original_frames)], label='Original 40Hz', alpha=0.7)
            axes[i].plot(resampled_frames, deltaf_resampled[neuron_idx, :len(resampled_frames)], label='Resampled 30Hz', alpha=0.7)
            axes[i].set_title(f'Neuron {neuron_idx}', fontsize=10)
            axes[i].set_ylabel('ΔF/F', fontsize=8)
            axes[i].legend(fontsize=8, loc='upper right')
            axes[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Hide x-axis numbers

        plt.tight_layout()
        plt.show()

    return deltaf_resampled



def apply_filter(
    data, 
    fps, 
    cutoff_freq # [.01, .2] in original notebook
):
    """
    Equivalent to load_pupil and load_brain in the original notebook.
    """
    # Bandpass filter
    sos = butter(1, cutoff_freq, btype='bandpass', output='sos', fs=fps)
    filtered = sosfiltfilt(sos, data, axis=0)
    # Standard scaler
    scaled = StandardScaler(with_std=False).fit_transform(filtered)
    return scaled



def load_and_preprocess_data(
    pupil_filepath,
    deltaf_filepath,
    stimuli_blocks,
    pupil_fps,
    deltaf_fps,
    cutoff_freq,
    plot: bool = True
):
    """
    Load and preprocess pupil and delta F data.
    """
    # Loading data
    print("Loading pupil...")
    pupil_df = get_pupil_data(pupil_filepath, stimuli_blocks, pupil_fps, plot)
    pupil = np.array(pupil_df['norm_area'])
    print("\nLoading deltaF...")
    deltaf = load_and_resample_deltaf(deltaf_filepath, deltaf_fps, pupil_fps, plot)

    # Reshape and resize
    print("\nReshaping and truncating data...")
    pupil = pupil.reshape(-1, 1)    # frames x 1
    deltaf = deltaf.T               # frames x neurons
    num_frames = np.sum([v for v in stimuli_blocks.values()]) # only stimuli frames
    pupil = pupil[:num_frames]
    deltaf = deltaf[:num_frames]
    print("pupil:", pupil.shape)
    print("deltaf:", deltaf.shape)

    # Apply filter to data (equivalent to load_pupil and load_brain in original notebook)
    print("\nApplying bandpass filter...")
    FPS = deltaf_fps  # both are now at pupil fps
    pupil_filtered = apply_filter(pupil, FPS, cutoff_freq)
    deltaf_filtered = apply_filter(deltaf, FPS, cutoff_freq)
    print("pupil:", pupil_filtered.shape)
    print("deltaf:", deltaf_filtered.shape)

    # Add NaN padding to the end
    print("\nAdding NaN padding...")
    num_neurons = deltaf_filtered.shape[1]
    pupil_padded = np.append(pupil_filtered, np.full((1, 1), np.nan), axis=0)
    deltaf_padded = np.append(deltaf_filtered, np.full((1, num_neurons), np.nan), axis=0)
    print("pupil:", pupil_padded.shape)
    print("deltaf:", deltaf_padded.shape)

    # Plot
    if plot:
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(deltaf)
        axes[0].set_title("deltaF")
        axes[1].plot(deltaf_filtered)
        axes[1].set_title("deltaF filtered")
        axes[2].plot(pupil)
        axes[2].set_title("pupil")
        axes[3].plot(pupil_filtered)
        axes[3].set_title("pupil_filtered")

        duration = [v for v in stimuli_blocks.values()]
        edges = np.cumsum([0] + duration)
        intervals = list(zip(edges[:-1], edges[1:]))

        for ax in axes:
            for i, (start, end) in enumerate(intervals):
                if i in [0, 4]:
                    ax.axvspan(start, end, color='gray', alpha=0.6)
                elif i in [1, 3]:
                    ax.axvspan(start, end, color='gray', alpha=0.2)

        plt.tight_layout()
        plt.show()

    return pupil_padded, deltaf_padded