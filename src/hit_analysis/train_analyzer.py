import uproot
import numpy as np
import hist
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import traceback
import re
from collections import Counter
from matplotlib.colors import LogNorm
from typing import List

# Import parallel I/O utilities
from src.utils.parallel_io import open_files_parallel
from src.utils.parallel_config import get_optimal_worker_count

# Import parallel train processing utilities
from src.utils.parallel_train_processing import process_trains_parallel
from src.utils.histogram_utils import (
    compute_rphi_area_map,
    compute_rz_area_map,
    extract_layer_geometry,
)
from src.hit_analysis.plotting import _configure_log_yticks, _get_detector_display_name


def _sanitize_for_filename(value: str) -> str:
    """Convert arbitrary labels into filesystem-friendly tokens."""
    sanitized = re.sub(r'[^0-9A-Za-z\-]+', '_', value.strip())
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return sanitized

def split_seeds_into_trains(seeds, bunches_per_train):
    """
    Split a list of seeds into train bunches.
    
    Parameters:
    -----------
    seeds : list
        List of all seed values
    bunches_per_train : int
        Number of bunches (seeds) per train
        
    Returns:
    --------
    list of lists, each representing a train of bunches
    """
    trains = []
    for i in range(0, len(seeds), bunches_per_train):
        train = seeds[i:i + bunches_per_train]
        if len(train) == bunches_per_train:  # Only include complete trains
            trains.append(train)
    return trains


def average_train_results(train_results):
    """
    Average occupancy statistics across multiple trains.
    
    Parameters:
    -----------
    train_results : list of dicts
        List of analysis results for each train
        
    Returns:
    --------
    dict with averaged statistics
    """
    if not train_results:
        return None
    
    # If workers returned bundles (ipc_only/hpp_only/combined), unwrap to average each mode
    if 'ipc_only' in train_results[0] or 'combined' in train_results[0]:
        return {
            'ipc_only': average_train_results([t['ipc_only'] for t in train_results if t.get('ipc_only')]),
            'hpp_only': average_train_results([t['hpp_only'] for t in train_results if t.get('hpp_only')]),
            'combined': average_train_results([t['combined'] for t in train_results if t.get('combined')])
        }

    # Start with a copy of the first train's results structure (plain stats)
    averaged = train_results[0].copy()
    
    # Get common thresholds
    thresholds = list(train_results[0]['threshold_stats'].keys())
    
    # Average threshold statistics
    for threshold in thresholds:
        threshold_stats = averaged['threshold_stats'][threshold]
        
        # Create a set of all layers across all trains
        all_layers = set()
        for train in train_results:
            all_layers.update(train['threshold_stats'][threshold]['per_layer'].keys())
        
        # Initialize a new per_layer dictionary with all layers
        new_per_layer = {}
        
        # Average per-layer statistics
        for layer in all_layers:
            layer_stats = {}
            
            # Get all the stats we need to average
            stats_to_average = ['occupancy', 'mean_hits', 'cells_hit', 'total_hits', 'cells_above_threshold']
            
            for stat in stats_to_average:
                values = []
                for train in train_results:
                    if layer in train['threshold_stats'][threshold]['per_layer']:
                        values.append(train['threshold_stats'][threshold]['per_layer'][layer].get(stat, 0))
                
                if values:
                    # Calculate mean
                    layer_stats[stat] = sum(values) / len(values)

                    # Calculate standard deviation and standard error
                    if len(values) > 1:
                        std_dev = np.std(values, ddof=1)  # Sample standard deviation
                        std_error = std_dev / np.sqrt(len(values))
                        layer_stats[f'{stat}_std_dev'] = std_dev
                        layer_stats[f'{stat}_error'] = std_error
                    else:
                        layer_stats[f'{stat}_std_dev'] = 0
                        layer_stats[f'{stat}_error'] = 0
                else:
                    layer_stats[stat] = 0
                    layer_stats[f'{stat}_std_dev'] = 0
                    layer_stats[f'{stat}_error'] = 0

            
            new_per_layer[layer] = layer_stats
        
        # Replace the per_layer dict with our averaged version
        threshold_stats['per_layer'] = new_per_layer
        
        # Average overall statistics
        for stat in ['overall_cells_hit', 'overall_cells_above_threshold', 'max_hits_per_cell']:
            values = [train['threshold_stats'][threshold].get(stat, 0) for train in train_results]
            if values:
                threshold_stats[stat] = sum(values) / len(values)
                if len(values) > 1:
                    std_dev = np.std(values, ddof=1)
                    std_error = std_dev / np.sqrt(len(values))
                    threshold_stats[f'{stat}_std_dev'] = std_dev
                    threshold_stats[f'{stat}_error'] = std_error
    
    # We're keeping the positions from the first train just for visualization
    # This is ok since we're focusing on averaged occupancy statistics
    
    return averaged

def analyze_detectors_and_plot_by_train(DETECTOR_CONFIGS=None, detectors_to_analyze=None, 
                                     all_seeds=None, bunches_per_train=None, 
                                     main_xml=None, base_path=None, filename_pattern=None,
                                     remove_zeros=True, time_cut=-1, 
                                     calo_hit_time_def=0, energy_thresholds=None,nlayer_batch=1,
                                     hpp_file=None, hpp_mu=None,
                                     scenario_label=None, detector_version=None,
                                     use_vectorized_processing=True,
                                     occupancy_ylim_map=None,
                                     occupancy_scaling_map=None,
                                     train_batch_size=None,
                                     max_train_workers=None):
    """
    Analyze detectors with train-based averaging.
    
    Parameters:
    -----------
    DETECTOR_CONFIGS : dict
        Dictionary of detector configurations
    detectors_to_analyze : list of tuples
        List of (detector_name, xml_file) pairs
    all_seeds : list
        List of all seed values
    bunches_per_train : int
        Number of bunches (seeds) per train
    main_xml : str
        Path to main XML file
    base_path : str
        Base path for root files
    filename_pattern : str
        Pattern for root filenames including {} placeholder for seed
    remove_zeros : bool
        Whether to remove hits with zero positions
    time_cut : float
        Cut on hit time in ns (-1 for no cut)
    calo_hit_time_def : int
        Time definition for calorimeter hits
    energy_thresholds : dict, optional
        Dictionary of energy thresholds
    use_vectorized_processing : bool
        If True, enable vectorized hit processing; otherwise use traditional path
    occupancy_ylim_map : dict, optional
        Optional mapping detector_name -> (ymin, ymax) for occupancy axis control
    occupancy_scaling_map : dict, optional
        Optional mapping detector_name -> multiplicative factor applied to occupancies
    train_batch_size : int, optional
        Number of trains to process per parallel batch (limits open files)
    max_train_workers : int, optional
        Override the worker count passed to process_trains_parallel
    """
    from src.geometry_parsing.k4geo_parsers import parse_detector_constants
    from src.geometry_parsing.geometry_info import get_geometry_info
    from src.hit_analysis.occupancy import analyze_detector_hits
    from src.hit_analysis.occupancy import summarize_stats_over_detector
        
    # Split seeds into trains
    trains = split_seeds_into_trains(all_seeds, bunches_per_train)
    print(f"Split {len(all_seeds)} seeds into {len(trains)} trains of {bunches_per_train} bunches each")
    
    # Note: File opening is now handled within parallel train processing
    # No need to pre-load all files into memory
    
    # Prepare summary accumulation
    summary_rows = []

    # Analyze each detector
    for detector_name, xml_file in detectors_to_analyze:
        print(f"\nAnalyzing {detector_name} with train averaging...")
        scale_factor = 1.0
        if occupancy_scaling_map:
            scale_factor = occupancy_scaling_map.get(detector_name, 1.0)

        try:
            detector_config = DETECTOR_CONFIGS[detector_name]
            constants = parse_detector_constants(main_xml, detector_name)
            geometry_info = get_geometry_info(xml_file, detector_config, constants=constants)
            
            # Print geometry info
            print(f"\nGeometry info for {detector_name}:")
            print(f"Total cells: {geometry_info['total_cells']}")
            for layer, info in sorted(geometry_info['layers'].items()):
                print(f"\nLayer {layer}:")
                for key, value in info.items():
                    if key not in ['cells_per_module', 'module_type']:
                        print(f"  {key}: {value}")
            
            # Define buffer depths to analyze (counts-per-cell thresholds)
            #buffer_depths = [1, 2, 3, 4,5,6,7,8,9,10,11]
            buffer_depths = [1, 2, 3, 4,5,6]
            
            # Use parallel train processing instead of sequential
            results_accum: List = []
            batch_size = train_batch_size or len(trains)
            batch_size = max(1, batch_size)
            total_batches = (len(trains) + batch_size - 1) // batch_size

            for batch_index, batch_start in enumerate(range(0, len(trains), batch_size), start=1):
                batch_trains = trains[batch_start:batch_start + batch_size]
                print(f"Processing {len(batch_trains)} trains (batch {batch_index}/{total_batches}) in parallel for {detector_name}...")

                batch_max_workers = max_train_workers if max_train_workers is not None else min(len(batch_trains), 4)
                batch_max_workers = max(1, min(batch_max_workers, len(batch_trains)))

                batch_results = process_trains_parallel(
                    batch_trains, detector_name, detector_config, buffer_depths,
                    xml_file, constants, main_xml, base_path, filename_pattern,
                    remove_zeros, time_cut, calo_hit_time_def, energy_thresholds,
                    hpp_file=hpp_file, hpp_mu=hpp_mu,
                    use_vectorized=use_vectorized_processing,
                    max_workers=batch_max_workers
                )
                results_accum.extend(batch_results)

            train_results = results_accum
            
            if not train_results:
                print(f"Warning: No successful train results for {detector_name}")
                continue
            
            # Average results across all trains (may be a dict of modes)
            averaged = average_train_results(train_results)

            def _attach_train_info(stats_obj):
                if stats_obj is None:
                    return None
                stats_obj['train_info'] = {
                    'bunches_per_train': bunches_per_train,
                    'num_trains': len(trains)
                }
                return stats_obj

            if isinstance(averaged, dict) and 'combined' in averaged:
                # Process three modes: IPC-only, HPP-only, Combined
                modes = [('IPC', averaged.get('ipc_only')), ('HPP', averaged.get('hpp_only')), ('SUM', averaged.get('combined'))]
                for mode_label, stats in modes:
                    stats = _attach_train_info(stats)
                    if stats is None:
                        continue
                    prefix = f"{detector_name}_train{bunches_per_train}_{mode_label}"
                    background_label = ('IPC+HPP' if mode_label == 'SUM' else mode_label)
                    ylim = None
                    if occupancy_ylim_map:
                        ylim = occupancy_ylim_map.get(detector_name)
                    plot_train_averaged_occupancy_analysis(
                        stats, geometry_info, output_prefix=prefix, nlayer_batch=nlayer_batch,
                        occupancy_ylim=ylim,
                        scenario_label=scenario_label, detector_version=detector_version, background_label=background_label,
                        occupancy_scale=scale_factor
                    )
                    plot_train_averaged_timing_analysis(
                        stats, geometry_info, output_prefix=prefix,
                        scenario_label=scenario_label, detector_version=detector_version, background_label=background_label
                    )

                    # Add summary row for this mode
                    try:
                        summary = summarize_stats_over_detector(stats, geometry_info, threshold=1)
                        summary_rows.append({
                            'detector': detector_name,
                            'mode': mode_label,
                            'total_hits': summary['total_hits'],
                            'total_cells': summary['total_cells'],
                            'mean_occupancy': summary['mean_occupancy'],
                            'scaled_mean_occupancy': summary['mean_occupancy'] * scale_factor
                        })
                    except Exception:
                        pass
            else:
                # Backward compatibility: single stats object
                stats = _attach_train_info(averaged)
                ylim = None
                if occupancy_ylim_map:
                    ylim = occupancy_ylim_map.get(detector_name)
                plot_train_averaged_occupancy_analysis(
                    stats, geometry_info, 
                    output_prefix=f"{detector_name}_train{bunches_per_train}",
                    nlayer_batch=nlayer_batch,
                    occupancy_ylim=ylim,
                    scenario_label=scenario_label,
                    detector_version=detector_version,
                    background_label='IPC',
                    occupancy_scale=scale_factor
                )
                plot_train_averaged_timing_analysis(
                    stats, geometry_info, 
                    output_prefix=f"{detector_name}_train{bunches_per_train}",
                    scenario_label=scenario_label,
                    detector_version=detector_version,
                    background_label='IPC'
                )

                # Add summary row for single-mode case
                try:
                    summary = summarize_stats_over_detector(stats, geometry_info, threshold=1)
                    summary_rows.append({
                        'detector': detector_name,
                        'mode': 'IPC',
                        'total_hits': summary['total_hits'],
                        'total_cells': summary['total_cells'],
                        'mean_occupancy': summary['mean_occupancy'],
                        'scaled_mean_occupancy': summary['mean_occupancy'] * scale_factor
                    })
                except Exception:
                    pass
            
            # Print detailed statistics for thresholds
            for threshold in range(1, 5):  # Keep output manageable
                print(f"\nTrain-averaged occupancy statistics (threshold={threshold}):")
                
                for layer, info in sorted(stats['threshold_stats'][threshold]['per_layer'].items()):
                    if layer in geometry_info['layers']:
                        print(f"\nLayer {layer}:")
                        print(f"  Cells hit: {info['cells_hit']:.1f}")
                        print(f"  Total hits: {info['total_hits']:.1f}")
                        print(f"  Occupancy: {info['occupancy']*100:.10f}%")
                        print(f"  Mean hits per hit cell: {info['mean_hits']:.10f}")
            
        except Exception as e:
            print(f"Error processing {detector_name}: {str(e)}")
            traceback.print_exc()
    
    # Print terminal summary table
    if summary_rows:
        try:
            header = (
                f"{'Detector':<18} {'Mode':<6} {'Total Hits/train':>18} "
                f"{'Total Cells':>14} {'Mean Occ.':>12} "
                f"{'Mean Occ. (w/ safety factor,cluster size)':>41}"
            )
            print("\nSummary per subdetector (per train):")
            print(header)
            print('-' * len(header))
            for row in summary_rows:
                det = row['detector']
                mode = row['mode']
                th = row['total_hits']
                tc = row['total_cells']
                mo = row['mean_occupancy']
                smo = row.get('scaled_mean_occupancy', mo)
                print(f"{det:<18} {mode:<6} {th:18.3f} {tc:14d} {mo:12.3e} {smo:41.3e}")
        except Exception:
            pass

    # Return summary of processing results
    return {
        'trains_processed': len(trains),
        'bunches_per_train': bunches_per_train,
        'total_seeds': len(all_seeds)
    }

def plot_train_averaged_occupancy_analysis(stats, geometry_info, output_prefix=None, nlayer_batch=1,
                                           scenario_label=None, detector_version=None, background_label='IPC',
                                           occupancy_ylim=None, occupancy_scale=1.0):
    """
    Create detailed visualizations of the train-averaged occupancy analysis
    
    Parameters:
    -----------
    stats : dict
        Statistics from average_train_results
    geometry_info : dict
        Geometry information
    output_prefix : str, optional
        If provided, save plots with this prefix
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    plt.style.use(hep.style.CMS)

    # Get train info
    bunches_per_train = stats.get('train_info', {}).get('bunches_per_train', 0)
    num_trains = stats.get('train_info', {}).get('num_trains', 0)
    

    # Retrieve assumed time cut for occupancy calculation
    time_cut = stats['time_cut']
    print(f"Time cut: {time_cut}")

    # Left/right titles matching requested style
    left_title = None
    if scenario_label is not None:
        left_title = f"{scenario_label} ({bunches_per_train} bunches/train)"
    else:
        left_title = f"({bunches_per_train} bunches/train)"
    detector_label = _get_detector_display_name(geometry_info["detector_name"])
    right_title = f"{detector_version or 'SiD_o2_v04'} - {detector_label} - {background_label}"
    time_note = (f" (t<{time_cut} ns)" if time_cut > 0 else "")
    fig.text(0.01, 0.98, left_title + time_note, ha='left', va='top', fontsize=16)
    fig.text(0.99, 0.98, right_title, ha='right', va='top', fontsize=16)

    gs = plt.GridSpec(2, 2)
    
    # 1. Occupancy vs threshold for each layer
    ax1 = fig.add_subplot(gs[0, 0])
    scale = occupancy_scale if occupancy_scale is not None else 1.0
    series_data = []
    thresholds = sorted(stats['threshold_stats'].keys())


    # Get all available layers and sort them
    all_layers = sorted(geometry_info['layers'].keys())
    
    # Create layer batches
    layer_batches = []
    for i in range(0, len(all_layers), nlayer_batch):
        batch = all_layers[i:i+nlayer_batch]
        if batch:  # Skip empty batches
            layer_batches.append(batch)
    
    # Plot averaged occupancy for each batch
    for i, batch in enumerate(layer_batches):
        # Label for the batch
        if len(batch) == 1:
            batch_label = f'Layer {batch[0]}'
        else:
            batch_label = f'Layers {batch[0]}-{batch[-1]}'
        
        # Calculate average occupancy and errors for each threshold
        avg_occupancies = []
        occupancy_errors = []
        
        for t in thresholds:
            # Collect occupancies and errors for all layers in batch
            batch_occupancies = []
            batch_errors = []
            
            for layer in batch:
                layer_data = stats['threshold_stats'][t]['per_layer'].get(layer, {})
                if 'occupancy' in layer_data:
                    batch_occupancies.append(layer_data['occupancy'])
                    
                    # Use stored error if available, otherwise calculate simple error
                    if 'occupancy_error' in layer_data:
                        batch_errors.append(layer_data['occupancy_error'])
                    elif 'occupancy_std_dev' in layer_data:
                        batch_errors.append(layer_data['occupancy_std_dev'] / np.sqrt(num_trains))
                    else:
                        batch_errors.append(0)
            
            # Calculate batch average and error
            if batch_occupancies:
                # Average occupancy across layers in batch
                avg_occ = sum(batch_occupancies) / len(batch_occupancies)
                avg_occupancies.append(avg_occ)
                
                # Combine errors (root sum of squares for independent measurements)
                if batch_errors:
                    # Root sum of squares, divided by number of layers to maintain average scale
                    combined_error = np.sqrt(sum(e**2 for e in batch_errors)) / len(batch_errors)
                    occupancy_errors.append(combined_error)
                else:
                    occupancy_errors.append(0)
            else:
                avg_occupancies.append(0)
                occupancy_errors.append(0)
        
        # Plot with error bars
        scaled_avg = [val * scale for val in avg_occupancies]
        scaled_err = [err * scale for err in occupancy_errors]

        ax1.errorbar(
            thresholds, 
            scaled_avg, 
            yerr=scaled_err,
            fmt='o-', 
            label=batch_label,
            capsize=3
        )
        series_data.append((batch_label, scaled_avg, scaled_err))
    
    ax1.set_xlabel('Buffer depth', fontsize=18)
    ax1.set_ylabel('Layer occupancy', fontsize=18)
    ax1.set_yscale('log')
    _configure_log_yticks(ax1)
    # Set x-axis to display only integer values
    ax1.set_xticks(thresholds)
    ax1.set_xticklabels([str(int(t)) for t in thresholds])
    ax1.grid(True)
    if occupancy_ylim is not None:
        try:
            ax1.set_ylim(occupancy_ylim)
        except ValueError:
            pass
    ax1.legend()
    
    # 2. R-Phi hit distribution
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    # Convert awkward arrays to numpy arrays
    phi_vals = ak.to_numpy(stats['positions']['phi'])
    r_vals = ak.to_numpy(stats['positions']['r'])

    # Create the edges first
    hist, xedges, yedges = np.histogram2d(phi_vals, r_vals, bins=[51, 21])

    layer_metadata = extract_layer_geometry(geometry_info)
    area_map = compute_rphi_area_map(layer_metadata, xedges, yedges)
    with np.errstate(invalid='ignore', divide='ignore'):
        hist_normalized = np.divide(hist, area_map, where=area_map > 0)

    # Mask zero or negative bins so they appear as white
    masked = np.ma.masked_where(hist_normalized.T <= 0, hist_normalized.T)
    cmap2 = plt.cm.get_cmap('viridis').copy()
    cmap2.set_bad(color='white')
    # Guard against all-masked (no data) to avoid colorbar LogNorm issues
    if masked.mask.all() if hasattr(masked.mask, 'all') else False:
        ax2.text(0.5, 0.5, 'No data', transform=ax2.transAxes, ha='center', va='center')
        pcm = None
    else:
        pcm = ax2.pcolormesh(xedges[:-1], yedges[:-1], masked,
                             shading='auto', cmap=cmap2,
                             edgecolors='none')
    # Set background to white so masked regions show white
    ax2.set_facecolor('white')
    ax2.set_ylabel('R [mm]',fontsize=18)
    ax2.set_xlabel('Phi [rad]',fontsize=18)
    #ax2.set_title('Hit Distribution (R-Phi)',fontsize=20)
    if pcm is not None:
        plt.colorbar(pcm, ax=ax2, label='Hits/mm²')

    # 3. R-Z hit distribution
    ax3 = fig.add_subplot(gs[1, :])
    # Convert awkward arrays to numpy arrays
    z_vals = ak.to_numpy(stats['positions']['z'])
    r_vals = ak.to_numpy(stats['positions']['r'])
    hist, xedges, yedges = np.histogram2d(z_vals, r_vals, bins=[100, 30])
    
    area_map_rz = compute_rz_area_map(layer_metadata, xedges, yedges)
    with np.errstate(invalid='ignore', divide='ignore'):
        hist_normalized = np.divide(hist, area_map_rz, where=area_map_rz > 0)

    # Mask zero or negative bins to show white
    masked_rz = np.ma.masked_where(hist_normalized.T <= 0, hist_normalized.T)
    cmap3 = plt.cm.get_cmap('viridis').copy()
    cmap3.set_bad(color='white')
    if masked_rz.mask.all() if hasattr(masked_rz.mask, 'all') else False:
        ax3.text(0.5, 0.5, 'No data', transform=ax3.transAxes, ha='center', va='center')
        pcm = None
    else:
        pcm = ax3.pcolormesh(xedges[:-1], yedges[:-1], masked_rz, shading='auto', cmap=cmap3)
    ax3.set_xlabel('Z [mm]',fontsize=18)
    ax3.set_ylabel('R [mm]',fontsize=18)
    #ax3.set_title('Hit Distribution (R-Z)',fontsize=20)
    if pcm is not None:
        plt.colorbar(pcm, ax=ax3, label='Hits/mm²')
    




    # Add a note about train averaging
    # note_text = f"Note: Occupancy values represent the average across {num_trains} trains with {bunches_per_train} bunches each."
    # note_text += f"\nHit distributions shown are from a sample train for visualization purposes."
    # plt.figtext(0.5, 0.02, note_text, ha='center', fontsize=10, 
    #            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    fig.tight_layout()  # Adjust for title and note

    # Create occupancy-only figure
    fig_occ, ax_occ = plt.subplots(figsize=(7, 5))
    # Replicate condensed headline without bunch-count parenthetical
    compact_left = left_title
    if bunches_per_train:
        compact_left = compact_left.replace(f" ({bunches_per_train} bunches/train)", "")
        compact_left = compact_left.replace(f"({bunches_per_train} bunches/train)", "")
    compact_left = compact_left.strip()
    fig_occ.text(0.01, 0.98, compact_left + time_note, ha='left', va='top', fontsize=16)
    fig_occ.text(0.99, 0.98, right_title, ha='right', va='top', fontsize=16)

    for label, vals, errs in series_data:
        ax_occ.errorbar(
            thresholds,
            vals,
            yerr=errs,
            fmt='o-',
            label=label,
            capsize=3
        )
    ax_occ.set_xlabel('Buffer depth', fontsize=18)
    ax_occ.set_ylabel('Layer occupancy', fontsize=18)
    ax_occ.set_yscale('log')
    _configure_log_yticks(ax_occ)
    ax_occ.set_xticks(thresholds)
    ax_occ.set_xticklabels([str(int(t)) for t in thresholds])
    ax_occ.grid(True)
    ax_occ.legend()
    fig_occ.tight_layout(rect=(0, 0, 1, 0.95))

    if output_prefix:
        if scenario_label:
            scenario_token = _sanitize_for_filename(scenario_label)
            base_path = f"{output_prefix}_{scenario_token}_occupancy"
        else:
            base_path = f"{output_prefix}_C3_250_occupancy"

        fig.savefig(f'{base_path}.png', dpi=300)
        fig.savefig(f'{base_path}.pdf')
        fig_occ.savefig(f'{base_path}_vs_buffer_only.png', dpi=300)
        fig_occ.savefig(f'{base_path}_vs_buffer_only.pdf')
    
    plt.show()
    plt.close(fig)
    plt.close(fig_occ)

# def plot_train_averaged_timing_analysis(stats, geometry_info, output_prefix=None):
#     """
#     Create timing analysis visualizations for train-averaged results
    
#     Parameters:
#     -----------
#     stats : dict
#         Statistics from average_train_results
#     geometry_info : dict
#         Geometry information
#     output_prefix : str, optional
#         If provided, save plots with this prefix
#     """
#     # Create figure with multiple subplots
#     fig = plt.figure(figsize=(15, 10))
#     plt.style.use(hep.style.CMS)

#     # Get train info
#     bunches_per_train = stats.get('train_info', {}).get('bunches_per_train', 0)
#     num_trains = stats.get('train_info', {}).get('num_trains', 0)
    
#     # Retrieve assumed time cut for occupancy calculation
#     time_cut = stats.get('time_cut', -1)
    
#     title = f'Timing Analysis for {geometry_info["detector_name"]}'
#     subtitle = f'Based on sampling from {num_trains} trains with {bunches_per_train} bunches per train'
    
#     if time_cut > 0:
#         title += f' (t<{time_cut} ns)'
    
#     fig.suptitle(title, fontsize=16)
#     plt.figtext(0.5, 0.92, subtitle, ha='center', fontsize=12)
    
#     gs = plt.GridSpec(2, 2)

#     # Convert awkward arrays to numpy if needed
#     if hasattr(stats['times'], 'to_numpy'):
#         time_vals = ak.to_numpy(stats['times'])
#         r_vals = ak.to_numpy(stats['positions']['r'])
#         z_vals = ak.to_numpy(stats['positions']['z'])
#         phi_vals = ak.to_numpy(stats['positions']['phi'])
#     else:
#         time_vals = stats['times']
#         r_vals = stats['positions']['r']
#         z_vals = stats['positions']['z']
#         phi_vals = stats['positions']['phi']

#     t_edges = np.linspace(0, min(100, max(time_vals)), 100)

#     # 1. Timing distribution
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax1.hist(time_vals, bins=100, range=(0, min(100, max(time_vals))), 
#             histtype='step', color='blue', linewidth=1.5)
#     ax1.set_xlabel('Time [ns]')
#     ax1.set_ylabel('Hits')
#     ax1.set_title('Timing Distribution (Sample Train)')
#     ax1.grid(True, alpha=0.3)

#     # 2. Timing vs R
#     ax2 = fig.add_subplot(gs[1, 0])
#     r_edges = np.linspace(0, max(r_vals), 21)

#     hist, xedges, yedges = np.histogram2d(r_vals, time_vals,
#                             bins=[r_edges, t_edges]) 

#     pcm = ax2.pcolormesh(xedges[:-1], yedges[:-1], hist.T, 
#                         shading='auto', 
#                         cmap='viridis',
#                         norm=LogNorm(vmin=1))
#     ax2.set_facecolor('#3F007D')
    
#     ax2.set_ylabel('Time [ns]')
#     ax2.set_xlabel('R [mm]')
#     cbar = plt.colorbar(pcm, ax=ax2)
#     cbar.set_label('Hits')

#     # 3. Timing vs Z
#     ax3 = fig.add_subplot(gs[1, 1])
#     z_edges = np.linspace(min(z_vals), max(z_vals), 21)

#     hist, xedges, yedges = np.histogram2d(z_vals, time_vals, 
#                             bins=[z_edges, t_edges]) 

#     pcm = ax3.pcolormesh(xedges[:-1], yedges[:-1], hist.T, 
#                         shading='auto', 
#                         cmap='viridis',
#                         norm=LogNorm(vmin=1))
#     ax3.set_facecolor('#3F007D')
#     ax3.set_xlabel('Z [mm]')
#     ax3.set_ylabel('Time [ns]')
#     cbar = plt.colorbar(pcm, ax=ax3)
#     cbar.set_label('Hits')

#     # 4. Timing vs Phi
#     ax4 = fig.add_subplot(gs[0, 1])
#     phi_edges = np.linspace(-np.pi, np.pi, 51)

#     hist, xedges, yedges = np.histogram2d(phi_vals, time_vals, 
#                             bins=[phi_edges, t_edges]) 

#     pcm = ax4.pcolormesh(xedges[:-1], yedges[:-1], hist.T, 
#                         shading='auto', 
#                         cmap='viridis',
#                         norm=LogNorm(vmin=1))
#     ax4.set_facecolor('#3F007D')
#     ax4.set_xlabel('Phi [rad]')
#     ax4.set_ylabel('Time [ns]')
#     cbar = plt.colorbar(pcm, ax=ax4)
#     cbar.set_label('Hits')

#     # Add a note about train sampling
#     note_text = f"Note: Timing distributions shown are from a sample train for visualization purposes."
#     note_text += f"\nOccupancy calculations are averaged across all {num_trains} trains."
#     plt.figtext(0.5, 0.02, note_text, ha='center', fontsize=10, 
#                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
#     plt.tight_layout(rect=[0, 0.05, 1, 0.9])  # Adjust for title and note

#     if output_prefix:
#         plt.savefig(f'{output_prefix}_timing.png', dpi=300)
#         plt.savefig(f'{output_prefix}_timing.pdf')
    
#     plt.show()


def plot_train_averaged_timing_analysis(stats, geometry_info, output_prefix=None,
                                        scenario_label=None, detector_version=None, background_label='IPC'):
    """
    Create timing analysis visualizations with hit density (hits/mm²) for train-averaged results
    
    Parameters:
    -----------
    stats : dict
        Statistics from average_train_results
    geometry_info : dict
        Geometry information
    output_prefix : str, optional
        If provided, save plots with this prefix
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    plt.style.use(hep.style.CMS)

    # Get train info
    bunches_per_train = stats.get('train_info', {}).get('bunches_per_train', 0)
    num_trains = stats.get('train_info', {}).get('num_trains', 0)
    
    # Retrieve assumed time cut for occupancy calculation
    time_cut = stats.get('time_cut', -1)
    
    # Top-left and top-right titles with scenario and background type
    left_title = None
    if scenario_label is not None:
        left_title = f"{scenario_label} ({bunches_per_train} bunches/train)"
    else:
        left_title = f"({bunches_per_train} bunches/train)"
    detector_label = _get_detector_display_name(geometry_info["detector_name"])
    right_title = f"{detector_version or 'SiD_o2_v04'} - {detector_label} - {background_label}"
    time_note = (f" (t<{time_cut} ns)" if time_cut > 0 else "")
    fig.text(0.01, 0.98, left_title + time_note, ha='left', va='top', fontsize=16)
    fig.text(0.99, 0.98, right_title, ha='right', va='top', fontsize=16)
    
    gs = plt.GridSpec(2, 2)

    # Convert awkward arrays to numpy if needed
    if hasattr(stats['times'], 'to_numpy'):
        time_vals = ak.to_numpy(stats['times'])
        r_vals = ak.to_numpy(stats['positions']['r'])
        z_vals = ak.to_numpy(stats['positions']['z'])
        phi_vals = ak.to_numpy(stats['positions']['phi'])
    else:
        time_vals = stats['times']
        r_vals = stats['positions']['r']
        z_vals = stats['positions']['z']
        phi_vals = stats['positions']['phi']

    if len(time_vals) == 0:
        t_edges = np.linspace(0, 1, 100)
    else:
        t_edges = np.linspace(0, min(100, max(time_vals)), 100)
    t_bin_width = np.diff(t_edges)[0]  # Time bin width in ns

    # 1. Timing distribution
    ax1 = fig.add_subplot(gs[0, 0])
    hist, edges = np.histogram(time_vals, bins=100, range=(0, min(100, max(time_vals))))
    bin_width = edges[1] - edges[0]
    ax1.step(edges[:-1], hist/bin_width, where='post', color='blue', linewidth=1.5)
    ax1.set_xlabel('Time [ns]')
    ax1.set_ylabel('Hit Density (hits/ns)')
    ax1.set_title('Timing Distribution (Sample Train)')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.95, f"Bin width: {bin_width:.1f} ns", 
             transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.7))

    # 2. Timing vs R (with density)
    ax2 = fig.add_subplot(gs[1, 0])
    if len(r_vals) == 0:
        r_edges = np.linspace(0, 1, 21)
    else:
        r_edges = np.linspace(0, max(r_vals), 21)
    r_bin_width = max(np.diff(r_edges)[0], 1e-9)  # R bin width in mm

    if len(r_vals) == 0 or len(time_vals) == 0:
        hist = np.zeros((len(r_edges)-1, len(t_edges)-1))
        xedges, yedges = r_edges, t_edges
    else:
        hist, xedges, yedges = np.histogram2d(r_vals, time_vals,
                                bins=[r_edges, t_edges]) 
    
    # Calculate bin area (mm × ns)
    bin_area = r_bin_width * t_bin_width
    density = hist.T / bin_area

    masked = np.ma.masked_where(density <= 0, density)
    cmap2 = plt.cm.get_cmap('viridis').copy()
    cmap2.set_bad(color='white')
    if hasattr(masked, 'mask') and np.all(masked.mask):
        ax2.text(0.5, 0.5, 'No data', transform=ax2.transAxes, ha='center', va='center')
        pcm = None
    else:
        pcm = ax2.pcolormesh(xedges[:-1], yedges[:-1], masked, 
                            shading='auto', 
                            cmap=cmap2,
                            norm=LogNorm(vmin=0.001))
    ax2.set_facecolor('white')
    
    ax2.set_ylabel('Time [ns]')
    ax2.set_xlabel('R [mm]')
    if pcm is not None:
        cbar = plt.colorbar(pcm, ax=ax2)
        cbar.set_label('Hit Density (hits/mm·ns)')
    ax2.text(0.02, 0.02, f"Bin size: {r_bin_width:.1f} mm × {t_bin_width:.1f} ns", 
            transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))

    # 3. Timing vs Z (with density)
    ax3 = fig.add_subplot(gs[1, 1])
    if len(z_vals) == 0:
        z_edges = np.linspace(0, 1, 21)
    else:
        z_edges = np.linspace(min(z_vals), max(z_vals), 21)
    z_bin_width = max(np.diff(z_edges)[0], 1e-9)  # Z bin width in mm

    if len(z_vals) == 0 or len(time_vals) == 0:
        hist = np.zeros((len(z_edges)-1, len(t_edges)-1))
        xedges, yedges = z_edges, t_edges
    else:
        hist, xedges, yedges = np.histogram2d(z_vals, time_vals, 
                                bins=[z_edges, t_edges]) 
    
    # Calculate bin area (mm × ns)
    bin_area = z_bin_width * t_bin_width
    density = hist.T / bin_area

    masked = np.ma.masked_where(density <= 0, density)
    cmap3 = plt.cm.get_cmap('viridis').copy()
    cmap3.set_bad(color='white')
    if hasattr(masked, 'mask') and np.all(masked.mask):
        ax3.text(0.5, 0.5, 'No data', transform=ax3.transAxes, ha='center', va='center')
        pcm = None
    else:
        pcm = ax3.pcolormesh(xedges[:-1], yedges[:-1], masked, 
                            shading='auto', 
                            cmap=cmap3,
                            norm=LogNorm(vmin=0.001))
    ax3.set_facecolor('white')
    ax3.set_xlabel('Z [mm]')
    ax3.set_ylabel('Time [ns]')
    if pcm is not None:
        cbar = plt.colorbar(pcm, ax=ax3)
        cbar.set_label('Hit Density (hits/mm·ns)')
    ax3.text(0.02, 0.02, f"Bin size: {z_bin_width:.1f} mm × {t_bin_width:.1f} ns", 
            transform=ax3.transAxes, bbox=dict(facecolor='white', alpha=0.7))

    # 4. Timing vs Phi (with density)
    ax4 = fig.add_subplot(gs[0, 1])
    phi_edges = np.linspace(-np.pi, np.pi, 51)
    phi_bin_width = np.diff(phi_edges)[0]  # Phi bin width in radians

    if len(phi_vals) == 0 or len(time_vals) == 0:
        hist = np.zeros((len(phi_edges)-1, len(t_edges)-1))
        xedges, yedges = phi_edges, t_edges
    else:
        hist, xedges, yedges = np.histogram2d(phi_vals, time_vals, 
                                bins=[phi_edges, t_edges]) 
    
    # Calculate bin area (rad × ns)
    bin_area = phi_bin_width * t_bin_width
    density = hist.T / bin_area

    masked = np.ma.masked_where(density <= 0, density)
    cmap4 = plt.cm.get_cmap('viridis').copy()
    cmap4.set_bad(color='white')
    if hasattr(masked, 'mask') and np.all(masked.mask):
        ax4.text(0.5, 0.5, 'No data', transform=ax4.transAxes, ha='center', va='center')
        pcm = None
    else:
        pcm = ax4.pcolormesh(xedges[:-1], yedges[:-1], masked, 
                            shading='auto', 
                            cmap=cmap4,
                            norm=LogNorm(vmin=0.001))
    ax4.set_facecolor('white')
    ax4.set_xlabel('Phi [rad]')
    ax4.set_ylabel('Time [ns]')
    if pcm is not None:
        cbar = plt.colorbar(pcm, ax=ax4)
        cbar.set_label('Hit Density (hits/rad·ns)')
    ax4.text(0.02, 0.02, f"Bin size: {phi_bin_width:.2f} rad × {t_bin_width:.1f} ns", 
            transform=ax4.transAxes, bbox=dict(facecolor='white', alpha=0.7))

    # Add a note about train sampling
    note_text = f"Note: Timing distributions shown are from a sample train for visualization purposes."
    note_text += f"\nOccupancy calculations are averaged across all {num_trains} trains."
    plt.figtext(0.5, 0.02, note_text, ha='center', fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.9])  # Adjust for title and note

    if output_prefix:
        if scenario_label:
            scenario_token = _sanitize_for_filename(scenario_label)
            base_path = f"{output_prefix}_{scenario_token}_timing_density"
        else:
            base_path = f"{output_prefix}_timing_density"

        plt.savefig(f'{base_path}.png', dpi=300)
        plt.savefig(f'{base_path}.pdf')
    
    plt.show()
