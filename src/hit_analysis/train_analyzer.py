import uproot
import numpy as np
import hist
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import traceback
from collections import Counter
from matplotlib.colors import LogNorm

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
    
    # Start with a copy of the first train's results structure
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
                                     calo_hit_time_def=0, energy_thresholds=None,nlayer_batch=1):
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
    """
    from src.geometry_parsing.k4geo_parsers import parse_detector_constants
    from src.geometry_parsing.geometry_info import get_geometry_info
    from src.hit_analysis.occupancy import analyze_detector_hits
    
    # Split seeds into trains
    trains = split_seeds_into_trains(all_seeds, bunches_per_train)
    print(f"Split {len(all_seeds)} seeds into {len(trains)} trains of {bunches_per_train} bunches each")
    
    # Create event trees by train
    events_trees_by_train = []
    for train_idx, train_seeds in enumerate(trains):
        print(f"Loading train {train_idx+1}/{len(trains)}...")
        train_trees = [uproot.open(base_path + filename_pattern.format(seed) + ':events') 
                     for seed in train_seeds]
        events_trees_by_train.append(train_trees)
    
    # Analyze each detector
    for detector_name, xml_file in detectors_to_analyze:
        print(f"\nAnalyzing {detector_name} with train averaging...")
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
            
            # Define thresholds to analyze
            hit_thresholds = [1, 2, 3, 4]
            
            # Analyze each train separately
            train_results = []
            for train_idx, train_trees in enumerate(events_trees_by_train):
                print(f"Processing train {train_idx+1}/{len(events_trees_by_train)}...")
                
                # Analyze this train
                train_stats = analyze_detector_hits(
                    train_trees, detector_name, detector_config, 
                    hit_thresholds, xml_file, constants, main_xml, 
                    remove_zeros, time_cut, calo_hit_time_def, energy_thresholds
                )
                
                train_results.append(train_stats)
            
            # Average results across all trains
            stats = average_train_results(train_results)
            
            # Add train info to stats for plotting
            stats['train_info'] = {
                'bunches_per_train': bunches_per_train,
                'num_trains': len(trains)
            }
            
            # Create visualizations with train-averaged data
            plot_train_averaged_occupancy_analysis(stats, geometry_info, 
                                                output_prefix=f"{detector_name}_train{bunches_per_train}",nlayer_batch=nlayer_batch)
            
            plot_train_averaged_timing_analysis(stats, geometry_info, 
                                             output_prefix=f"{detector_name}_train{bunches_per_train}")
            
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
    
    return events_trees_by_train

def plot_train_averaged_occupancy_analysis(stats, geometry_info, output_prefix=None, nlayer_batch=1):
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

    title_text = f'SiD_o2_v04 - {geometry_info["detector_name"]}'
    if time_cut > 0:
        title_text += f' (t<{time_cut} ns)'

    # Add the C^3 info and use rich text formatting for "Preliminary"
    #title_text += r'                   SiD_o2_v04@C$^{3}$-550 (266 bunches) - $\boldsymbol{\it{Preliminary}}$'
    #title_text += r'                   C$^{3}$ 550 - s.u. (150 bunches) - $\boldsymbol{\it{Preliminary}}$'
    title_text += r'                   C$^{3}$ 250 - s.u. (266 bunches) - $\boldsymbol{\it{Preliminary}}$'

    fig.suptitle(title_text, fontsize=24)

    gs = plt.GridSpec(2, 2)
    
    # 1. Occupancy vs threshold for each layer
    ax1 = fig.add_subplot(gs[0, 0])
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
        ax1.errorbar(
            thresholds, 
            avg_occupancies, 
            yerr=occupancy_errors,
            fmt='o-', 
            label=batch_label,
            capsize=3
        )
    
    ax1.set_xlabel('Buffer depth', fontsize=18)
    ax1.set_ylabel('Occupancy', fontsize=18)
    ax1.set_yscale('log')
    # Set x-axis to display only integer values
    ax1.set_xticks(thresholds)
    ax1.set_xticklabels([str(int(t)) for t in thresholds])
    ax1.grid(True)
    ax1.legend()
    
    # 2. R-Phi hit distribution
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    # Convert awkward arrays to numpy arrays
    phi_vals = ak.to_numpy(stats['positions']['phi'])
    r_vals = ak.to_numpy(stats['positions']['r'])

    # Create the edges first
    hist, xedges, yedges = np.histogram2d(phi_vals, r_vals, bins=[51, 21])


    # Calculate bin sizes
    dphi = xedges[1] - xedges[0]
    dr = yedges[1] - yedges[0]

    z_length = 0
    for layer_id, layer_info in geometry_info['layers'].items():
        if 'z_length' in layer_info:
            z_length = max(z_length, layer_info['z_length'])
    

    r_centers = 0.5 * (yedges[1:] + yedges[:-1])

    bin_areas = np.zeros((len(xedges)-1, len(yedges)-1))
    for j in range(len(r_centers)):
        bin_areas[:, j] = r_centers[j] * z_length * dphi

    # Calculate cylindrical surface area: r × dphi × Z_length
    #bin_areas = r_centers * dphi * z_length

    hist_normalized = hist / (bin_areas + 1e-10)
    print("dphi = ",dphi,",  z_length = ",z_length,"mm ,  bin_area[0,0] = ",bin_areas[0,0],"mm²")

    pcm = ax2.pcolormesh(xedges[:-1], yedges[:-1], hist_normalized.T, 
                         shading='auto',cmap='viridis',
                         vmin=0,
                         edgecolors='none')
    ax2.set_facecolor('#3F007D')
    ax2.set_ylabel('R [mm]',fontsize=18)
    ax2.set_xlabel('Phi [rad]',fontsize=18)
    #ax2.set_title('Hit Distribution (R-Phi)',fontsize=20)
    plt.colorbar(pcm, ax=ax2, label='Hits/mm²')

    # 3. R-Z hit distribution
    ax3 = fig.add_subplot(gs[1, :])
    # Convert awkward arrays to numpy arrays
    z_vals = ak.to_numpy(stats['positions']['z'])
    r_vals = ak.to_numpy(stats['positions']['r'])
    hist, xedges, yedges = np.histogram2d(z_vals, r_vals, bins=[100, 30])
    
    # Calculate bin sizes for normalization
    # Get the center of each radial bin
    r_centers = 0.5 * (yedges[1:] + yedges[:-1])
    dz = xedges[1] - xedges[0]

    cylindrical_areas = np.outer(np.ones(len(xedges)-1), 2*np.pi*r_centers*dz)  # Area in mm²

    dr = yedges[1] - yedges[0]

    bin_areas = np.zeros((len(xedges)-1, len(yedges)-1))
    for i in range(len(xedges)-1):
        for j in range(len(yedges)-1):
            bin_areas[i, j] = dz * dr
  
    #hist_normalized = hist / (bin_areas + 1e-10)
    hist_normalized = hist / cylindrical_areas

    print("zmin = ",np.min(z_vals), " mm, zmax = ", np.max(z_vals),"mm , dz,", dz, " mm, rmin = ", np.min(r_vals), " mm, rmax = ", np.max(r_vals), " mm, dr = ",  dr, " mm, bin area = ", bin_areas[0,0], " mm²")


    pcm = ax3.pcolormesh(xedges[:-1], yedges[:-1], hist_normalized.T, shading='auto')
    ax3.set_xlabel('Z [mm]',fontsize=18)
    ax3.set_ylabel('R [mm]',fontsize=18)
    #ax3.set_title('Hit Distribution (R-Z)',fontsize=20)
    plt.colorbar(pcm, ax=ax3, label='Hits/mm²')
    




    # Add a note about train averaging
    # note_text = f"Note: Occupancy values represent the average across {num_trains} trains with {bunches_per_train} bunches each."
    # note_text += f"\nHit distributions shown are from a sample train for visualization purposes."
    # plt.figtext(0.5, 0.02, note_text, ha='center', fontsize=10, 
    #            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()  # Adjust for title and note
    
    if output_prefix:
        plt.savefig(f'{output_prefix}_C3_250_occupancy.png', dpi=300)
        plt.savefig(f'{output_prefix}_C3_250_occupancy.pdf')
    
    plt.show()

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


def plot_train_averaged_timing_analysis(stats, geometry_info, output_prefix=None):
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
    
    title = f'Timing Analysis for {geometry_info["detector_name"]}'
    subtitle = f'Based on sampling from {num_trains} trains with {bunches_per_train} bunches per train'
    
    if time_cut > 0:
        title += f' (t<{time_cut} ns)'
    
    fig.suptitle(title, fontsize=16)
    plt.figtext(0.5, 0.92, subtitle, ha='center', fontsize=12)
    
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
    r_edges = np.linspace(0, max(r_vals), 21)
    r_bin_width = np.diff(r_edges)[0]  # R bin width in mm

    hist, xedges, yedges = np.histogram2d(r_vals, time_vals,
                            bins=[r_edges, t_edges]) 
    
    # Calculate bin area (mm × ns)
    bin_area = r_bin_width * t_bin_width
    density = hist.T / bin_area

    pcm = ax2.pcolormesh(xedges[:-1], yedges[:-1], density, 
                        shading='auto', 
                        cmap='viridis',
                        norm=LogNorm(vmin=0.001))
    ax2.set_facecolor('#3F007D')
    
    ax2.set_ylabel('Time [ns]')
    ax2.set_xlabel('R [mm]')
    cbar = plt.colorbar(pcm, ax=ax2)
    cbar.set_label('Hit Density (hits/mm·ns)')
    ax2.text(0.02, 0.02, f"Bin size: {r_bin_width:.1f} mm × {t_bin_width:.1f} ns", 
            transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))

    # 3. Timing vs Z (with density)
    ax3 = fig.add_subplot(gs[1, 1])
    z_edges = np.linspace(min(z_vals), max(z_vals), 21)
    z_bin_width = np.diff(z_edges)[0]  # Z bin width in mm

    hist, xedges, yedges = np.histogram2d(z_vals, time_vals, 
                            bins=[z_edges, t_edges]) 
    
    # Calculate bin area (mm × ns)
    bin_area = z_bin_width * t_bin_width
    density = hist.T / bin_area

    pcm = ax3.pcolormesh(xedges[:-1], yedges[:-1], density, 
                        shading='auto', 
                        cmap='viridis',
                        norm=LogNorm(vmin=0.001))
    ax3.set_facecolor('#3F007D')
    ax3.set_xlabel('Z [mm]')
    ax3.set_ylabel('Time [ns]')
    cbar = plt.colorbar(pcm, ax=ax3)
    cbar.set_label('Hit Density (hits/mm·ns)')
    ax3.text(0.02, 0.02, f"Bin size: {z_bin_width:.1f} mm × {t_bin_width:.1f} ns", 
            transform=ax3.transAxes, bbox=dict(facecolor='white', alpha=0.7))

    # 4. Timing vs Phi (with density)
    ax4 = fig.add_subplot(gs[0, 1])
    phi_edges = np.linspace(-np.pi, np.pi, 51)
    phi_bin_width = np.diff(phi_edges)[0]  # Phi bin width in radians

    hist, xedges, yedges = np.histogram2d(phi_vals, time_vals, 
                            bins=[phi_edges, t_edges]) 
    
    # Calculate bin area (rad × ns)
    bin_area = phi_bin_width * t_bin_width
    density = hist.T / bin_area

    pcm = ax4.pcolormesh(xedges[:-1], yedges[:-1], density, 
                        shading='auto', 
                        cmap='viridis',
                        norm=LogNorm(vmin=0.001))
    ax4.set_facecolor('#3F007D')
    ax4.set_xlabel('Phi [rad]')
    ax4.set_ylabel('Time [ns]')
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
        plt.savefig(f'{output_prefix}_timing_density.png', dpi=300)
        plt.savefig(f'{output_prefix}_timing_density.pdf')
    
    plt.show()

