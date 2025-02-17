import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import traceback

from src.detector_config import get_detector_configs, get_xmls
from src.geometry_parsing.geometry_info import get_geometry_info

from src.hit_analysis.occupancy import analyze_detector_hits
from src.geometry_parsing.k4geo_parsers import parse_detector_constants 

def plot_hit_distribution(stats, output_file=None):
    """
    Create visualization of hit distribution
    
    Parameters:
    -----------
    stats : dict
        Statistics from analyze_vertex_detector
    output_file : str, optional
        If provided, save plot to this file
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Hit distribution histogram
    hits, counts = zip(*sorted(stats['hit_distribution'].items()))
    ax1.bar(hits, counts)
    ax1.set_xlabel('Hits per cell')
    ax1.set_ylabel('Number of cells')
    ax1.set_title('Hit Distribution')
    ax1.set_yscale('log')
    
    # Per-layer statistics
    layers = sorted(stats['per_layer'].keys())
    #occupancies = [stats['per_layer'][layer]['cells_hit'] for layer in layers]
    occupancies = [stats['per_layer'][layer]['pixels_hit'] for layer in layers]

    ax2.bar(layers, occupancies)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Number of cells hit')
    ax2.set_title('Cells Hit per Layer')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
    plt.show()



def plot_occupancy_analysis(stats, geometry_info, output_prefix=None):
    """
    Create detailed visualizations of the occupancy analysis
    
    Parameters:
    -----------
    stats : dict
        Statistics from analyze_vertex_detector
    geometry_info : dict
        Geometry information
    output_prefix : str, optional
        If provided, save plots with this prefix
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2)
    
    # 1. Occupancy vs threshold for each layer
    ax1 = fig.add_subplot(gs[0, 0])
    thresholds = sorted(stats['threshold_stats'].keys())
    for layer in sorted(geometry_info['layers'].keys()):
        #occupancies = [stats['threshold_stats'][t]['per_layer'].get(layer, {'occupancy': 0})['occupancy'] * 100 
        #              for t in thresholds]
        occupancies = [stats['threshold_stats'][t]['per_layer'].get(layer, {'occupancy': 0})['occupancy'] 
                      for t in thresholds]
        ax1.plot(thresholds, occupancies, 'o-', label=f'Layer {layer}')
    
    ax1.set_xlabel('Hit Threshold')
    #ax1.set_ylabel('Occupancy (%)')
    ax1.set_ylabel('Occupancy')
    ax1.set_yscale('log')
    ax1.set_title('Occupancy vs Hit Threshold by Layer')
    ax1.grid(True)
    ax1.legend()
    
    # 2. R-Phi hit distribution
    # 2. R-Phi hit distribution
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    # Convert awkward arrays to numpy arrays
    phi_vals = ak.to_numpy(stats['positions']['phi'])
    r_vals = ak.to_numpy(stats['positions']['r'])

    # Create the edges first
    r_edges = np.linspace(0, max(stats['positions']['r']), 21)
    phi_edges = np.linspace(-np.pi, np.pi, 51)

    # Create histogram with the same number of bins as your edges
    hist, _, _ = np.histogram2d(phi_vals, r_vals, 
                            bins=[phi_edges, r_edges])  # Use the same edges here

    # Create meshgrid
    R, PHI = np.meshgrid(r_edges[:-1], phi_edges[:-1])

    # Create a colormap that uses purple for zero values
    pcm = ax2.pcolormesh(PHI, R, hist, 
                        shading='auto', 
                        cmap='viridis',
                        vmin=0,
                        edgecolors='none')
    ax2.set_facecolor('#3F007D')
    





    # 3. R-Z hit distribution
    ax3 = fig.add_subplot(gs[1, :])
    # Convert awkward arrays to numpy arrays
    z_vals = ak.to_numpy(stats['positions']['z'])
    r_vals = ak.to_numpy(stats['positions']['r'])
    hist, xedges, yedges = np.histogram2d(z_vals, r_vals, bins=[100, 30])
    

    pcm = ax3.pcolormesh(xedges[:-1], yedges[:-1], hist.T, shading='auto')
    ax3.set_xlabel('Z [mm]')
    ax3.set_ylabel('R [mm]')
    ax3.set_title('Hit Distribution (R-Z)')
    plt.colorbar(pcm, ax=ax3, label='Hits')
    
    plt.tight_layout()
    
    if output_prefix:
        plt.savefig(f'{output_prefix}_occupancy_analysis.png')
        plt.savefig(f'{output_prefix}_occupancy_analysis.pdf')
    plt.show()



   
def plot_detector_analysis(stats, geometry_info, detector_name, output_prefix=None):
    """
    Create detailed visualizations of the detector analysis
    
    Parameters as before
    """
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2)
    
    # 1. Occupancy vs threshold for each layer
    ax1 = fig.add_subplot(gs[0, 0])
    thresholds = sorted(stats['threshold_stats'].keys())
    for layer in sorted(geometry_info['layers'].keys()):
        occupancies = [stats['threshold_stats'][t]['per_layer'].get(layer, {'occupancy': 0})['occupancy'] * 100 
                      for t in thresholds]
        ax1.plot(thresholds, occupancies, 'o-', label=f'Layer {layer}')
    
    ax1.set_xlabel('Hit Threshold')
    ax1.set_ylabel('Occupancy (%)')
    ax1.set_title(f'{detector_name} Occupancy vs Hit Threshold by Layer')
    ax1.grid(True)
    ax1.legend()
    
    # 2. R-Phi hit distribution
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    phi_vals = ak.to_numpy(stats['positions']['phi'])
    r_vals = ak.to_numpy(stats['positions']['r'])
    hist, _, _ = np.histogram2d(phi_vals, r_vals, bins=[50, 20])
    r_edges = np.linspace(0, max(r_vals), 21)
    phi_edges = np.linspace(-np.pi, np.pi, 51)
    R, PHI = np.meshgrid(r_edges[:-1], phi_edges[:-1])
    
    pcm = ax2.pcolormesh(PHI, R, hist, shading='auto')
    ax2.set_title(f'{detector_name} Hit Distribution (R-Phi)')
    plt.colorbar(pcm, ax=ax2, label='Hits')
    
    # 3. R-Z hit distribution
    ax3 = fig.add_subplot(gs[1, :])
    z_vals = ak.to_numpy(stats['positions']['z'])
    hist, xedges, yedges = np.histogram2d(z_vals, r_vals, bins=[100, 20])
    
    pcm = ax3.pcolormesh(xedges[:-1], yedges[:-1], hist.T, shading='auto')
    ax3.set_xlabel('Z [mm]')
    ax3.set_ylabel('R [mm]')
    ax3.set_title(f'{detector_name} Hit Distribution (R-Z)')
    plt.colorbar(pcm, ax=ax3, label='Hits')
    
    plt.tight_layout()
    
    if output_prefix:
        plt.savefig(f'{output_prefix}_{detector_name}_analysis.png')
        plt.savefig(f'{output_prefix}_{detector_name}_analysis.pdf')
    plt.show()
                
                
def print_occupancy_statistics(results, geometry_info):
    """
    Print detailed occupancy statistics
    """
    print("\nDetailed occupancy statistics:")
    
    for threshold in sorted(results['threshold_stats'].keys()):
        print(f"\nThreshold = {threshold}:")
        stats = results['threshold_stats'][threshold]
        
        for layer in sorted(stats['per_layer'].keys()):
            layer_stats = stats['per_layer'][layer]
            print(f"\nLayer {layer}:")
            print(f"  Cells hit: {layer_stats['cells_hit']}")
            print(f"  Total hits: {layer_stats['total_hits']}")
            print(f"  Occupancy: {layer_stats['occupancy']*100:.10f}%")
            print(f"  Mean hits per hit cell: {layer_stats['mean_hits']:.10f}")




def analyze_detectors_and_plot(DETECTOR_CONFIGS=None,detectors_to_analyze=None,event_trees=None,main_xml=None,remove_zeros=True):

        print("\nAnalyzing detectors:")
        for detector_name, xml_file in detectors_to_analyze:
            print(f"\nAnalyzing {detector_name}...")
            try:
                detector_config = DETECTOR_CONFIGS[detector_name]
                xmls = get_xmls()
                main_xml = xmls['main_xml']
                # Pass detector name to get specific debug info
                constants = parse_detector_constants(main_xml, detector_name)
                geometry_info = get_geometry_info(xml_file, detector_config, constants=constants)
                
                # Print geometry info
                print(f"\nGeometry info for {detector_name}:")
                print(f"Total cells: {geometry_info['total_cells']}")
                for layer, info in sorted(geometry_info['layers'].items()):
                    print(f"\nLayer {layer}:")
                    for key, value in info.items():
                        print(f"  {key}: {value}")
                
        
                # Define thresholds to analyze
                hit_thresholds = [1, 2, 3, 4, 5,6,7,8,9, 10]
                stats = analyze_detector_hits(event_trees,detector_name=detector_name,config=detector_config, hit_thresholds=hit_thresholds, geometry_file=xml_file,constants=constants,main_xml=main_xml,remove_zeros=True)
                #plot_detector_analysis(stats, geometry_info, detector_name=detector_name, output_prefix=None)
                
                # Create visualizations
                plot_occupancy_analysis(stats, geometry_info, output_prefix=detector_name)
                
                # Print detailed statistics for threshold=1
                for threshold in range(1, 10):  # Loop from 1 to 5 (inclusive)
                    print(f"\nDetailed occupancy statistics (threshold={threshold}):")

                    print(stats['threshold_stats'])
                    
                    # for layer, info in sorted(stats['threshold_stats'][threshold]['per_layer'].items()):
                    #     if layer in geometry_info['layers']:
                    #         print(f"\nLayer {layer}:")
                    #         #print(f"  Cells hit: {info['cells_hit']}")
                    #         print(f"  Cells hit: {info['pixels_hit']}")

                    #         print(f"  Total hits: {info['total_hits']}")
                    #         print(f"  Occupancy: {info['occupancy']*100:.10f}%")
                    #         print(f"  Mean hits per hit cell: {info['mean_hits']:.10f}")
                
                        
            except Exception as e:
                print(f"Error processing {detector_name}: {str(e)}")
                traceback.print_exc()
