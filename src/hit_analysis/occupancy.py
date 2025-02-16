def process_hit(hit_data, detector_name, layer_info, config):
    """
    Process a single hit to determine its pixel/cell location.
    
    Parameters:
    -----------
    hit_data : dict
        {'pos': (x,y,z), 'cellid_decoded': decoded_cellid}
    detector_name : str
        Name of the detector
    layer_info : dict
        Layer geometry information
    config : DetectorConfig
        Detector configuration
        
    Returns:
    --------
    tuple: pixel key for hit counting
    """
    decoded = hit_data['cellid_decoded']
    hit_pos = hit_data['pos']
    layer = decoded['layer']
    
    # Handle forward calorimeters directly
    if detector_name in ['BeamCal', 'LumiCal']:
        return (layer, decoded['x'], decoded['y'])
    
    # Get module number from decoded cellID
    module = decoded['module']
    
    # Determine if this is an endcap by checking for rings
    is_endcap = 'rings' in layer_info
    
    try:
        if is_endcap:
            # Find which ring this hit belongs to
            ring_info = find_matching_ring(hit_pos, layer_info)
            if ring_info is None:
                print(f"Warning: Hit at {hit_pos} doesn't match any ring in layer {layer}")
                return (layer, module)
                
            # Calculate local coordinates within ring module
            module_pos = calculate_module_position(ring_info, module, 'endcap')
            if module_pos is None:
                return (layer, module)
                
            local_coords = calculate_local_coordinates(
                hit_pos, module_pos, module_pos[3], 'endcap'
            )
            
            # Get pixel indices using ring module dimensions
            if ring_info['type'] == 'trd':
                # For trapezoid, use average width
                module_dims = (ring_info['width'], ring_info['length'])
            else:
                module_dims = (ring_info['width'], ring_info['length'])
                
            pixel_indices = get_pixel_indices(
                local_coords, module_dims, config.get_cell_size(layer=layer)
            )
            
            # Include ring number in pixel key for endcap
            ring_idx = layer_info['rings'].index(ring_info)
            return (layer, ring_idx, module, pixel_indices[0], pixel_indices[1])
            
        else:
            # Barrel detector
            if 'rc' not in layer_info:
                # Try to calculate rc from inner/outer radii
                if 'inner_r' in layer_info and 'outer_r' in layer_info:
                    layer_info['rc'] = (layer_info['inner_r'] + layer_info['outer_r']) / 2
                else:
                    print(f"Warning: Cannot determine radius for layer {layer}")
                    return (layer, module)
            
            # Calculate module position and local coordinates
            module_pos = calculate_module_position(layer_info, module, 'barrel')
            if module_pos is None:
                return (layer, module)
                
            local_coords = calculate_local_coordinates(
                hit_pos, module_pos, module_pos[3], 'barrel'
            )
            
            # Get module dimensions
            if all(key in layer_info for key in ['module_width', 'module_length']):
                module_dims = (layer_info['module_width'], layer_info['module_length'])
            else:
                print(f"Warning: Missing module dimensions for layer {layer}")
                return (layer, module)
            
            # Calculate pixel indices
            pixel_indices = get_pixel_indices(
                local_coords, module_dims, config.get_cell_size(layer=layer)
            )
            
            return (layer, module, pixel_indices[0], pixel_indices[1])
            
    except Exception as e:
        print(f"Warning: Error processing hit in {detector_name}, layer {layer}: {str(e)}")
        return (layer, module)




















def calculate_layer_occupancy(layer_cells, total_cells, threshold):
    """
    Calculate occupancy for a layer at given threshold
    
    Parameters:
    -----------
    layer_cells : dict
        Dictionary of cellID -> hit count for cells in this layer
    total_cells : int
        Total number of cells in layer from geometry
    threshold : int
        Hit threshold value
        
    Returns:
    --------
    float : Fraction of cells with hits ≥ threshold
    """
    if total_cells == 0:
        return 0.0
    cells_above = sum(1 for hits in layer_cells.values() if hits >= threshold)
    return cells_above / total_cells



def analyze_detector_hits(events_trees, detector_name, config, hit_thresholds=None, 
                        geometry_file=None, constants=None, main_xml=None):
    """
    Analyze hits with improved coordinate handling for different detector types.
    """
    if hit_thresholds is None:
        hit_thresholds = [1]
    
    # Build branch names
    hits_prefix = f"{detector_name}Hits"
    cellid_branch = f"{hits_prefix}/{hits_prefix}.cellID"
    pos_branches = [f"{hits_prefix}/{hits_prefix}.position.x",
                    f"{hits_prefix}/{hits_prefix}.position.y",
                    f"{hits_prefix}/{hits_prefix}.position.z"]
    
    # Read event arrays
    try:
        arrays = [events_tree.arrays([cellid_branch] + pos_branches)
                  for events_tree in events_trees]
        cellids = [array[cellid_branch] for array in arrays]
        cellids_combined = ak.concatenate([ak.flatten(arr) for arr in cellids])
        x = ak.concatenate([ak.flatten(array[pos_branches[0]]) for array in arrays])
        y = ak.concatenate([ak.flatten(array[pos_branches[1]]) for array in arrays])
        z = ak.concatenate([ak.flatten(array[pos_branches[2]]) for array in arrays])
    except Exception as e:
        print(f"Error reading event data for {detector_name}: {e}")
        return None
    
    # Get geometry info if available
    geometry_info = None
    if geometry_file:
        try:
            geometry_info = get_geometry_info(geometry_file, config, constants, main_xml)
        except Exception as e:
            print(f"Error processing {detector_name}: {e}")
            return None
    
    pixel_hits = {}
    # Use "config" (not detector_config)
    for cellid, hit_x, hit_y, hit_z in zip(cellids_combined, x, y, z):
        try:
            decoded = decode_dd4hep_cellid(cellid, detector_name)
            # Call our unified function; note we pass config (not detector_config)
            pixel_key = get_pixel_id((hit_x, hit_y, hit_z), decoded,
                                     config, geometry_info,
                                     effective_cell_size=None, verbose=True)
            pixel_hits[pixel_key] = pixel_hits.get(pixel_key, 0) + 1
        except Exception as e:
            print(f"Warning: Error processing hit in {detector_name}: {e}")
            continue
    
    # Compute per-threshold statistics
    stats = {}
    for threshold in hit_thresholds:
        layer_stats = {}
        for pixel_key, count in pixel_hits.items():
            layer = pixel_key[0]
            if layer not in layer_stats:
                layer_stats[layer] = {
                    'cells_hit': 0,
                    'total_hits': 0,
                    'cells_above_threshold': 0,
                    'max_hits': 0,
                    'hit_counts': []
                }
            layer_stats[layer]['cells_hit'] += 1
            layer_stats[layer]['total_hits'] += count
            if count >= threshold:
                layer_stats[layer]['cells_above_threshold'] += 1
            layer_stats[layer]['max_hits'] = max(layer_stats[layer]['max_hits'], count)
            layer_stats[layer]['hit_counts'].append(count)
        for layer, lstats in layer_stats.items():
            if geometry_info and layer in geometry_info['layers']:
                total_cells = geometry_info['layers'][layer].get('total_cells', 0)
                lstats['occupancy'] = (lstats['cells_above_threshold'] / total_cells) if total_cells > 0 else 0.0
            else:
                lstats['occupancy'] = 0.0
            lstats['mean_hits'] = np.mean(lstats['hit_counts'])
            del lstats['hit_counts']
        stats[threshold] = {
            'overall_cells_hit': sum(s['cells_hit'] for s in layer_stats.values()),
            'overall_cells_above_threshold': sum(s['cells_above_threshold'] for s in layer_stats.values()),
            'max_hits_per_cell': max(s['max_hits'] for s in layer_stats.values()),
            'hit_distribution': dict(Counter(count for count in pixel_hits.values())),
            'per_layer': layer_stats
        }
    
    return {
        'detector_name': detector_name,
        'detector_class': config.detector_class,
        'threshold_stats': stats,
        'positions': {
            'r': np.sqrt(x**2 + y**2),
            'phi': np.arctan2(y, x),
            'z': z
        }
    }

     




def analyze_vertex_detector(events_trees, hit_thresholds=None, geometry_file=None):
    """
    Comprehensive analysis of SiVertexBarrel hits
    
    Parameters:
    -----------
    events_trees : list of uproot.TTree
        List of event trees from uproot
    hit_thresholds : list of int, optional
        List of hit thresholds to analyze occupancy
    geometry_file : str, optional
        Path to geometry XML file
        
    Returns:
    --------
    dict with analysis results
    """
    if hit_thresholds is None:
        hit_thresholds = [1]
        
    # Get cellIDs and positions
    arrays = [
        events_tree.arrays([
            'SiVertexBarrelHits/SiVertexBarrelHits.cellID',
            'SiVertexBarrelHits/SiVertexBarrelHits.position.x',
            'SiVertexBarrelHits/SiVertexBarrelHits.position.y',
            'SiVertexBarrelHits/SiVertexBarrelHits.position.z'
        ])
        for events_tree in events_trees
    ]
    
    # Combine all hits
    cellids = [array['SiVertexBarrelHits/SiVertexBarrelHits.cellID'] for array in arrays]
    cellids_combined = ak.concatenate([ak.flatten(array) for array in cellids])
    
    x = ak.concatenate([ak.flatten(array['SiVertexBarrelHits/SiVertexBarrelHits.position.x']) for array in arrays])
    y = ak.concatenate([ak.flatten(array['SiVertexBarrelHits/SiVertexBarrelHits.position.y']) for array in arrays])
    z = ak.concatenate([ak.flatten(array['SiVertexBarrelHits/SiVertexBarrelHits.position.z']) for array in arrays])
    
    # Calculate r and phi
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    # Count hits per cell
    hit_counts = Counter(cellids_combined.tolist())
    
    # Get geometry information if file provided
    geometry_info = None
    if geometry_file:
        geometry_info = get_geometry_info(geometry_file)
    
    # Analyze for each threshold
    threshold_stats = {}
    for threshold in hit_thresholds:
        # Decode all cellIDs and organize by layer
        layer_hits = {}
        for cellid, count in hit_counts.items():
            decoded = decode_cellID(cellid)
            layer = decoded['layer'] + 1  # Add 1 to match geometry
            if layer not in layer_hits:
                layer_hits[layer] = {'hit_counts': [], 'total_hits': 0, 'positions': {'r': [], 'phi': [], 'z': []}}
            layer_hits[layer]['hit_counts'].append(count)
            layer_hits[layer]['total_hits'] += count
        
        # Calculate statistics for this threshold
        threshold_stats[threshold] = {
            'total_cells_hit': len(hit_counts),
            'cells_above_threshold': sum(1 for count in hit_counts.values() if count >= threshold),
            'max_hits_per_cell': max(hit_counts.values()) if hit_counts else 0,
            'hit_distribution': dict(Counter(hit_counts.values())),
            'per_layer': {
                layer: {
                    'cells_hit': len(hits['hit_counts']),
                    'cells_above_threshold': sum(1 for count in hits['hit_counts'] if count >= threshold),
                    'total_hits': hits['total_hits'],
                    'max_hits': max(hits['hit_counts']),
                    'mean_hits': np.mean(hits['hit_counts']),
                    'occupancy': None  # Will fill if geometry info available
                }
                for layer, hits in layer_hits.items()
            }
        }
        
        # Add occupancy if geometry information available
        if geometry_info:
            for layer in threshold_stats[threshold]['per_layer']:
                threshold_stats[threshold]['per_layer'][layer]['occupancy'] = 0.0
                if layer in geometry_info['layers']:
                    total_cells = geometry_info['layers'][layer]['total_cells']
                    if total_cells > 0:
                        cells_above = threshold_stats[threshold]['per_layer'][layer]['cells_above_threshold']
                        threshold_stats[threshold]['per_layer'][layer]['occupancy'] = cells_above / total_cells
    
    return {
        'threshold_stats': threshold_stats,
        'positions': {
            'r': r,
            'phi': phi,
            'z': z
        }
    }


