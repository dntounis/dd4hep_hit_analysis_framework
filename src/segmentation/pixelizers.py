def calculate_local_coordinates(hit_pos, module_pos, module_orientation, geometry_type='barrel'):
    """
    Convert hit position to local coordinates within a module.
    Works with existing geometry information.
    
    Parameters:
    -----------
    hit_pos : tuple
        (x, y, z) coordinates of the hit
    module_pos : tuple
        (x, y, z, phi) module position and orientation
    module_orientation : float
        Module orientation angle 
    geometry_type : str
        'barrel' or 'endcap'
        
    Returns:
    --------
    tuple (local_x, local_y)
    """
    # Extract positions
    hit_x, hit_y, hit_z = hit_pos
    mod_x, mod_y, mod_z, mod_phi = module_pos
    
    # Calculate position relative to module center
    dx = hit_x - mod_x
    dy = hit_y - mod_y
    dz = hit_z - mod_z
    
    if geometry_type == 'barrel':
        # For barrel, rotate in r-phi plane
        cos_phi = np.cos(-mod_phi)  # Negative for inverse rotation
        sin_phi = np.sin(-mod_phi)
        
        local_x = dx * cos_phi - dy * sin_phi
        local_y = dx * sin_phi + dy * cos_phi
        
    else:  # endcap
        # For endcap, consider module is perpendicular to z
        cos_phi = np.cos(-mod_phi)
        sin_phi = np.sin(-mod_phi)
        
        # Project onto module plane
        local_x = dx * cos_phi - dy * sin_phi
        local_y = dy * cos_phi + dx * sin_phi
        
    return (local_x, local_y)

def get_pixel_indices(local_coords, module_dims, cell_size):
    """Convert local coordinates to pixel indices with proper bounds checking"""
    local_x, local_y = local_coords
    width, length = module_dims
    
    # Convert to pixel indices
    # Add width/2 and length/2 to shift from [-w/2, w/2] to [0, w]
    pixel_x = int((local_x + width/2) / cell_size['x'])
    pixel_y = int((local_y + length/2) / cell_size['y'])
    
    # Ensure indices are within module bounds
    pixels_x = int(width / cell_size['x'])
    pixels_y = int(length / cell_size['y'])
    
    pixel_x = max(0, min(pixel_x, pixels_x - 1))
    pixel_y = max(0, min(pixel_y, pixels_y - 1))
    
    return (pixel_x, pixel_y)


def process_hit_segmentation(hit_data, layer_info, detector_config, detector_name):
    """
    Process a hit to determine its pixel/cell location.
    
    Parameters:
    -----------
    hit_data : dict
        Dictionary containing:
        - pos: tuple of (x, y, z) hit position
        - cellid_decoded: decoded cellID information
    layer_info : dict
        Layer geometry information
    detector_config : DetectorConfig
        Detector configuration object
    detector_name : str
        Name of the detector
        
    Returns:
    --------
    tuple: For module-based detectors: (layer, module, pixel_x, pixel_y)
           For forward calos: (layer, cell_x, cell_y)
    """
    # Extract basic hit information
    decoded = hit_data['cellid_decoded']
    layer = decoded['layer']
    
    # Handle forward calorimeters (BeamCal, LumiCal) differently
    if detector_name in ['BeamCal', 'LumiCal']:
        # For these detectors, x and y are directly encoded in the cellID
        cell_x = decoded['x']
        cell_y = decoded['y']
        return (layer, cell_x, cell_y)
    
    # For module-based detectors
    hit_pos = hit_data['pos']
    module = decoded['module']
    
    # Determine if this is an endcap detector
    is_endcap = 'inner_r' in layer_info and 'zstart' in layer_info
    
    # Calculate module position
    module_pos = calculate_module_position(layer_info, module)
    
    # Get module orientation
    if is_endcap:
        nmodules = layer_info['nmodules']
        module_phi = 2 * np.pi * module / nmodules
        if 'phi0' in layer_info:
            module_phi += layer_info['phi0']
    else:
        module_phi = 2 * np.pi * module / layer_info['nphi']
        if 'phi0' in layer_info:
            module_phi += layer_info['phi0']
        if 'phi_tilt' in layer_info:
            module_phi += layer_info['phi_tilt']
    
    # Convert to local coordinates
    local_coords = calculate_local_coordinates(hit_pos, module_pos, module_phi, is_endcap)
    
    # Get module dimensions
    module_dims = (layer_info['module_width'], layer_info['module_length'])
    
    # Get pixel size from detector configuration
    pixel_size = detector_config.get_cell_size(layer=layer)
    
    # Calculate pixel indices
    pixel_indices = get_pixel_indices(local_coords, module_dims, pixel_size)
    
    return (layer, module, pixel_indices[0], pixel_indices[1])

def get_local_coordinates_barrel_polar(hit_pos, layer_info, module, cell_size, effective_cell_size=None):
    """
    Convert global hit coordinates to module-local pixel indices for a barrel detector,
    using a polar-coordinate transformation.
    
    Parameters:
      hit_pos: tuple (hit_x, hit_y, hit_z) in mm
      layer_info: dict with keys:
          'rc' or ('inner_r' and 'outer_r'),
          'nphi', 'phi0', 'phi_tilt', 'width' (module width, tangential), 'length' (module length, radial)
      module: module index (integer)
      cell_size: dict with keys 'x' and 'y' in mm (ideal pixel dimensions)
      effective_cell_size: dict with keys 'x' and 'y' in mm; if provided, used instead of cell_size
      
    Returns:
      tuple (pixel_x, pixel_y) or None.
    """
    hit_x, hit_y, hit_z = hit_pos

    # Use effective cell size if provided
    if effective_cell_size is not None:
        csize = effective_cell_size
    else:
        csize = cell_size

    # Nominal radius
    rc = layer_info.get('rc')
    if rc is None:
        inner_r = layer_info.get('inner_r')
        outer_r = layer_info.get('outer_r')
        if inner_r is None or outer_r is None:
            return None
        rc = (inner_r + outer_r) / 2

    nphi = layer_info.get('nphi')
    if not nphi:
        return None

    phi0 = layer_info.get('phi0', 0.0)
    phi_tilt = layer_info.get('phi_tilt', 0.0)

    # Compute hit polar coordinates
    r_hit = np.sqrt(hit_x**2 + hit_y**2)
    phi_hit = np.arctan2(hit_y, hit_x)

    # Compute module center phi using fixed nominal radius (with +0.5 for centering)
    delta_phi = 2 * np.pi / nphi
    module_center_phi = phi0 + (module + 0.5) * delta_phi + phi_tilt

    # Use polar differences
    local_radial = r_hit - rc
    dphi = (phi_hit - module_center_phi + np.pi) % (2*np.pi) - np.pi
    local_tangential = dphi * rc

    # Use the module dimensions from layer_info (in mm)
    module_length = layer_info.get('length')
    module_width = layer_info.get('width')
    if module_length is None or module_width is None:
        return None

    # Map local coordinates to pixel indices
    pixel_x = int((local_radial + module_length/2) / csize['x'])
    pixel_y = int((local_tangential + module_width/2) / csize['y'])

    # Optionally enforce bounds
    max_pixels_x = int(module_length / csize['x'])
    max_pixels_y = int(module_width / csize['y'])
    pixel_x = max(0, min(pixel_x, max_pixels_x - 1))
    pixel_y = max(0, min(pixel_y, max_pixels_y - 1))

    counter =0
    if counter <10 : 
        print("Hit:", hit_x, hit_y, "r_hit=", r_hit, "phi_hit=", phi_hit)
        print("Module center phi=", module_center_phi)
        print("local_radial =", local_radial, "local_tangential =", local_tangential)
        print("pixel indices =", (pixel_x, pixel_y))
        counter += 1
    
    return (pixel_x, pixel_y)




def get_pixel_id(hit_pos, decoded, config, geometry_info, effective_cell_size=None, verbose=False):
    """
    Compute a pixel (cell) identifier for a hit.

    For tracker/vertex detectors in a barrel, the function uses a polar transformation.
    For calorimeters and forward detectors, it uses a simple Cartesian grid.

    Parameters:
      hit_pos: tuple (x, y, z) in mm (global coordinates)
      decoded: dictionary from your DD4hep decoder (e.g. contains 'layer' and, for trackers, 'module')
      config: DetectorConfig instance (its detector_class and detector_type are used)
      geometry_info: dict from get_geometry_info with per-layer geometry information
      effective_cell_size: optional dict overriding default cell size (keys 'x' and 'y')
      verbose: if True, print debug info for a few hits

    Returns:
      A tuple uniquely identifying the pixel. For a barrel tracker, it returns
      (layer, module, pixel_t, pixel_z), where:
         - pixel_t is the index in the tangential direction (from the polar transform)
         - pixel_z is the index along z.
      For calorimeters, it returns a Cartesian (pixel_x, pixel_y).
    """
    # Get cell size (default: use config.get_cell_size())
    cell_size = effective_cell_size if effective_cell_size is not None else config.get_cell_size(layer=decoded.get('layer'))
    det_class = config.detector_class.lower()
    det_type  = config.detector_type.lower()  # expected "barrel" or "endcap"
    layer = decoded.get('layer')
    
    # --- Tracker/Vertex devices ---
    if det_class in ["vertex", "tracker"]:
        if geometry_info and (layer in geometry_info['layers']):
            layer_geom = geometry_info['layers'][layer]
            if det_type == "barrel":
                # Get global hit coordinates
                hit_x, hit_y, hit_z = hit_pos
                r_hit = np.sqrt(hit_x**2 + hit_y**2)
                phi_hit = np.arctan2(hit_y, hit_x)
                nphi = layer_geom.get('nphi')
                phi0 = layer_geom.get('phi0', 0.0)
                phi_tilt = layer_geom.get('phi_tilt', 0.0)
                # Compute nominal radius: either directly from "rc" or from inner/outer radii.
                rc = layer_geom.get('rc')
                if rc is None:
                    inner_r = layer_geom.get('inner_r')
                    outer_r = layer_geom.get('outer_r')
                    if inner_r is not None and outer_r is not None:
                        rc = (inner_r + outer_r) / 2
                    else:
                        rc = 0.0
                module = decoded.get('module', 0)
                delta_phi = 2 * np.pi / nphi
                # Compute the module center in φ
                module_center_phi = phi0 + (module + 0.5)*delta_phi + phi_tilt
                # Wrap the difference into [–π, π]
                dphi = (phi_hit - module_center_phi + np.pi) % (2*np.pi) - np.pi
                # Compute local tangential coordinate
                local_t = rc * dphi
                # For the vertex barrel, the sensor’s “width” (tangential) and “length” (z)
                sensor_width = layer_geom.get('width', 1.0)   # e.g. 9.8 mm
                sensor_length = layer_geom.get('length', 1.0)   # e.g. 126 mm
                # Get z0 offset (default 0)
                z0 = layer_geom.get('z0', 0.0)
                local_z = hit_z - z0
                # Map local coordinates (which we assume span from –width/2 to +width/2, etc.)
                pixel_t = int((local_t + sensor_width/2) / cell_size['x'])
                pixel_z = int((local_z + sensor_length/2) / cell_size['y'])
                # Optionally enforce bounds:
                max_pixels_t = int(sensor_width / cell_size['x'])
                max_pixels_z = int(sensor_length / cell_size['y'])
                pixel_t = max(0, min(pixel_t, max_pixels_t - 1))
                pixel_z = max(0, min(pixel_z, max_pixels_z - 1))
                if verbose:
                    if not hasattr(get_pixel_id, "verbose_count"):
                        get_pixel_id.verbose_count = 0
                    if get_pixel_id.verbose_count < 3:
                        print("=== Tracker Barrel Hit ===")
                        print("Global hit pos:", hit_pos)
                        print("r_hit =", r_hit, "phi_hit =", phi_hit)
                        print("Module index:", module, "Module center φ =", module_center_phi)
                        print("dphi =", dphi, "=> local tangential =", local_t)
                        print("Local z =", local_z)
                        print("Pixel indices: (t, z) =", (pixel_t, pixel_z))
                        get_pixel_id.verbose_count += 1
                return (layer, module, pixel_t, pixel_z)
            # For an endcap tracker we might simply use a Cartesian projection
            elif det_type == "endcap":
                hit_x, hit_y, hit_z = hit_pos
                pixel_x = int(hit_x / cell_size['x'])
                pixel_z = int(hit_z / cell_size['y'])
                return (layer, pixel_x, pixel_z)
        # Fallback: if no geometry info, simply return (layer, module)
        module = decoded.get('module', 0)
        return (layer, module)
    
    # --- Calorimeters and forward detectors ---
    elif det_class in ["ecal", "hcal", "muon", "beamcal", "lumical"]:
        hit_x, hit_y, hit_z = hit_pos
        pixel_x = int(hit_x / cell_size['x'])
        pixel_y = int(hit_y / cell_size['y'])
        return (pixel_x, pixel_y)
    
    # --- Fallback for unknown types ---
    else:
        module = decoded.get('module', 0)
        return (layer, module)


def get_local_pixel_coordinates(hit_pos, layer_info, module, cell_size):
    """
    Convert a hit position to local pixel coordinates within a module.
    
    Parameters:
    -----------
    hit_pos : tuple (x, y, z)
        Hit position in global coordinates
    layer_info : dict
        Layer geometry information including module dimensions
    module : int
        Module number 
    cell_size : dict
        Cell size in x,y directions
        
    Returns:
    --------
    tuple (pixel_x, pixel_y) or None if conversion fails
    """
    hit_x, hit_y, hit_z = hit_pos
    
    # Get module phi position
    nphi = layer_info.get('nphi', 0)
    if nphi == 0:
        return None
        
    phi0 = layer_info.get('phi0', 0.0)
    phi_tilt = layer_info.get('phi_tilt', 0.0)
    
    # Calculate module center phi
    module_phi = phi0 + (2 * np.pi * module / nphi) + phi_tilt
    
    # Get module radius
    rc = layer_info.get('rc')
    if rc is None:
        return None
        
    # Calculate module center position
    module_center_x = rc * np.cos(module_phi)
    module_center_y = rc * np.sin(module_phi)
    
    # Transform hit to module local coordinates
    # First rotate to module's phi
    dx = hit_x - module_center_x
    dy = hit_y - module_center_y
    
    cos_phi = np.cos(-module_phi)
    sin_phi = np.sin(-module_phi)
    
    local_x = dx * cos_phi - dy * sin_phi
    local_y = dx * sin_phi + dy * cos_phi
    
    # Convert to pixel coordinates
    module_width = layer_info.get('width', 0)
    module_length = layer_info.get('length', 0)
    
    if module_width == 0 or module_length == 0:
        return None
        
    # Convert from [-width/2, width/2] to [0, pixels_x]
    pixel_x = int((local_x + module_width/2) / cell_size['x'])
    pixel_y = int((local_y + module_length/2) / cell_size['y'])
    
    # Ensure pixels are within module bounds
    pixels_x = int(module_width / cell_size['x'])
    pixels_y = int(module_length / cell_size['y'])
    
    if 0 <= pixel_x < pixels_x and 0 <= pixel_y < pixels_y:
        return (pixel_x, pixel_y)
        
    return None