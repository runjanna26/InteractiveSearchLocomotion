import numpy as np

class TerrainGenerator:
    def __init__(self, seed=None):
        """
        Initializes the generator. Optional seed for reproducible terrain.
        """
        if seed is not None:
            np.random.seed(seed)

    def generate_rough_terrain(self,
                               name='generated_terrain', 
                               n_rows=10, 
                               n_cols=3, 
                               box_size=0.125, 
                               h_base=0.0, 
                               h_dev=0.05, 
                               spacing=0.25, 
                               start_pos=(2.5, 0, 1.5),
                               slope_deg=0.0,                   # NEW: Slope in degrees
                               friction=(2.0, 0.05, 0.01)):     # NEW: Configurable friction
        """Generates a MuJoCo XML string for a block of rough terrain."""
        xml_lines = []
        # Apply slope using euler pitch (rotation around Y axis)
        xml_lines.append(f'<body name="{name}" pos="{start_pos[0]} {start_pos[1]} {start_pos[2]}" euler="0 {slope_deg} 0">')

        f_str = f"{friction[0]} {friction[1]} {friction[2]}"

        for i in range(n_rows):
            for j in range(n_cols):
                x = i * spacing
                y = (j - (n_cols - 1) / 2) * (spacing - 0.01) 
                geom_name = f"{name}_g_{i}_{j}"

                random_h = max(0.01, h_base + np.random.uniform(0, h_dev))
                z = random_h / 2
                line = (f'<geom name="{geom_name}" type="box" size="{box_size} {box_size} {random_h:.3f}" '
                        f'pos="{x:.2f} {y:.2f} {z:.3f}" '
                        f'rgba="0.7 0.1 0.1 1" friction="{f_str}" material="matrough"/>')
                xml_lines.append(line)

        xml_lines.append('</body>')
        return "\n".join(xml_lines)
    
    def generate_flat_terrain(self,
                              name='generated_terrain', 
                              n_rows=10, 
                              n_cols=3, 
                              box_size=0.125, 
                              h_base=0.1, 
                              spacing=0.25, 
                              start_pos=(2.5, 0, 1.5),
                              slope_deg=0.0,                   # NEW: Slope
                              friction=(2.0, 0.05, 0.01)):     # NEW: Configurable friction
        """Generates a MuJoCo XML string for flat terrain."""
        total_length_x = n_rows * spacing
        total_width_y = n_cols * spacing
        
        size_x = total_length_x / 2.0
        size_y = total_width_y / 2.0
        size_z = max(0.01, h_base / 2.0)
        
        pos_x = size_x - (spacing / 2.0)
        pos_y = 0.0
        pos_z = size_z

        xml_lines = []
        xml_lines.append(f'<body name="{name}" pos="{start_pos[0]} {start_pos[1]} {start_pos[2]}" euler="0 {slope_deg} 0">')

        f_str = f"{friction[0]} {friction[1]} {friction[2]}"
        geom_name = f"{name}_geom"
        line = (f'<geom name="{geom_name}" type="box" size="{size_x:.3f} {size_y:.3f} {size_z:.3f}" '
                f'pos="{pos_x:.3f} {pos_y:.3f} {pos_z:.3f}" '
                f'material="solid_ground" friction="{f_str}" contype="1" conaffinity="1"/>')
        xml_lines.append(line)

        xml_lines.append('</body>')
        return "\n".join(xml_lines)
    
    def generate_soft_terrain(self,
                              name='generated_terrain', 
                              n_rows=10, 
                              n_cols=3, 
                              h_base=0.2, 
                              spacing=0.25, 
                              start_pos=(2.5, 0, 1.5),
                              slope_deg=0.0,                   # NEW: Slope
                              friction=(1.0, 0.05, 0.01),      # NEW: Friction
                              solref=(0.1, 0.7),               # NEW: Softness/Bounce (timeconst, dampratio)
                              solimp=(0.1, 0.99, 0.08)):       # NEW: Hardness profile
        """Generates a MuJoCo XML string for sponge terrain."""
        total_length_x = n_rows * spacing
        total_width_y = n_cols * spacing
        
        size_x = total_length_x / 2.0
        size_y = total_width_y / 2.0
        size_z = max(0.01, h_base / 2.0)
        
        pos_x = size_x - (spacing / 2.0)
        pos_y = 0.0
        pos_z = size_z

        xml_lines = []
        xml_lines.append(f'<body name="{name}" pos="{start_pos[0]} {start_pos[1]} {start_pos[2]}" euler="0 {slope_deg} 0">')

        f_str = f"{friction[0]} {friction[1]} {friction[2]}"
        sref_str = f"{solref[0]} {solref[1]}"
        simp_str = f"{solimp[0]} {solimp[1]} {solimp[2]}"

        geom_name = f"{name}_geom"
        line = (f'<geom name="{geom_name}" type="box" size="{size_x:.3f} {size_y:.3f} {size_z:.3f}" '
                f'pos="{pos_x:.3f} {pos_y:.3f} {pos_z:.3f}" '
                f'rgba="0.5 0.647 0 0.5" friction="{f_str}" '
                f'material="soft_ground" solref="{sref_str}" solimp="{simp_str}"/>')
        xml_lines.append(line)

        xml_lines.append('</body>')
        return "\n".join(xml_lines)
    
    def generate_slippery_terrain(self,
                                  name='generated_terrain', 
                                  n_rows=10, 
                                  n_cols=3, 
                                  h_base=0.2, 
                                  spacing=0.25, 
                                  start_pos=(2.5, 0, 1.5),
                                  slope_deg=0.0,                   # NEW: Slope
                                  friction=(0.0, 0.005, 0.0001)):  # NEW: Super low friction
        """Generates a MuJoCo XML string for icy terrain."""
        total_length_x = n_rows * spacing
        total_width_y = n_cols * spacing
        
        size_x = total_length_x / 2.0
        size_y = total_width_y / 2.0
        size_z = max(0.01, h_base / 2.0)
        
        pos_x = size_x - (spacing / 2.0)
        pos_y = 0.0
        pos_z = size_z

        xml_lines = []
        xml_lines.append(f'<body name="{name}" pos="{start_pos[0]} {start_pos[1]} {start_pos[2]}" euler="0 {slope_deg} 0">')

        f_str = f"{friction[0]} {friction[1]} {friction[2]}"

        geom_name = f"{name}_geom"
        line = (f'<geom name="{geom_name}" type="box" size="{size_x:.3f} {size_y:.3f} {size_z:.3f}" '
                f'pos="{pos_x:.3f} {pos_y:.3f} {pos_z:.3f}" '
                f'material="slippery_ground" friction="{f_str}" />')
        xml_lines.append(line)

        xml_lines.append('</body>')
        return "\n".join(xml_lines)
    
    def generate_muddy_terrain(self,
                               name='generated_terrain', 
                               n_rows=10, 
                               n_cols=3, 
                               h_base=0.2, 
                               spacing=0.25, 
                               start_pos=(2.5, 0, 1.5),
                               slope_deg=0.0,                   # NEW: Slope
                               friction=(1.0, 0.05, 0.01),      # NEW: Friction
                               solref=(0.2, 3.0),               # NEW: Viscous damping (timeconst, dampratio)
                               solimp=(0.0, 0.99, 0.15)):       # NEW: Penetration resistance
        """Generates a MuJoCo XML string for muddy terrain."""
        total_length_x = n_rows * spacing
        total_width_y = n_cols * spacing
        
        size_x = total_length_x / 2.0
        size_y = total_width_y / 2.0
        size_z = max(0.01, h_base / 2.0)
        
        pos_x = size_x - (spacing / 2.0)
        pos_y = 0.0
        pos_z = size_z

        xml_lines = []
        xml_lines.append(f'<body name="{name}" pos="{start_pos[0]} {start_pos[1]} {start_pos[2]}" euler="0 {slope_deg} 0">')

        f_str = f"{friction[0]} {friction[1]} {friction[2]}"
        sref_str = f"{solref[0]} {solref[1]}"
        simp_str = f"{solimp[0]} {solimp[1]} {solimp[2]}"

        geom_name = f"{name}_geom"
        line = (f'<geom name="{geom_name}" type="box" size="{size_x:.3f} {size_y:.3f} {size_z:.3f}" '
                f'pos="{pos_x:.3f} {pos_y:.3f} {pos_z:.3f}" '
                f'rgba="0.445 0.171 0.075 1" friction="{f_str}" '
                f'material="muddy_ground" solref="{sref_str}" solimp="{simp_str}"/>')  
        xml_lines.append(line)

        xml_lines.append('</body>')
        return "\n".join(xml_lines)

    def generate_water_pool(self,
                           name='water_pool', 
                           n_rows=10, 
                           n_cols=3, 
                           depth=1.0, 
                           spacing=0.25, 
                           start_pos=(2.5, 0, 1.5)):
        """Generates a MuJoCo XML string for a visual-only block of water."""
        total_length_x = n_rows * spacing
        total_width_y = n_cols * spacing
        
        size_x = total_length_x / 2.0
        size_y = total_width_y / 2.0
        size_z = depth / 2.0
        
        pos_x = size_x - (spacing / 2.0)
        pos_y = 0.0
        pos_z = size_z

        xml_lines = []
        xml_lines.append(f'<body name="{name}" pos="{start_pos[0]} {start_pos[1]} {start_pos[2]}">')

        geom_name = f"{name}_geom"
        line = (f'<geom name="{geom_name}" type="box" size="{size_x:.3f} {size_y:.3f} {size_z:.3f}" '
                f'pos="{pos_x:.3f} {pos_y:.3f} {pos_z:.3f}" '
                f'rgba="0.1 0.4 0.8 0.4" contype="0" conaffinity="0" />')  
        xml_lines.append(line)

        xml_lines.append('</body>')
        return "\n".join(xml_lines)