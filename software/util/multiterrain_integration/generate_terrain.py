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
                               start_pos=(2.5, 0, 1.5)):
        """
        Generates a MuJoCo XML string for a block of rough terrain.
        
        Args:
            n_rows: Number of boxes along X (path length)
            n_cols: Number of boxes along Y (path width)
            box_size: Half-length of the box sides
            h_base: Minimum box thickness
            h_dev: Random height variation
            spacing: Distance between box centers
            start_pos: Tuple (x, y, z) for the start of the terrain block
        """
        xml_lines = []
        # Body wrapper for the entire terrain block
        xml_lines.append(f'<body name="{name}" pos="{start_pos[0]} {start_pos[1]} {start_pos[2]}">')

        for i in range(n_rows):
            for j in range(n_cols):
                # Calculate position relative to body origin
                x = i * spacing
                y = (j - (n_cols - 1) / 2) * (spacing - 0.01) 
                
                geom_name = f"{name}_g_{i}_{j}"

                # Randomize height
                random_h = max(0.01, h_base + np.random.uniform(0, h_dev))
                # MuJoCo boxes expand from the center, so z is half the total height
                z = random_h / 2
                line = (f'<geom name="{geom_name}" type="box" size="{box_size} {box_size} {random_h:.3f}" '
                        f'pos="{x:.2f} {y:.2f} {z:.3f}" '
                        f'rgba="0.7 0.1 0.1 1" friction="2.0 0.05 0.01" material="matrough"/>')
                xml_lines.append(line)

        xml_lines.append('</body>')
        return "\n".join(xml_lines)
    
    def generate_flat_terrain(self,
                          name='generated_terrain', 
                          n_rows=10, 
                          n_cols=3, 
                          box_size=0.125, 
                          h_base=0.1,  # Set > 0 so the box has some physical thickness
                          spacing=0.25, 
                          start_pos=(2.5, 0, 1.5)):
        """
        Generates a MuJoCo XML string for a single monolithic block of flat terrain.
        """
        # 1. Calculate total dimensions based on your old grid parameters
        total_length_x = n_rows * spacing
        total_width_y = n_cols * spacing
        
        # 2. MuJoCo box 'size' takes half-extents (half-length, half-width, half-height)
        size_x = total_length_x / 2.0
        size_y = total_width_y / 2.0
        size_z = max(0.01, h_base / 2.0)
        
        # 3. Calculate position relative to body origin
        # We shift X forward by size_x so the back edge of the box starts exactly at start_pos
        pos_x = size_x - (spacing / 2.0)
        pos_y = 0.0
        pos_z = size_z

        xml_lines = []
        
        # Body wrapper for the entire terrain block
        xml_lines.append(f'<body name="{name}" pos="{start_pos[0]} {start_pos[1]} {start_pos[2]}">')

        # The single monolithic box
        geom_name = f"{name}_geom"
        line = (f'<geom name="{geom_name}" type="box" size="{size_x:.3f} {size_y:.3f} {size_z:.3f}" '
                f'pos="{pos_x:.3f} {pos_y:.3f} {pos_z:.3f}" '
                f'material="solid_ground" friction="2.0 0.05 0.01" contype="1" conaffinity="1"/>')
        xml_lines.append(line)

        xml_lines.append('</body>')
        return "\n".join(xml_lines)
    
    def generate_soft_terrain(self,
                            name='generated_terrain', 
                            n_rows=10, 
                            n_cols=3, 
                            h_base=0.2, 
                            spacing=0.25, 
                            start_pos=(2.5, 0, 1.5)):
        """
        Generates a MuJoCo XML string for a single monolithic block of sponge terrain.
        
        Args:
            n_rows: Length multiplier
            n_cols: Width multiplier
            h_base: Total box thickness
            spacing: Used to scale the total size based on grid parameters
            start_pos: Tuple (x, y, z) for the start of the terrain block
            
            solref="timeconst dampratio": 
            -   timeconst: Increasing this makes the contact softer and slower to bounce back. 
            -   dampratio: Keep this around 1 (critical damping) to avoid unrealistic bouncing.
            solimp="dmin dmax width [midpoint] [power]": 
            -   Defines how "hard" the constraint is depending on penetration depth.
        """
        # 1. Calculate total dimensions based on the grid parameters
        total_length_x = n_rows * spacing
        total_width_y = n_cols * spacing
        
        # 2. MuJoCo box 'size' takes half-extents (half-length, half-width, half-height)
        size_x = total_length_x / 2.0
        size_y = total_width_y / 2.0
        size_z = max(0.01, h_base / 2.0)
        
        # 3. Calculate position relative to body origin
        pos_x = size_x - (spacing / 2.0)
        pos_y = 0.0
        pos_z = size_z

        xml_lines = []
        
        # Body wrapper for the entire terrain block
        xml_lines.append(f'<body name="{name}" pos="{start_pos[0]} {start_pos[1]} {start_pos[2]}">')

        # The single monolithic sponge box
        geom_name = f"{name}_geom"
        line = (f'<geom name="{geom_name}" type="box" size="{size_x:.3f} {size_y:.3f} {size_z:.3f}" '
                f'pos="{pos_x:.3f} {pos_y:.3f} {pos_z:.3f}" '
                f'rgba="0.5 0.647 0 0.5" friction="1.0 0.05 0.01" '
                f'material="soft_ground" solref="0.1 0.7" solimp="0.1 0.99 0.08"/>')
        xml_lines.append(line)

        xml_lines.append('</body>')
        return "\n".join(xml_lines)
    
    def generate_slippery_terrain(self,
                               name='generated_terrain', 
                               n_rows=10, 
                               n_cols=3, 
                               h_base=0.2, 
                               spacing=0.25, 
                               start_pos=(2.5, 0, 1.5)):
        """
        Generates a MuJoCo XML string for a single monolithic block of icy terrain.
        
        Args:
            n_rows: Length multiplier
            n_cols: Width multiplier
            h_base: Total box thickness
            spacing: Used to scale the total size based on grid parameters
            start_pos: Tuple (x, y, z) for the start of the terrain block
        """
        # 1. Calculate total dimensions based on the grid parameters
        total_length_x = n_rows * spacing
        total_width_y = n_cols * spacing
        
        # 2. MuJoCo box 'size' takes half-extents
        size_x = total_length_x / 2.0
        size_y = total_width_y / 2.0
        size_z = max(0.01, h_base / 2.0)
        
        # 3. Calculate position relative to body origin
        pos_x = size_x - (spacing / 2.0)
        pos_y = 0.0
        pos_z = size_z

        xml_lines = []
        
        # Body wrapper for the entire terrain block
        xml_lines.append(f'<body name="{name}" pos="{start_pos[0]} {start_pos[1]} {start_pos[2]}">')

        # --- ICE MODIFICATIONS ---
        # 1. rgba changed to a translucent light blue to look like ice
        # 2. friction lowered massively (0.02 sliding friction)
        # 3. solref and solimp removed so the ice acts like a hard, solid surface
        geom_name = f"{name}_geom"
        line = (f'<geom name="{geom_name}" type="box" size="{size_x:.3f} {size_y:.3f} {size_z:.3f}" '
                f'pos="{pos_x:.3f} {pos_y:.3f} {pos_z:.3f}" '
                f'material="slippery_ground" friction="0.03 0.005 0.0001" />')
        xml_lines.append(line)

        xml_lines.append('</body>')
        
        return "\n".join(xml_lines)
    
    def generate_muddy_terrain(self,
                           name='generated_terrain', 
                           n_rows=10, 
                           n_cols=3, 
                           h_base=0.2, 
                           spacing=0.25, 
                           start_pos=(2.5, 0, 1.5)):
        """
        Generates a MuJoCo XML string for a single monolithic block of muddy terrain.
        
        Args:
            n_rows: Length multiplier
            n_cols: Width multiplier
            h_base: Total box thickness
            spacing: Used to scale the total size based on grid parameters
            start_pos: Tuple (x, y, z) for the start of the terrain block
            
            solref/solimp: Tuned specifically for a softer, viscous, muddy contact dynamic.
        """
        # 1. Calculate total dimensions based on the grid parameters
        total_length_x = n_rows * spacing
        total_width_y = n_cols * spacing
        
        # 2. MuJoCo box 'size' takes half-extents
        size_x = total_length_x / 2.0
        size_y = total_width_y / 2.0
        size_z = max(0.01, h_base / 2.0)
        
        # 3. Calculate position relative to body origin
        pos_x = size_x - (spacing / 2.0)
        pos_y = 0.0
        pos_z = size_z

        xml_lines = []
        
        # Body wrapper for the entire terrain block
        xml_lines.append(f'<body name="{name}" pos="{start_pos[0]} {start_pos[1]} {start_pos[2]}">')

        # The single monolithic muddy box
        geom_name = f"{name}_geom"
        line = (f'<geom name="{geom_name}" type="box" size="{size_x:.3f} {size_y:.3f} {size_z:.3f}" '
                f'pos="{pos_x:.3f} {pos_y:.3f} {pos_z:.3f}" '
                f'rgba="0.445 0.171 0.075 1" friction="1.0 0.05 0.01" '
                f'material="muddy_ground" solref="0.2 3.0" solimp="0.0 0.99 0.15"/>')  
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
        """
        Generates a MuJoCo XML string for a visual-only block of water.
        Physics (buoyancy/drag) must be handled by the Python Hydrodynamics class!
        """
        # 1. Calculate total dimensions
        total_length_x = n_rows * spacing
        total_width_y = n_cols * spacing
        
        # 2. MuJoCo box 'size' takes half-extents
        size_x = total_length_x / 2.0
        size_y = total_width_y / 2.0
        size_z = depth / 2.0
        
        # 3. Calculate position relative to body origin
        pos_x = size_x - (spacing / 2.0)
        pos_y = 0.0
        pos_z = size_z

        xml_lines = []
        
        xml_lines.append(f'<body name="{name}" pos="{start_pos[0]} {start_pos[1]} {start_pos[2]}">')

        # --- WATER MODIFICATIONS ---
        # 1. contype="0" conaffinity="0" : Makes the water a ghost (no rigid collisions)
        # 2. rgba : Translucent blue so you can see the robot swimming underneath
        # 3. Removed friction, solref, and solimp (Ghosts don't have friction!)
        geom_name = f"{name}_geom"
        line = (f'<geom name="{geom_name}" type="box" size="{size_x:.3f} {size_y:.3f} {size_z:.3f}" '
                f'pos="{pos_x:.3f} {pos_y:.3f} {pos_z:.3f}" '
                f'rgba="0.1 0.4 0.8 0.4" contype="0" conaffinity="0" />')  
        xml_lines.append(line)

        xml_lines.append('</body>')
        
        return "\n".join(xml_lines)