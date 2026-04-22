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
                               h_base=0.0, 
                               spacing=0.25, 
                               start_pos=(2.5, 0, 1.5)):
        """
        Generates a MuJoCo XML string for a block of flat terrain.
        
        Args:
            n_rows: Number of boxes along X (path length)
            n_cols: Number of boxes along Y (path width)
            box_size: Half-length of the box sides
            h_base: Minimum box thickness
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
                random_h = max(0.01, h_base + np.random.uniform(0, 0))
                # MuJoCo boxes expand from the center, so z is half the total height
                z = random_h / 2
                line = (f'<geom name="{geom_name}" type="box" size="{box_size} {box_size} {random_h:.3f}" '
                        f'pos="{x:.2f} {y:.2f} {z:.3f}" '
                        f'rgba="0.1 0.0 0.2 1" friction="2.0 0.05 0.01"/>')
                xml_lines.append(line)

        xml_lines.append('</body>')
        return "\n".join(xml_lines)
    
    def generate_sponge_terrain(self,
                               name='generated_terrain', 
                               n_rows=10, 
                               n_cols=3, 
                               box_size=0.125, 
                               h_base=0.2, 
                               spacing=0.25, 
                               start_pos=(2.5, 0, 1.5)):
        """
        Args:
            n_rows: Number of boxes along X (path length)
            n_cols: Number of boxes along Y (path width)
            box_size: Half-length of the box sides
            h_base: Minimum box thickness
            spacing: Distance between box centers
            start_pos: Tuple (x, y, z) for the start of the terrain block

            solref="timeconst dampratio": 
            -   timeconst: Increasing this makes the contact softer and slower to bounce back. 
                The default is usually around 0.02. Try pushing it to 0.05 or 0.1 for a noticeable sponge effect.
            -   dampratio: Keep this around 1 (critical damping) to avoid unrealistic bouncing 
                unless you want a highly elastic surface like a trampoline.
            solimp="dmin dmax width [midpoint] [power]": 
            -   This defines how "hard" the constraint is depending on how deeply the robot's foot penetrates the floor. 
                To make a surface softer, decrease dmin (e.g., to 0.001 or lower) and increase the width so the transition 
                from soft to hard is more gradual.
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
                random_h = max(0.01, h_base + np.random.uniform(0, 0))
                # MuJoCo boxes expand from the center, so z is half the total height
                z = random_h / 2
                line = (f'<geom name="{geom_name}" type="box" size="{box_size} {box_size} {random_h:.3f}" '
                        f'pos="{x:.2f} {y:.2f} {z:.3f}" '
                        f'rgba="0.5 0.647 0 0.5" friction="1.0 0.05 0.01"  solref="0.1 5.0"  solimp="0.001 0.99 0.05 0.5 2"/>')
                xml_lines.append(line)

        xml_lines.append('</body>')
        return "\n".join(xml_lines)
    
    def generate_sandy_terrain(self,
                               name='generated_terrain', 
                               n_rows=10, 
                               n_cols=3, 
                               box_size=0.125, 
                               h_base=0.2, 
                               spacing=0.25, 
                               start_pos=(2.5, 0, 1.5)):
        """
        Args:
            n_rows: Number of boxes along X (path length)
            n_cols: Number of boxes along Y (path width)
            box_size: Half-length of the box sides
            h_base: Minimum box thickness
            spacing: Distance between box centers
            start_pos: Tuple (x, y, z) for the start of the terrain block

            solref="timeconst dampratio": 
            -   timeconst: Increasing this makes the contact softer and slower to bounce back. 
                The default is usually around 0.02. Try pushing it to 0.05 or 0.1 for a noticeable sponge effect.
            -   dampratio: Keep this around 1 (critical damping) to avoid unrealistic bouncing 
                unless you want a highly elastic surface like a trampoline.
            solimp="dmin dmax width [midpoint] [power]": 
            -   This defines how "hard" the constraint is depending on how deeply the robot's foot penetrates the floor. 
                To make a surface softer, decrease dmin (e.g., to 0.001 or lower) and increase the width so the transition 
                from soft to hard is more gradual.
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
                random_h = max(0.01, h_base + np.random.uniform(0, 0))
                # MuJoCo boxes expand from the center, so z is half the total height
                z = random_h / 2
                line = (f'<geom name="{geom_name}" type="box" size="{box_size} {box_size} {random_h:.3f}" '
                        f'pos="{x:.2f} {y:.2f} {z:.3f}" '
                        f'rgba="0.957 0.643 0.376 1" friction="1.0 0.05 0.01" solref="0.04 1.1" solimp="0.01 0.99 0.03"/>')
                xml_lines.append(line)

        xml_lines.append('</body>')
        return "\n".join(xml_lines)
    
    def generate_muddy_terrain(self,
                               name='generated_terrain', 
                               n_rows=10, 
                               n_cols=3, 
                               box_size=0.125, 
                               h_base=0.2, 
                               spacing=0.25, 
                               start_pos=(2.5, 0, 1.5)):
        """
        Args:
            n_rows: Number of boxes along X (path length)
            n_cols: Number of boxes along Y (path width)
            box_size: Half-length of the box sides
            h_base: Minimum box thickness
            spacing: Distance between box centers
            start_pos: Tuple (x, y, z) for the start of the terrain block

            solref="timeconst dampratio": 
            -   timeconst: Increasing this makes the contact softer and slower to bounce back. 
                The default is usually around 0.02. Try pushing it to 0.05 or 0.1 for a noticeable sponge effect.
            -   dampratio: Keep this around 1 (critical damping) to avoid unrealistic bouncing 
                unless you want a highly elastic surface like a trampoline.
            solimp="dmin dmax width [midpoint] [power]": 
            -   This defines how "hard" the constraint is depending on how deeply the robot's foot penetrates the floor. 
                To make a surface softer, decrease dmin (e.g., to 0.001 or lower) and increase the width so the transition 
                from soft to hard is more gradual.
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
                random_h = max(0.01, h_base + np.random.uniform(0, 0))
                # MuJoCo boxes expand from the center, so z is half the total height
                z = random_h / 2
                line = (f'<geom name="{geom_name}" type="box" size="{box_size} {box_size} {random_h:.3f}" '
                        f'pos="{x:.2f} {y:.2f} {z:.3f}" '
                        f'rgba="0.445 0.171 0.075 1" friction="1.0 0.05 0.01" solref="0.1 1.5" solimp="0.9 0.99 0.001 0.5 2"/>')
                xml_lines.append(line)

        xml_lines.append('</body>')
        return "\n".join(xml_lines)
