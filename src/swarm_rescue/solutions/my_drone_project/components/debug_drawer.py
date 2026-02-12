# src/swarm_rescue/solutions/my_drone_project/components/debug_drawer.py
import arcade
import numpy as np
import math

class DebugDrawer:
    def __init__(self, drone_id):
        self.drone_id = int(drone_id)
        # Palette identique au prototype
        self.palette = [
            (200, 60, 60),   # Red
            (60, 200, 60),   # Green
            (60, 60, 200),   # Blue
            (200, 200, 60),  # Yellow
            (200, 60, 200),  # Magenta
            (60, 200, 200),  # Cyan
            (255, 128, 0),   # Orange
            (128, 0, 255),   # Purple
        ]
        self.detection_color = self.palette[self.drone_id % len(self.palette)]

    def draw(self, pose, half_size_array, state_name, navigator, mapper, perception):
        """
        Version fidèle au prototype pour le rendu graphique de l'arène.
        """
        # 1. DESSIN DU CHEMIN (NAVIGATOR)
        path = navigator.current_path
        if path and len(path) > 0:
            # Points du chemin
            for pt in path:
                pt_scr = pt + half_size_array
                arcade.draw_circle_filled(pt_scr[0], pt_scr[1], radius=7, color=(0, 0, 255))
            # Lignes de liaison
            for i in range(len(path) - 1):
                p1 = path[i] + half_size_array
                p2 = path[i+1] + half_size_array
                arcade.draw_line(p1[0], p1[1], p2[0], p2[1], color=(0, 255, 0), line_width=3)

        # 2. DESSIN DES FRONTIÈRES (MAPPER)
        if hasattr(mapper, 'frontier_clusters') and mapper.frontier_clusters:
            # Tri et affichage des 5 plus proches (comme le prototype)
            sorted_clusters = sorted(
                mapper.frontier_clusters,
                key=lambda c: np.linalg.norm(c['barycenter'] - pose[:2])
            )[:5]
            
            for cluster in sorted_clusters:
                bc = cluster.get('barycenter')
                if bc is not None:
                    pt_scr = bc + half_size_array
                    arcade.draw_circle_filled(pt_scr[0], pt_scr[1], radius=6, color=self.detection_color)

        # 3. DESSIN DES BLESSÉS ET ASSIGNATIONS (PERCEPTION)
        if hasattr(perception, 'wounded') and perception.wounded:
            for (xw, yw) in perception.wounded:
                pt_scr = np.array([xw, yw]) + half_size_array
                
                # Logique de correspondance d'assignation
                assigned_id = None
                for w_pos, d_id in perception.assignments.items():
                    # Gestion des clés String (comm) ou Tuple
                    if isinstance(w_pos, str):
                        try:
                            coords = [float(x) for x in w_pos.strip("()").split(",")]
                            kx, ky = coords[0], coords[1]
                        except: continue
                    else:
                        kx, ky = w_pos[0], w_pos[1]
                    
                    if math.hypot(xw - kx, yw - ky) < 10.0:
                        assigned_id = d_id
                        break
                
                if assigned_id is not None:
                    color = self.palette[int(assigned_id) % len(self.palette)]
                    label = f"ASSIGNED: DRONE {assigned_id}"
                else:
                    color = (255, 255, 255)
                    label = "AVAILABLE"

                arcade.draw_circle_outline(pt_scr[0], pt_scr[1], 20, color, 2)
                arcade.draw_text(label, pt_scr[0] + 25, pt_scr[1] - 10, color, 11, bold=True)

        # 4. DESSIN DES ZONES DE SECOURS (RESCUE ZONES)
        if hasattr(perception, 'rescue_zones'):
            for (xr, yr) in perception.rescue_zones:
                pt_scr = np.array([xr, yr]) + half_size_array
                arcade.draw_rectangle_outline(pt_scr[0], pt_scr[1], 30, 30, (0, 160, 0), 2)
                arcade.draw_text("RZ", pt_scr[0] + 12, pt_scr[1] + 12, (0, 120, 0), 10)

        # 5. DESSIN DES ZONES DE RETOUR (RETURN AREA)
        if hasattr(mapper, 'return_area_points'):
            for (rx, ry) in mapper.return_area_points:
                pt_scr = np.array([rx, ry]) + half_size_array
                arcade.draw_rectangle_outline(pt_scr[0], pt_scr[1], 30, 30, (0, 160, 255), 2)
                arcade.draw_text("RA", pt_scr[0] + 12, pt_scr[1] + 12, (0, 120, 200), 10)

        # 6. POSITION DU DRONE ET ÉTAT
        my_scr = pose[:2] + half_size_array
        # Le gros rond du drone
        arcade.draw_circle_filled(my_scr[0], my_scr[1], radius=18, color=self.detection_color)
        
        # Le texte d'état au-dessus
        try:
            arcade.draw_text(state_name, 
                           my_scr[0] - 30, 
                           my_scr[1] + 25, 
                           (255, 255, 255), 
                           12, 
                           bold=True)
        except: pass