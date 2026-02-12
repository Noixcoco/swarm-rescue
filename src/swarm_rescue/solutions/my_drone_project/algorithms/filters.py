# algorithms/filters.py
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, generate_binary_structure

def find_frontier_clusters(grid_map, grid_obj):
    """
    Version optimisée pour encourager l'exploration des zones inconnues
    et éviter les couloirs déjà lourdement explorés.
    """
    # SEUILS RELAXÉS (Prototype)
    SEUIL_FREE = -3.0
    SEUIL_MUR = 6.0
    SEUIL_UNEXPLORED_MIN = -2.99
    SEUIL_UNEXPLORED_MAX = 5.99
    
    # Masques de base
    is_unknown = (grid_map >= SEUIL_UNEXPLORED_MIN) & (grid_map <= SEUIL_UNEXPLORED_MAX)
    is_wall = (grid_map >= SEUIL_MUR)
    is_free = (grid_map < SEUIL_FREE)
    is_heavily_explored = (grid_map < -40.0) # Exclure le couloir bleu foncé

    # Détection des voisins de l'inconnu
    structure_neigh = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=bool)
    unknown_neighbors = binary_dilation(is_unknown, structure=structure_neigh)
    
    # Masque des frontières : libre ET proche inconnu ET PAS lourdement exploré
    frontier_mask = is_free & (~is_heavily_explored) & unknown_neighbors

    # Marge de sécurité réduite pour les couloirs étroits
    struct_safety = np.ones((5, 5), dtype=bool)
    danger_zone = binary_dilation(is_wall, structure=struct_safety, iterations=1)
    frontier_mask = frontier_mask & (~danger_zone)

    # Clustering
    struct_label = generate_binary_structure(2, 2)
    labeled, num_features = ndimage.label(frontier_mask, structure=struct_label)

    frontier_clusters = []
    min_cluster_size = 3

    for label_idx in range(1, num_features + 1):
        ys, xs = np.where(labeled == label_idx)
        size = ys.size
        if size < min_cluster_size:
            continue

        mean_y = float(np.mean(ys))
        mean_x = float(np.mean(xs))
        
        # Conversion monde
        x_world, y_world = grid_obj._conv_grid_to_world(mean_y, mean_x)
        
        # VALIDATION : Analyse du voisinage immédiat (fenêtre de 15)
        window = 15
        by, bx = int(mean_y), int(mean_x)
        y0, y1 = max(0, by - window), min(grid_map.shape[0], by + window)
        x0, x1 = max(0, bx - window), min(grid_map.shape[1], bx + window)
        neighborhood = grid_map[y0:y1, x0:x1]
        
        # 1. Rejeter si trop peu de cellules inexplorées à proximité
        unexplored_nearby = np.sum((neighborhood >= SEUIL_UNEXPLORED_MIN) & 
                                   (neighborhood <= SEUIL_UNEXPLORED_MAX))
        if unexplored_nearby < 10:
            continue
            
        # 2. Rejeter si le voisinage est à 70%+ déjà lourdement exploré
        heavily_explored_nearby = np.sum(neighborhood < -15.0)
        if heavily_explored_nearby > 0.7 * neighborhood.size:
            continue
            
        frontier_clusters.append({
            "barycenter": np.array([x_world, y_world]),
            "size": int(size)
        })

    # Tri par taille décroissante
    frontier_clusters.sort(key=lambda c: c["size"], reverse=True)
    return frontier_clusters