import numpy as np
import math
import matplotlib.pyplot as plt

def normalize_angle(angle_rad: float) -> float:
    """
    Normalise un angle en radians pour qu'il soit dans [-pi, pi].
    """
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi

def calculate_ekf_quality(estimated_poses_list, true_poses_list):
    """
    Calcule le Root Mean Squared Error (RMSE) pour l'estimation EKF.
    """
    if not estimated_poses_list or not true_poses_list:
        print("Erreur : Listes de poses vides. Avez-vous lancé la simulation ?")
        return

    # Convertir les listes en arrays NumPy pour le calcul vectorisé
    est_poses = np.array(estimated_poses_list)
    true_poses = np.array(true_poses_list)

    if est_poses.shape != true_poses.shape:
        print(f"Erreur : Les données de pose n'ont pas la même taille. "
              f"Estimé: {est_poses.shape}, Vrai: {true_poses.shape}")
        min_len = min(est_poses.shape[0], true_poses.shape[0])
        est_poses = est_poses[:min_len]
        true_poses = true_poses[:min_len]

    # 1. Calculer l'erreur
    error = est_poses - true_poses

    # 2. Gérer la normalisation de l'angle
    error[:, 2] = [normalize_angle(e) for e in error[:, 2]]

    # 3. Calculer l'erreur quadratique (Squared Error)
    squared_error = error**2

    # 4. Calculer l'erreur quadratique moyenne (Mean Squared Error)
    mean_squared_error = np.mean(squared_error, axis=0)

    # 5. Calculer la racine de l'erreur quadratique moyenne (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error)

    # 6. Afficher le rapport
    print("--- Rapport de Qualité EKF (RMSE) ---")
    print(f"RMSE Position X: {rmse[0]:.4f} (pixels)")
    print(f"RMSE Position Y: {rmse[1]:.4f} (pixels)")
    print(f"RMSE Angle:      {rmse[2]:.4f} (radians)")
    print("---------------------------------------")
    
    return rmse

def plot_x_comparison(estimated_poses_list, true_poses_list):
    """
    Crée un graphique Matplotlib comparant la position X vraie
    à la position X estimée (EKF) au fil du temps.
    """
    if not estimated_poses_list or not true_poses_list:
        print("Erreur : Impossible de tracer, listes de poses vides.")
        return

    # Convertir les listes en arrays NumPy
    est_poses = np.array(estimated_poses_list)
    true_poses = np.array(true_poses_list)

    # Tronquer à la taille la plus courte pour la comparaison
    min_len = min(est_poses.shape[0], true_poses.shape[0])
    est_poses = est_poses[:min_len]
    true_poses = true_poses[:min_len]

    # Extraire les données X
    true_x = true_poses[:, 0]
    est_x = est_poses[:, 0]
    
    # Créer un axe de temps (nombre de pas de simulation)
    time = np.arange(min_len)

    # Créer le graphique
    plt.figure(figsize=(12, 6))
    plt.plot(time, true_x, 'b-', label='True X Position (Ground Truth)')
    plt.plot(time, est_x, 'r--', label='Estimated X Position (EKF)')
    plt.xlabel('Simulation Step')
    plt.ylabel('X Position (pixels)')
    plt.title('EKF Performance Comparison: X-Axis')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- EXEMPLE D'UTILISATION ---
if __name__ == "__main__":
    
    # --- CELA SERAIT REMPLI PAR VOTRE SIMULATION ---
    # Par exemple, après l'exécution de votre simulation :
    # sim_drone = run_simulation() 
    # estimated_poses = sim_drone.estimated_poses
    # true_poses = sim_drone.true_poses
    # ---------------------------------------------

    # --- Pour tester, voici des données factices ---
    print("Test avec des données factices...")
    
    # Simuler une trajectoire avec du bruit
    steps = 400
    dummy_true_list = []
    dummy_est_list = []
    true_val = 0.0
    est_val = 0.0
    
    for i in range(steps):
        true_val += 0.5  # Mouvement constant
        noise = np.random.randn() * 0.5  # Bruit de mesure
        drift = (i / steps) * 0.3 # Petite dérive de l'estimation
        est_val = true_val + noise + drift
        
        # Ajouter des données factices pour y et angle (non utilisées dans ce graphique)
        dummy_true_list.append(np.array([true_val, true_val * 0.5, 0.1]))
        dummy_est_list.append(np.array([est_val, est_val * 0.5 + noise, 0.12]))

    # 1. Calculer le RMSE
    calculate_ekf_quality(dummy_est_list, dummy_true_list)
    
    # 2. Afficher le graphique
    plot_x_comparison(dummy_est_list, dummy_true_list)