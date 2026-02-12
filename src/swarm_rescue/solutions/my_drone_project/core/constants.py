# core/constants.py

# --- GRID & MAP ---
RESOLUTION = 10
ROBOT_RADIUS_PIXELS = 30
SEUIL_MUR = 3.0
SEUIL_FREE = -5.0
SEUIL_UNEXPLORED_MAX = 2.99
SEUIL_UNEXPLORED_MIN = -4.99

# --- PID CONTROLLER (Fidèle au prototype) ---
PID_ROT_KP = 5.0
PID_ROT_KD = 3.0
PID_POS_KP = 6.0
PID_POS_KD = 11.0

# --- NAVIGATION ---
PATH_LOOKAHEAD = 35.0
MAX_SPEED = 22.0
COMFORT_DISTANCE_WORLD = 200.0  # Distance de sécurité par rapport aux murs

# --- DETECTION ---
DEAD_DRONE_RADIUS = 60.0
DEAD_CONFIRM_ITERATIONS = 5

# Paramètres de navigation avancés du prototype
PATH_LOOKAHEAD_DEFAULT = 40.0
CLOSE_THRESHOLD = 50.0  # Distance pour valider un waypoint
MAX_SPEED_EXPLORATION = 12.0

# --- SLAM & LIDAR ---
MAX_RANGE_LIDAR_SENSOR = 1000.0  # Ajustez selon la config de la simulation

# --- DEATH DETECTION ---
DEATH_TIMEOUT = 100         # itérations avant suspicion
CONFIRMATION_TIMEOUT = 50   # itérations avant confirmation
MAX_COMM_RANGE = 200.0      # portée max radio