import numpy as np

class KalmanFilter:
    def __init__(self, initial_pos):
        self.state = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0])
        self.P = np.eye(4) * 100.0
        self.Q = np.eye(4) * 8.0
        self.Q[2, 2] = 1.0; self.Q[3, 3] = 1.0
        self.R = np.eye(2) * 10.0
        self.initialized = False

    def predict(self, dt, measured_vel=None):
        if not self.initialized: return
        
        # Propagation
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        
        # Mise à jour vitesse estimée si dispo
        if measured_vel is not None:
            self.state[2] = 0.7 * measured_vel[0] + 0.3 * self.state[2]
            self.state[3] = 0.7 * measured_vel[1] + 0.3 * self.state[3]

    def update(self, gps_pos):
        if not self.initialized:
            self.state[0], self.state[1] = gps_pos
            self.initialized = True
            return
            
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        z = np.array(gps_pos)
        y = z - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

    def get_pos(self):
        return self.state[:2]