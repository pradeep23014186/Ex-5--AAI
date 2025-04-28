<H3>NAME: PRADEEP KUMAR G</H3>
<H3>REGISTER NO: 212223230150</H3>
<H3>EX. NO.5</H3>
<H3>DATE: 28/04/2025</H3>
<H1 ALIGN =CENTER> Implementation of Kalman Filter</H1>
<H3>Aim:</H3> To Construct a Python Code to implement the Kalman filter to predict the position and velocity of an object.
<H3>Algorithm:</H3>
Step 1: Define the state transition model F, the observation model H, the process noise covariance Q, the measurement noise covariance R, the initial state estimate x0, and the initial error covariance P0.<BR>
Step 2:  Create a KalmanFilter object with these parameters.<BR>
Step 3: Simulate the movement of the object for a number of time steps, generating true states and measurements. <BR>
Step 3: For each measurement, predict the next state using kf.predict().<BR>
Step 4: Update the state estimate based on the measurement using kf.update().<BR>
Step 5: Store the estimated state in a list.<BR>
Step 6: Plot the true and estimated positions.<BR>

## Program:
```python

# Developed By: Pradeep Kumar G
# Reg No: 212223230150

import numpy as np
import matplotlib.pyplot as plt

# Kalman Filter class
class KalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F  # State transition matrix
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x0  # Initial state estimate
        self.P = P0  # Initial estimate covariance

    def predict(self):
        self.x = np.dot(self.F, self.x)  # Predict the new state
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q  # Predict the new covariance

    def update(self, z):
        y = z - np.dot(self.H, self.x)  # Measurement residual
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # Residual covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain
        self.x = self.x + np.dot(K, y)  # Update the state estimate
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)  # Update the covariance

# Define parameters
dt = 0.1  # Time step
F = np.array([[1, dt], [0, 1]])  # State transition matrix
H = np.array([[1, 0]])  # Measurement matrix
Q = np.array([[0.1, 0], [0, 0.1]])  # Process noise covariance
R = np.array([[1]])  # Measurement noise covariance
x0 = np.array([0, 0])  # Initial state estimate (position, velocity)
P0 = np.array([[1, 0], [0, 1]])  # Initial estimate covariance

# Create the Kalman Filter object
kf = KalmanFilter(F, H, Q, R, x0, P0)

# Generate true states and noisy measurements
true_states = []
measurements = []
for i in range(100):
    true_states.append([i * dt, 1])  # True position and velocity
    measurements.append(i * dt + np.random.normal(scale=1))  # Noisy measurements

# Estimate states using Kalman filter
estimated_states = []
for z in measurements:
    kf.predict()  # Predict step
    kf.update(np.array([z]))  # Update step with measurement
    estimated_states.append(kf.x)

# Plot the true vs estimated positions
plt.plot([state[0] for state in true_states], label="True Position")
plt.plot([state[0] for state in estimated_states], label="Estimated Position")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.title("Kalman Filter: True vs Estimated Positions")
plt.show()
```
## Output:
![image](https://github.com/user-attachments/assets/900a46d2-18dd-4687-8aaa-d58f2728f167)

## Results:
Thus, Kalman filter is implemented to predict the next position and   velocity in Python
