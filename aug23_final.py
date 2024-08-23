import numpy as np
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from scipy.stats import chi2

# Define lists to store results
r = []
el = []
az = []

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))  # Predicted state vector
        self.Pp = np.eye(6)  # Predicted state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.prev_Time = 0
        self.Q = np.eye(6)
        self.Phi = np.eye(6)
        self.Z = np.zeros((3, 1)) 
        self.Z1 = np.zeros((3, 1)) # Measurement vector
        self.Z2 = np.zeros((3, 1)) 
        self.first_rep_flag = False
        self.second_rep_flag = False
        self.gate_threshold = 9000.21  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        if not self.first_rep_flag:
            self.Z1 = np.array([[x], [y], [z]])
            self.Sf[0] = x
            self.Sf[1] = y
            self.Sf[2] = z
            self.Meas_Time = time
            self.prev_Time = self.Meas_Time
            self.first_rep_flag = True
        elif self.first_rep_flag and not self.second_rep_flag:
            self.Z2 = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time
            dt = self.Meas_Time - self.prev_Time
            self.vx = (self.Z1[0] - self.Z2[0]) / dt
            self.vy = (self.Z1[1] - self.Z2[1]) / dt
            self.vz = (self.Z1[2] - self.Z2[2]) / dt

            self.Meas_Time = time
            self.second_rep_flag = True
        else:
            self.Z = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time

    def predict_step(self, current_time):
        dt = current_time - self.prev_Time
        T_2 = (dt*dt)/2.0
        T_3 = (dt*dt*dt)/3.0
        self.Phi[0, 3] = dt
        self.Phi[1, 4] = dt
        self.Phi[2, 5] = dt
              
        self.Q[0, 0] = T_3
        self.Q[1, 1] = T_3
        self.Q[2, 2] = T_3
        self.Q[0, 3] = T_2
        self.Q[1, 4] = T_2
        self.Q[2, 5] = T_2
        self.Q[3, 0] = T_2
        self.Q[4, 1] = T_2
        self.Q[5, 2] = T_2
        self.Q[3, 3] = dt
        self.Q[4, 4] = dt
        self.Q[5, 5] = dt
        self.Q = self.Q * self.plant_noise
        self.Sp = np.dot(self.Phi, self.Sf)
        self.Pp = np.dot(np.dot(self.Phi, self.Pf), self.Phi.T) + self.Q
        self.Meas_Time = current_time

    def update_step(self, Z):
        Inn = Z - np.dot(self.H, self.Sp)
        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pp, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sp + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pp)
        # print("Updated filter state:", self.Sf.flatten())

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            r, az, el = cart2sph(x, y, z)  # Convert Cartesian to spherical coordinates
            measurements.append((r, az, el, mt))
    return measurements

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    az = math.atan2(y, x)

    if x > 0.0:
        az = np.pi / 2 - az
    else:
        az = 3 * np.pi / 2 - az

    az = az * 180 / np.pi

    if az < 0.0:
        az = 360 + az

    if az > 360:
        az = az - 360

    return r, az, el

def cart2sph2(x: float, y: float, z: float, filtered_values_csv):
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i] / np.sqrt(x[i]**2 + y[i]**2)) * 180 / 3.14)
        az.append(math.atan(y[i] / x[i]))

        if x[i] > 0.0:                
            az[i] = 3.14 / 2 - az[i]
        else:
            az[i] = 3 * 3.14 / 2 - az[i]       

        az[i] = az[i] * 180 / 3.14 

        if az[i] < 0.0:
            az[i] = 360 + az[i]

        if az[i] > 360:
            az[i] = az[i] - 360   
      
    return r, az, el

def form_measurement_groups(measurements, max_time_diff=0.050):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]

    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]

    if current_group:
        measurement_groups.append(current_group)

    return measurement_groups

def generate_hypotheses(tracks, reports):
    hypotheses = []
    for i, track in enumerate(tracks):
        for j, report in enumerate(reports):
            if i == j:
                hypotheses.append([(i, j)])
    return hypotheses

def calculate_probabilities(hypotheses, tracks, reports, cov_inv):
    probabilities = []
    for hypothesis in hypotheses:
        prob = 1.0
        for track_idx, report_idx in hypothesis:
            track = tracks[track_idx]
            report = reports[report_idx]
            residual = np.array(report) - np.array(track)
            prob *= np.exp(-0.5 * np.dot(np.dot(residual.T, cov_inv), residual))
        probabilities.append(prob)
    return probabilities

def get_association_weights(hypotheses, probabilities, num_tracks):
    # Initialize the association_weights list with empty lists for each track
    association_weights = [[] for _ in range(num_tracks)]

    for hypothesis, prob in zip(hypotheses, probabilities):
        for track_idx, report_idx in hypothesis:
            if track_idx < len(association_weights):
                association_weights[track_idx].append((report_idx, prob))
            else:
                print(f"Error: track_idx {track_idx} is out of range.")
    
    return association_weights

def form_clusters_via_association(tracks, reports, kalman_filter, chi2_threshold):
    clusters = []
    for track_idx, track in enumerate(tracks):
        for report_idx, report in enumerate(reports):
            # Calculate Mahalanobis distance
            residual = np.array(report) - np.array(track)
            cov_inv = np.linalg.inv(kalman_filter.Pf[:3, :3])  # Use 3x3 covariance matrix for position only
            dist = np.dot(np.dot(residual.T, cov_inv), residual)
            if dist <= chi2_threshold:
                clusters.append((tracks, reports))
    return clusters

def main():
    # File path for measurements CSV
    file_path = 'ttk.csv'

    # Read measurements from CSV
    measurements = read_measurements_from_csv(file_path)
    
    kalman_filter = CVFilter()
    
    csv_file_predicted = "ttk.csv"
    df_predicted = pd.read_csv(csv_file_predicted)
    filtered_values_csv = df_predicted[['F_TIM', 'F_X', 'F_Y', 'F_Z']].values
    measured_values_csv = df_predicted[['MT', 'MR', 'MA', 'ME']].values

    A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3], filtered_values_csv)
    number = 1000

    result = np.divide(A[0], number)              

    # Form measurement groups based on time intervals less than 50 milliseconds
    measurement_groups = form_measurement_groups(measurements, max_time_diff=0.050)

    t = []
    rnge = []
    azme = []
    elem = []

    # Process each group of measurements
    for group_idx, group in enumerate(measurement_groups):
        print(f"Processing measurement group {group_idx + 1}...")

        # List of tracks and reports (in Cartesian coordinates)
        tracks = []
        reports = []

        for i, (rng, azm, ele, mt) in enumerate(group):
            print(f"\nMeasurement {i + 1}: (az={azm}, el={ele}, r={rng}, t={mt})\n")

            x, y, z = sph2cart(azm, ele, rng)

            if not kalman_filter.first_rep_flag:
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
                print("Initialized Filter state:", kalman_filter.Sf.flatten())
                continue

            elif kalman_filter.first_rep_flag and not kalman_filter.second_rep_flag:
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
                print("Initialized Filter state 2nd M:", kalman_filter.Sf.flatten())
            else:
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
                
                kalman_filter.predict_step(mt)
                reports.append((x, y, z))  # Append as a tuple of Cartesian coordinates

                # Use the Kalman filter state as the track
                tracks.append(kalman_filter.Sf[:3].flatten())

        # Form clusters using the association logic
        clusters = form_clusters_via_association(tracks, reports, kalman_filter, chi2_threshold=kalman_filter.gate_threshold)
        print("Clusters formed:", clusters)  # Print clusters
        print("Number of clusters:", len(clusters))

        # Process each cluster and perform JPDA
        for cluster_tracks, cluster_reports in clusters:
            if cluster_tracks and cluster_reports:
                print(f"Cluster Reports: {cluster_reports}")  # Debugging: Print cluster reports to inspect structure

                # Calculate inverse of covariance matrix
                cov_inv = np.linalg.inv(kalman_filter.Pf[:3, :3])  # Use 3x3 covariance matrix for position only

                # Generate hypotheses
                hypotheses = generate_hypotheses(cluster_tracks, cluster_reports)
                print("Generated Hypotheses:", hypotheses)  # Print hypotheses

                # Calculate probabilities
                probabilities = calculate_probabilities(hypotheses, cluster_tracks, cluster_reports, cov_inv)
                print("Probabilities:", probabilities)  # Print probabilities

                # Get association weights
                num_tracks = len(tracks)  # Replace 'tracks' with the actual variable representing your tracks
                association_weights = get_association_weights(hypotheses, probabilities, num_tracks)
                print("Association Weights:", association_weights)  # Print association weights

                # Determine the best hypothesis for each track based on association weights
                for track_idx, weights in enumerate(association_weights):
                    if not weights:
                        print(f"No valid reports for Track t{track_idx + 1}. Skipping...")
                        continue

                    # Find the report with the highest probability
                    max_prob = 0.0
                    best_report_idx = -1
                    for report_idx, weight in weights:
                        if weight > max_prob:
                            max_prob = weight
                            best_report_idx = report_idx

                    if best_report_idx != -1:
                        print(f"Most likely association for Track t{track_idx + 1}: Report r{best_report_idx + 1}, weight: {max_prob}")
                        
                        # Retrieve the best report from the cluster reports
                        best_report = cluster_reports[best_report_idx]
                        print("Best Report:", best_report)
                        
                        # Prepare the measurement vector Z for the Kalman filter update
                        Z = np.array([[best_report[0]], [best_report[1]], [best_report[2]]])
                        
                        print("Measurement Vector Z:", Z)

                        # Perform the Kalman filter update step with the selected report
                        kalman_filter.update_step(Z)
                        print("Updated filter state:", kalman_filter.Sf.flatten())

                        # Save the updated state for plotting or further processing
                        r_val, az_val, el_val = cart2sph(kalman_filter.Sf[0], kalman_filter.Sf[1], kalman_filter.Sf[2])
                        t.append(mt)
                        rnge.append(r_val)
                        azme.append(az_val)
                        elem.append(el_val)
                    else:
                        print(f"No valid report found for Track t{track_idx + 1}. Skipping...")

    # Plotting remains unchanged
    plt.figure(figsize=(12, 6))
    plt.subplot(facecolor="white")
    plt.scatter(t, rnge, label='filtered range (code)', color='green', marker='*')
    plt.scatter(filtered_values_csv[:, 0], result, label='filtered range (track id 31)', color='red', marker='*')
    plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 1], label='measured range (code)', color='blue', marker='o')
    plt.xlabel('Time', color='black')
    plt.ylabel('Range (r)', color='black')
    plt.title('Range vs. Time', color='black')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)
    plt.show()

    # Plot azimuth (az) vs. time
    plt.figure(figsize=(12, 6))
    plt.subplot(facecolor="white")
    plt.scatter(t, azme, label='filtered azimuth (code)', color='green', marker='*')
    plt.scatter(filtered_values_csv[:, 0], A[1], label='filtered azimuth (track id 31)', color='red', marker='*')
    plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 2], label='measured azimuth (code)', color='blue', marker='o')
    plt.xlabel('Time', color='black')
    plt.ylabel('Azimuth (az)', color='black')
    plt.title('Azimuth vs. Time', color='black')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)
    plt.show()

    # Plot elevation (el) vs. time
    plt.figure(figsize=(12, 6))
    plt.subplot(facecolor="white")
    plt.scatter(t, elem, label='filtered elevation (code)', color='green', marker='*')
    plt.scatter(filtered_values_csv[:, 0], A[2], label='filtered elevation (track id 31)', color='red', marker='*')
    plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 3], label='measured elevation (code)', color='blue', marker='o')
    plt.xlabel('Time', color='black')
    plt.ylabel('Elevation (el)', color='black')
    plt.title('Elevation vs. Time', color='black')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)
    plt.show()

if __name__ == "__main__":
    main()
