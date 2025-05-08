# Copyright 2019 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import mmwave as mm
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
from mmwave.tracking import EKF
from mmwave.tracking import gtrack_visualize
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image
import sys 
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import statistics
from scipy import stats

# Class to detect static object
class DetectStatic:
    def __init__(self,vel_eps=0.025, xyz_std=1, min_points=50):
        self.min_max_scaler=MinMaxScaler()
        self.vel_scanner = DBSCAN(eps=vel_eps, min_samples=5)
        self.xy_scanner= lambda e: (np.linalg.norm(e[:,:2].std(axis=0))<xyz_std) and (e.shape[0]>min_points)
        
    def static_clusters(self,pointCloud): #col_vec of vel
        #Find phi
        phi_list = []
        for j in range(pointCloud.shape[0]-1):
            phi_list.append(calc_phi(pointCloud[j], pointCloud[j+1]))
        phi = find_phi(phi_list, '', '')
        
        self.vel_scanner.fit(
            self.min_max_scaler.fit_transform(
                pointCloud[:,[3]] / np.cos(phi - np.arctan(pointCloud[:,[0]]/pointCloud[:,[1]]))  #vd / cos(phi - tan-1(x/y))
            )
        )
        clusters=self.vel_scanner.labels_
        # print(f'clusters = {clusters}')
        unique_cids=np.unique(clusters)
        # print(f"unique_cids = {unique_cids}")
        #e[0]-->ucid, e[1]--> pointCloud
        # return dict(filter(lambda e: self.xy_scanner(e[1]),{ucid:pointCloud[np.where(clusters==ucid)] for ucid in unique_cids}.items()))
        # print(f"cluster_dict = {dict({ucid:pointCloud[np.where(clusters==ucid)] for ucid in unique_cids}.items())}")
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_cids)))
        cluster_dict = dict({ucid:pointCloud[np.where(clusters==ucid)] for ucid in unique_cids}.items())
        vel_estimates = (pointCloud[:,[3]] / np.cos(phi- np.arctan(pointCloud[:,[0]]/pointCloud[:,[1]]))).T[0]
        plt.hist(pointCloud[:,[3]] / np.cos(phi- np.arctan(pointCloud[:,[0]]/pointCloud[:,[1]])), bins=50, range=(-6, 6))
        print(f"Mode = {statistics.mode(vel_estimates)}")
        # mu, std = stats.norm.fit(vel_estimates)
        # p = stats.norm.pdf(np.arange(-6, 6, 0.1), mu, std)
        # plt.plot(np.arange(-6, 6, 0.1), 400*p, c='r')
        # print(f"mu, sigma = {mu, std}")
        kde = stats.gaussian_kde(vel_estimates)
        x_vals = np.linspace(min(vel_estimates), max(vel_estimates), 1000)
        pdf_vals = kde(x_vals)
        mode_kde = x_vals[np.argmax(pdf_vals)]
        cdf_vals = np.cumsum(pdf_vals)
        cdf_vals /= cdf_vals[-1] 
        mask = cdf_vals >= (1 - 0.95) 
        selected_x = x_vals[mask]
        selected_pdf = pdf_vals[mask]
        mean_kde = np.sum(selected_x * selected_pdf) / np.sum(selected_pdf)
        variance_kde = np.sum((selected_x - mean_kde) ** 2 * selected_pdf) / np.sum(selected_pdf)
        std_kde = np.sqrt(variance_kde)
        print(f"KDE Mode: {mode_kde}")
        plt.plot(x_vals, pdf_vals, label="KDE")
        plt.fill_between(selected_x, selected_pdf, alpha=0.5, color="green", label="Top 95% Density")
        plt.axvline(mean_kde, color='r', linestyle='--', label="Mean (Top 95%)")
        plt.axvline(mean_kde + std_kde, color='g', linestyle='--', label="+1 Std Dev")
        plt.axvline(mean_kde - std_kde, color='g', linestyle='--', label="-1 Std Dev")
        plt.legend()
        static_mask = (mean_kde - std_kde < vel_estimates) & (vel_estimates < mean_kde + std_kde) 
        static_points = pointCloud[static_mask]
        dynamic_points = pointCloud[~static_mask]
        # plt.xlabel("Value")
        # plt.ylabel("Density")
        # plt.title("Gaussian KDE with Top 95% Density Selection")
        # plt.show()
        plt.savefig('vel_hist.png')
        plt.clf()
        plt.close()
        
        plt.scatter(static_points[:, 0], static_points[:, 1], color='r', label=f"static", s=10)
        plt.scatter(dynamic_points[:, 0], dynamic_points[:, 1], color='g', label=f"dynamic", s=10)
        # cluster_dict = dict(filter(lambda e: self.xy_scanner(e[1]),{ucid:pointCloud[np.where(clusters==ucid)] for ucid in unique_cids}.items()))
        # for cid, color in zip(unique_cids, colors):
        #     # cluster_points = pointCloud[np.where(clusters == cid)]
        #     cluster_points = cluster_dict[cid]
        #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
        #     color=color, label=f"Cluster {cid}", s=10)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Point Cloud Clusters")
        plt.legend()
        plt.savefig(fname='cluster_static_dynamic.png', dpi=300)
        return dict({ucid:pointCloud[np.where(clusters==ucid)] for ucid in unique_cids}.items())

sdetect = DetectStatic()

def calc_phi(point1, point2):
    """
    Calcs phi : angle between the radar boarside axis and the velocity vector

    Args:
    point1 [x, y, z, V, energy, R]
    point2 [x, y, z, V, energy, R]
    """
    # theta1 = np.arctan2(point1[0], point1[1])
    # theta2 = np.arctan2(point2[0], point2[1])
    theta1 = np.arctan(point1[0] / point1[1])
    theta2 = np.arctan(point2[0] / point2[1])
    # print(f"theta1, theta2 = {theta1}, {theta2}")
    a = point2[3]*np.cos(theta1) - point1[3]*np.cos(theta2)
    b = point1[3]*np.sin(theta2) - point2[3]*np.sin(theta1)
    # phi = np.arctan2(a, b)
    phi = np.arctan(a/b)
    return phi

def calc_phi_object(obj1, obj2):
    """
    Calcs phi : angle between the radar boarside axis and the velocity vector

    Args:
    obj1 [x, y, x_vel, y_vel]: One object in the frame
    obj2 [x, y, x_vel, y_vel]: Another object in the frame
    """
    v_dop1 = np.sqrt(obj1[2]**2 + obj1[3]**2)
    v_dop2 = np.sqrt(obj2[2]**2 + obj2[3]**2)
    theta1 = np.arctan2(obj1[0], obj1[1])
    theta2 = np.arctan2(obj2[0], obj2[1])
    a = v_dop2*np.cos(theta1) - v_dop1*np.cos(theta2)
    b = v_dop1*np.sin(theta2) - v_dop2*np.sin(theta1)
    phi = np.arctan2(a, b)
    
    return phi

def find_phi(list_of_values,x_label,y_label):
    counts, bin_edges, patches = plt.hist(list_of_values, bins=100, range=(-1.2*np.pi, 1.2*np.pi))
    # plt.savefig('hist.png')
    plt.clf()
    plt.close()
    max_index = np.argmax(counts)
    max_count = counts[max_index]
    max_bin_range = (bin_edges[max_index], bin_edges[max_index + 1])
    max_bin_center = (max_bin_range[0] + max_bin_range[1]) / 2

    return max_bin_center


# Radar specific parameters
NUM_RX = 4
VIRT_ANT = 12

# Data specific parameters
NUM_CHIRPS = 182
NUM_ADC_SAMPLES = 256
RANGE_RESOLUTION, bandwidth = dsp.range_resolution(NUM_ADC_SAMPLES, dig_out_sample_rate=4400, freq_slope_const=60.012)
DOPPLER_RESOLUTION = dsp.doppler_resolution(bandwidth)
NUM_FRAMES = 151

# DSP processing parameters
SKIP_SIZE = 4
ANGLE_RES = 1
ANGLE_RANGE = 90
ANGLE_BINS = (ANGLE_RANGE * 2) // ANGLE_RES + 1
BINS_PROCESSED = 256

# Read in adc data file
load_data = True
if load_data:
    adc_data = np.fromfile('./dataset/bot_2025-02-11_20_26_36_6static_2dynamic_objectsday1_1__only_sensor.bin', dtype=np.uint16)    
    adc_data = adc_data.reshape(NUM_FRAMES, -1)
    all_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=NUM_CHIRPS*3, num_rx=NUM_RX, num_samples=NUM_ADC_SAMPLES)

# Convert the tracker.points object into our pointCloud format 
def convert_object_to_pcd(points):
    """Convert the tracker.points object into our pointCloud format

        Args:
            points: tracker.points object (self.range, self.angle, self.snr, self.doppler)  

        Returns:
            pointCloud: (#points, 6): (x, y, z, V, energy, R)
            
        Note: 
            Here, z is taken as 0, since we are only interested in 2D scatter. energy is replaced with SNR since their object does not contain energy values. 
    """
    
    n = len(points)
    pointCloud = np.zeros((n, 6))
    for i in range(len(points)):
        point = points[i]
        x = - point.range * np.sin(point.angle)
        y = point.range * np.cos(point.angle)
        z = 0
        V = point.doppler
        energy = point.snr
        R = point.range
        pointCloud[i, :] = (x, y, z, V, energy, R)
        
    return pointCloud


def plot_objects(x_positions,y_positions,k):
    #by SohamC
    num_objects=len(x_positions)
    
    x_positions = [1000 * p for p in x_positions]
    y_positions = [1000 * p for p in y_positions]


    plt.figure(figsize=(8, 6))
    plt.scatter(x_positions, y_positions, color='red', label='Objects')

    # Setting limits
    plt.xlim(-1000, 2500)
    plt.ylim(-1000, 6000)

    # Display number of objects
    plt.text(0,5000, f'N_obj: {num_objects}', fontsize=12, color='blue')

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Object Positions')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'humara_tracking/{k}.png')
    plt.clf()
    plt.close()

def plot_histogram(list_of_values,x_lable,y_label):
    plt.hist(list_of_values)
    plt.savefig('hist.png')
    plt.clf()
    plt.close()
    
# Start DSP processing
range_azimuth = np.zeros((ANGLE_BINS, BINS_PROCESSED))
num_vec, steering_vec = dsp.gen_steering_vec(ANGLE_RANGE, ANGLE_RES, VIRT_ANT)
tracker = EKF()
k=1
for frame in all_data:    
    """ 1 (Range Processing) """

    # --- range fft
    radar_cube = dsp.range_processing(frame)

    """ 2 (Capon Beamformer) """

    # --- static clutter removal
    mean = radar_cube.mean(0)                 
    radar_cube = radar_cube - mean            

    # --- capon beamforming
    beamWeights   = np.zeros((VIRT_ANT, BINS_PROCESSED), dtype=np.complex_)
    radar_cube = np.concatenate((radar_cube[0::3, ...], radar_cube[1::3, ...], radar_cube[2::3, ...]), axis=1)
    # Note that when replacing with generic doppler estimation functions, radarCube is interleaved and
    # has doppler at the last dimension.
    for i in range(BINS_PROCESSED):
        try:
            range_azimuth[:,i], beamWeights[:,i] = dsp.aoa_capon(radar_cube[:, :, i].T, steering_vec, magnitude=True)
        except Exception as e:
            continue
    
    """ 3 (Object Detection) """
    heatmap_log = np.log2(range_azimuth)
    
    # --- cfar in azimuth direction
    first_pass, _ = np.apply_along_axis(func1d=dsp.ca_,
                                        axis=0,
                                        arr=heatmap_log,
                                        l_bound=1.5,
                                        guard_len=4,
                                        noise_len=16)
    
    # --- cfar in range direction
    second_pass, noise_floor = np.apply_along_axis(func1d=dsp.ca_,
                                                   axis=0,
                                                   arr=heatmap_log.T,
                                                   l_bound=2.5,
                                                   guard_len=4,
                                                   noise_len=16)

    # --- classify peaks and caclulate snrs
    noise_floor = noise_floor.T
    first_pass = (heatmap_log > first_pass)
    second_pass = (heatmap_log > second_pass.T)
    peaks = first_pass #(first_pass & second_pass)
    peaks[:SKIP_SIZE, :] = 0
    peaks[-SKIP_SIZE:, :] = 0
    peaks[:, :SKIP_SIZE] = 0
    peaks[:, -SKIP_SIZE:] = 0
    pairs = np.argwhere(peaks)
    azimuths, ranges = pairs.T
    snrs = heatmap_log[pairs[:,0], pairs[:,1]] - noise_floor[pairs[:,0], pairs[:,1]]

    """ 4 (Doppler Estimation) """

    # --- get peak indices
    # beamWeights should be selected based on the range indices from CFAR.
    dopplerFFTInput = radar_cube[:, :, ranges]
    beamWeights  = beamWeights[:, ranges]

    # --- estimate doppler values
    # For each detected object and for each chirp combine the signals from 4 Rx, i.e.
    # For each detected object, matmul (numChirpsPerFrame, numRxAnt) with (numRxAnt) to (numChirpsPerFrame)
    dopplerFFTInput = np.einsum('ijk,jk->ik', dopplerFFTInput, beamWeights)
    if not dopplerFFTInput.shape[-1]:
        continue
    dopplerEst = np.fft.fft(dopplerFFTInput, axis=0)
    dopplerEst = np.argmax(dopplerEst, axis=0)
    dopplerEst[dopplerEst[:]>=NUM_CHIRPS/2] -= NUM_CHIRPS
    
    """ 5 (Extended Kalman Filter) """

    # --- convert bins to units
    ranges = ranges * RANGE_RESOLUTION
    azimuths = (azimuths - (ANGLE_BINS // 2)) * (np.pi / 180)
    dopplers = dopplerEst * DOPPLER_RESOLUTION
    snrs = snrs
    # print("azimuths: ", azimuths.shape)
    
    # --- put into EKF
    tracker.update_point_cloud(ranges, azimuths, dopplers, snrs)
    targetDescr, tNum = tracker.step()
    # print(int(tNum[0]),len(ranges))
    x_positions=[]
    y_positions=[]
    x_velocities = []
    y_velocities = []
    for t, tid in zip(targetDescr, range(int(tNum[0]))):
        x_pos, y_pos,x_vel,y_vel= t.S[:4]
        # print(f'x_vel: {x_vel} y_vel: {y_vel}')
        x_positions.append(x_pos)
        y_positions.append(y_pos)
        x_velocities.append(x_vel)
        y_velocities.append(y_vel)
    # plot_objects(x_positions,y_positions,k)
    k+=1
    for p in range(len(x_positions)-1):
        obj1 = [x_positions[p], y_positions[p], x_velocities[p], y_velocities[p]]
        obj2 = [x_positions[p+1], y_positions[p+1], x_velocities[p+1], y_velocities[p+1]]
        # print(f"phi = {calc_phi_object(obj1, obj2)}")
    

    
#     """ 6 (Visualize Output) """
#     frame = gtrack_visualize.get_empty_frame()
#     try:
        # frame = gtrack_visualize.update_frame(targetDescr, int(tNum[0]), frame)
#     except:
#         pass
    frame = gtrack_visualize.draw_points(tracker.point_cloud, len(ranges), frame)
#     # print(frame.shape)
#     # if not gtrack_visualize.show(frame, wait=1):
#     #     break
    
#     # if k==50:
    # print("Point Cloud data: tracker.point_cloud", tracker.point_cloud[0])
    pointCloud = convert_object_to_pcd(tracker.point_cloud)
    # print(f'pointCloud.shape = {pointCloud.shape}')
    # clusters = sdetect.static_clusters(pointCloud)
    # print(clusters)
    
    if k==100:
        clusters = sdetect.static_clusters(pointCloud)
        # break
    
#     if k==50:
#         image = Image.fromarray(frame)
#         image.save('pcd.png')

#     if k>50:
#         break
#     k=k+1
    
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
    
# cv2.destroyAllWindows()