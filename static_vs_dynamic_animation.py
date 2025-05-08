
import mmwave as mm
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
from mmwave.tracking import EKF
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys

# fig, ax = plt.subplots(figsize=(8, 6))
# static_points_label = ax.scatter([], [], color='r', s=10)
# dynamic_points_label = ax.scatter([], [], color='g', label="Dynamic", s=10)
# ax.set_xlim(-10, 10)  # Adjust limits based on your data
# ax.set_ylim(0, 10)
# ax.grid(True)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.legend()

    
def static_clusters(pointCloud, frame_no): #col_vec of vel
    phi_list = []
    std_dev_mult_factor = 1
    for j in range(pointCloud.shape[0]-1):
        phi_list.append(calc_phi(pointCloud[j], pointCloud[j+1]))
    phi = find_phi(phi_list, '', '')
    angle = np.arctan2(pointCloud[:,[0]], pointCloud[:,[1]])
    angle = np.where(angle > np.pi/2, angle - np.pi, angle)
    angle = np.where(angle < -np.pi/2, angle + np.pi, angle)
    vel_estimates = (pointCloud[:,[3]] / np.cos(phi- angle)).T[0]
    # print("vel_estimates: ", vel_estimates)
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
    
    #Plotting 
    if frame_no+1 == 89:
        plt.figure()
        # counts, bin_edges, _ = np.histogram(vel_estimates, bins=50)
        counts, bins, _ = plt.hist(vel_estimates, bins=40, density=False)
        # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # bin_widths =  np.diff(bin_edges) * 0.8 
        # plt.bar(bin_centers, counts, width=bin_widths)
        # kde_scaled = pdf_vals * max(counts) / max(pdf_vals)  # Scale KDE to histogram counts    
        # plt.plot(x_vals, pdf_vals, 'r-', label='KDE')
        # plt.axvline(mean_kde, color='r', linestyle='--', label='Mean (Top 95%)')
        plt.axvline(mode_kde, color='r', linestyle='--', lw=4, label="Mode")
        # plt.axvline(mean_kde + std_kde, color='g', linestyle='--', label='+1 Std Dev')
        # plt.axvline(mean_kde - std_kde, color='g', linestyle='--', label='-1 Std Dev')
        plt.axvline(mode_kde + std_kde, color='g', linestyle='--', lw=4, label=r'Mode $\pm 1 \sigma$')
        plt.axvline(mode_kde - std_kde, color='g', linestyle='--', lw=4) #, label='Mode -1 Std Dev')
        # plt.fill_between(selected_x, selected_pdf, alpha=0.3, color='lightgreen', label='Top 95% Density')
        # plt.twinx()  # Create secondary y-axis
        # plt.plot(x_vals, pdf_vals * max(counts) / max(pdf_vals), color='r', label="KDE")
        plt.ylabel("No. of occurances")
        plt.xlabel("Speed (m/s)")
        plt.xlim((-30, 30))
        plt.xticks([-30, -15, 0, 15, 30])
        # plt.title(f'Frame = {frame_no+1}')
        plt.legend(ncol=2, fontsize=19, loc=(0,1.01))
        plt.tight_layout()
        plt.grid()
        plt.savefig(fname=f'vel_histograms/hist_{frame_no+1}.png', dpi=300)
        sys.exit()
    
    static_mask = (mean_kde - std_dev_mult_factor * std_kde < vel_estimates) & (vel_estimates < mean_kde + std_dev_mult_factor * std_kde) 
    static_points = pointCloud[static_mask]
    dynamic_points = pointCloud[~static_mask]
    return static_points, dynamic_points


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


def find_phi(list_of_values,x_label,y_label):
    counts, bin_edges, patches = plt.hist(list_of_values, bins=100, range=(-1.2*np.pi, 1.2*np.pi))
    # plt.savefig('hist.png')
    # plt.clf()
    # plt.close()
    max_index = np.argmax(counts)
    max_count = counts[max_index]
    max_bin_range = (bin_edges[max_index], bin_edges[max_index + 1])
    max_bin_center = (max_bin_range[0] + max_bin_range[1]) / 2

    return max_bin_center


NUM_RX = 4
VIRT_ANT = 12

NUM_CHIRPS = 182
NUM_ADC_SAMPLES = 256
RANGE_RESOLUTION, bandwidth = dsp.range_resolution(NUM_ADC_SAMPLES, dig_out_sample_rate=4400, freq_slope_const=60.012)
DOPPLER_RESOLUTION = dsp.doppler_resolution(bandwidth)

SKIP_SIZE = 4
ANGLE_RES = 1
ANGLE_RANGE = 90
ANGLE_BINS = (ANGLE_RANGE * 2) // ANGLE_RES + 1
BINS_PROCESSED = 256

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
        # print(f"Range = {point.range}")
        x = - point.range * np.sin(point.angle)
        y = point.range * np.cos(point.angle)
        z = 0
        V = point.doppler
        energy = point.snr
        R = point.range
        pointCloud[i, :] = (x, y, z, V, energy, R)
        
    return pointCloud


def get_pcd(frame_no, all_data, tracker, range_azimuth, num_vec, steering_vec):
    frame = all_data[frame_no]
    # --- range fft
    radar_cube = dsp.range_processing(frame)

    """ 2 (Capon Beamformer) """

    # --- static clutter removal
    mean = radar_cube.mean(0)                 
    radar_cube = radar_cube - mean            

    # --- capon beamforming
    beamWeights   = np.zeros((VIRT_ANT, BINS_PROCESSED), dtype=np.complex_)
    radar_cube = np.concatenate((radar_cube[0::3, ...], radar_cube[1::3, ...], radar_cube[2::3, ...]), axis=1)
    
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

    noise_floor = noise_floor.T
    first_pass = (heatmap_log > first_pass)
    second_pass = (heatmap_log > second_pass.T)
    peaks = (first_pass & second_pass)
    # print(f"peaks.shape = {peaks.shape}")
    # print(f"Range resolution = {RANGE_RESOLUTION}")
    peaks[:SKIP_SIZE, :] = 0            #applying skip on angle 
    peaks[-SKIP_SIZE:, :] = 0            
    #peaks[:, :SKIP_SIZE] = 0            #applying skip on range    
    #peaks[:, -SKIP_SIZE:] = 0
    peaks[:, int(-4/RANGE_RESOLUTION):] = 0
    peaks[:, :int(0.1/RANGE_RESOLUTION)] = 0
    pairs = np.argwhere(peaks)
    azimuths, ranges = pairs.T
    snrs = heatmap_log[pairs[:,0], pairs[:,1]] - noise_floor[pairs[:,0], pairs[:,1]]

    dopplerFFTInput = radar_cube[:, :, ranges]
    beamWeights  = beamWeights[:, ranges]

    dopplerFFTInput = np.einsum('ijk,jk->ik', dopplerFFTInput, beamWeights)
    dopplerEst = np.fft.fft(dopplerFFTInput, axis=0)
    dopplerEst = np.argmax(dopplerEst, axis=0)
    dopplerEst[dopplerEst[:]>=NUM_CHIRPS/2] -= NUM_CHIRPS
    
    ranges = ranges * RANGE_RESOLUTION
    azimuths = (azimuths - (ANGLE_BINS // 2)) * (np.pi / 180)
    dopplers = dopplerEst * DOPPLER_RESOLUTION
    snrs = snrs
    tracker.update_point_cloud(ranges, azimuths, dopplers, snrs)
    targetDescr, tNum = tracker.step()
    
    pointCloud = convert_object_to_pcd(tracker.point_cloud)
    static_points, dynamic_points = static_clusters(pointCloud, frame_no)
    return static_points, dynamic_points

# def update(frame_no):
#     static_points, dynamic_points = get_pcd(frame_no, all_data)
#     # if len(static_points) == 0 or len(dynamic_points) == 0:
#     #     return static_points_label, dynamic_points_label,    
#     # static_points_label.set_offsets([]) 
#     # dynamic_points_label.set_offsets([])
#     # static_points_label.set_offsets(static_points[:, :2])  # X, Y for static points
#     # dynamic_points_label.set_offsets(dynamic_points[:, :2])  
#     plt.clf()
#     plt.close()
#     fig, ax = plt.subplots(figsize=(8, 6))
#     plt.scatter(static_points[:, 0], static_points[:, 1], color='r', label=f"static", s=10)
#     plt.scatter(dynamic_points[:, 0], dynamic_points[:, 1], color='g', label=f"dynamic", s=10)  
    
#     plt.xlim(-10, 10)  # Adjust limits based on your data
#     plt.ylim(0, 10)
#     plt.grid(True)
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.legend()
#     plt.tight_layout()
#     fname = f'static_dynamic/{frame_no}.png'
#     plt.savefig(fname)

# Create pictures for animation 
# for frame_no in range(NUM_FRAMES):
#     update(frame_no)

# ani = animation.FuncAnimation(fig, update, frames=range(NUM_FRAMES), interval=200, blit=True)
# ani.save('animation.gif', writer='pillow', fps=200)
# exit(1)
# ani.save('animation.gif', writer='pillow', fps=30)

def export_points_framewise(filename, n_frames):
    NUM_FRAMES = n_frames
    
    # Load the data
    adc_data = np.fromfile(f'./dataset/{filename}', dtype=np.uint16)    
    adc_data = adc_data.reshape(NUM_FRAMES, -1)
    all_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=NUM_CHIRPS*3, num_rx=NUM_RX, num_samples=NUM_ADC_SAMPLES)
    
    range_azimuth = np.zeros((ANGLE_BINS, BINS_PROCESSED))
    num_vec, steering_vec = dsp.gen_steering_vec(ANGLE_RANGE, ANGLE_RES, VIRT_ANT)
    tracker = EKF()
    
    static_pcd = []
    dynamic_pcd = []
    for frame_no in range(NUM_FRAMES):
        static_points, dynamic_points = get_pcd(frame_no, all_data, tracker, range_azimuth, num_vec, steering_vec)
        static_pcd.append(static_points)
        dynamic_pcd.append(dynamic_points)
        
    return static_pcd, dynamic_pcd
    
    

