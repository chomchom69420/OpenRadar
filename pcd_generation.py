import sys
import os
import struct
import time
import numpy as np
import array as arr
import configuration as cfg
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
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

def read8byte(x):
    return struct.unpack('<hhhh', x)

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
    plt.clf()
    plt.close()
    max_index = np.argmax(counts)
    max_count = counts[max_index]
    max_bin_range = (bin_edges[max_index], bin_edges[max_index + 1])
    max_bin_center = (max_bin_range[0] + max_bin_range[1]) / 2

    return max_bin_center

class FrameConfig:  #
    def __init__(self):
        #  configs in configuration.py
        self.numTxAntennas = cfg.NUM_TX
        self.numRxAntennas = cfg.NUM_RX
        self.numLoopsPerFrame = cfg.LOOPS_PER_FRAME
        self.numADCSamples = cfg.ADC_SAMPLES
        self.numAngleBins = cfg.NUM_ANGLE_BINS

        self.numChirpsPerFrame = self.numTxAntennas * self.numLoopsPerFrame
        self.numRangeBins = self.numADCSamples
        self.numDopplerBins = self.numLoopsPerFrame

        # calculate size of one chirp in short.
        self.chirpSize = self.numRxAntennas * self.numADCSamples
        # calculate size of one chirp loop in short. 3Tx has three chirps in one loop for TDM.
        self.chirpLoopSize = self.chirpSize * self.numTxAntennas
        # calculate size of one frame in short.
        self.frameSize = self.chirpLoopSize * self.numLoopsPerFrame


class PointCloudProcessCFG:  #
    def __init__(self):
        self.frameConfig = FrameConfig()
        self.enableStaticClutterRemoval = False
        self.EnergyTop128 = True
        self.RangeCut = False
        self.outputVelocity = True
        self.outputSNR = True
        self.outputRange = True
        self.outputInMeter = True
        self.EnergyThrMed = False
        self.EnergyThrPer95 = True
        self.ConstNoPCD = False
        self.dopplerToLog = False
        self.NoStaticPoints = True

        # 0,1,2 for x,y,z
        dim = 3
        if self.outputVelocity:
            self.velocityDim = dim
            dim += 1
        if self.outputSNR:
            self.SNRDim = dim
            dim += 1
        if self.outputRange:
            self.rangeDim = dim
            dim += 1
        self.couplingSignatureBinFrontIdx = 5
        self.couplingSignatureBinRearIdx = 4
        self.sumCouplingSignatureArray = np.zeros((self.frameConfig.numTxAntennas, self.frameConfig.numRxAntennas,
                                                   self.couplingSignatureBinFrontIdx + self.couplingSignatureBinRearIdx),
                                                  dtype=np.complex128)


class RawDataReader:
    def __init__(self, path):
        self.path = path
        self.ADCBinFile = open(path, 'rb')

    def getNextFrame(self, frameconfig):
        frame = np.frombuffer(self.ADCBinFile.read(frameconfig.frameSize * 4), dtype=np.int16)
        return frame

    def close(self):
        self.ADCBinFile.close()


def bin2np_frame(bin_frame):  #
    np_frame = np.zeros(shape=(len(bin_frame) // 2), dtype=np.complex_)
    np_frame[0::2] = bin_frame[0::4] + 1j * bin_frame[2::4]
    np_frame[1::2] = bin_frame[1::4] + 1j * bin_frame[3::4]
    return np_frame


def frameReshape(frame, frameConfig):  #
    frameWithChirp = np.reshape(frame, (
    frameConfig.numLoopsPerFrame, frameConfig.numTxAntennas, frameConfig.numRxAntennas, -1))
    return frameWithChirp.transpose(1, 2, 0, 3)


def rangeFFT(reshapedFrame, frameConfig):  #
    windowedBins1D = reshapedFrame
    rangeFFTResult = np.fft.fft(windowedBins1D)
    return rangeFFTResult


def clutter_removal(input_val, axis=0):  #
    # Reorder the axes
    reordering = np.arange(len(input_val.shape))
    reordering[0] = axis
    reordering[axis] = 0
    input_val = input_val.transpose(reordering)
    # Apply static clutter removal
    mean = input_val.mean(0)
    output_val = input_val - mean
    return output_val.transpose(reordering)


def dopplerFFT(rangeResult, frameConfig):  #
    windowedBins2D = rangeResult * np.reshape(np.hamming(frameConfig.numLoopsPerFrame), (1, 1, -1, 1))
    dopplerFFTResult = np.fft.fft(windowedBins2D, axis=2)
    dopplerFFTResult = np.fft.fftshift(dopplerFFTResult, axes=2)
    return dopplerFFTResult


def naive_xyz(virtual_ant, num_tx=3, num_rx=4, fft_size=64):  #
    assert num_tx > 2, "need a config for more than 2 TXs"
    num_detected_obj = virtual_ant.shape[1]
    azimuth_ant = virtual_ant[:2 * num_rx, :]
    azimuth_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
    azimuth_ant_padded[:2 * num_rx, :] = azimuth_ant

    # Process azimuth information
    azimuth_fft = np.fft.fft(azimuth_ant_padded, axis=0)
    k_max = np.argmax(np.abs(azimuth_fft), axis=0)
    peak_1 = np.zeros_like(k_max, dtype=np.complex_)
    for i in range(len(k_max)):
        peak_1[i] = azimuth_fft[k_max[i], i]

    k_max[k_max > (fft_size // 2) - 1] = k_max[k_max > (fft_size // 2) - 1] - fft_size
    wx = 2 * np.pi / fft_size * k_max
    x_vector = wx / np.pi

    # Zero pad elevation
    elevation_ant = virtual_ant[2 * num_rx:, :]
    elevation_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
    elevation_ant_padded[:num_rx, :] = elevation_ant

    # Process elevation information
    elevation_fft = np.fft.fft(elevation_ant, axis=0)
    elevation_max = np.argmax(np.log2(np.abs(elevation_fft)), axis=0)  # shape = (num_detected_obj, )
    peak_2 = np.zeros_like(elevation_max, dtype=np.complex_)
    for i in range(len(elevation_max)):
        peak_2[i] = elevation_fft[elevation_max[i], i]

    # Calculate elevation phase shift
    wz = np.angle(peak_1 * peak_2.conj() * np.exp(1j * 2 * wx))
    z_vector = wz / np.pi
    ypossible = 1 - x_vector ** 2 - z_vector ** 2
    y_vector = ypossible
    x_vector[ypossible < 0] = 0
    z_vector[ypossible < 0] = 0
    y_vector[ypossible < 0] = 0
    y_vector = np.sqrt(y_vector)
    return x_vector, y_vector, z_vector

def frame2pointcloud(dopplerResult, rangeResult, pointCloudProcessCFG, selected_range_bins=None):
    
    dopplerResultSumAllAntenna = np.sum(np.abs(dopplerResult), axis=(0, 1))
    if pointCloudProcessCFG.dopplerToLog:
        dopplerResultInDB = np.log10(np.absolute(dopplerResultSumAllAntenna))
    else:
        dopplerResultInDB = np.absolute(dopplerResultSumAllAntenna)

    if pointCloudProcessCFG.RangeCut:  
        dopplerResultInDB[:, :25] = -100
        dopplerResultInDB[:, 125:] = -100
    
    if selected_range_bins is not None:
        mask = np.zeros_like(dopplerResultInDB, dtype=bool)
        mask[:, selected_range_bins] = True
        dopplerResultInDB = dopplerResultInDB * mask

    cfarResult = np.zeros(dopplerResultInDB.shape, bool)
    if pointCloudProcessCFG.EnergyTop128:
        top_size = 128
        energyThre128 = np.partition(dopplerResultInDB.ravel(), 128 * 256 - top_size - 1)[128 * 256 - top_size - 1]
        cfarResult[dopplerResultInDB > energyThre128] = True
    det_peaks_indices = np.argwhere(cfarResult == True)
    R = det_peaks_indices[:, 1].astype(np.float64)
    V = (det_peaks_indices[:, 0] - FrameConfig().numDopplerBins // 2).astype(np.float64)
    if pointCloudProcessCFG.outputInMeter:
        R *= cfg.RANGE_RESOLUTION
        V *= cfg.DOPPLER_RESOLUTION
    energy = dopplerResultInDB[cfarResult == True]
    AOAInput = rangeResult[:, :, cfarResult == True]
    AOAInput = AOAInput.reshape(12, -1)
    if AOAInput.shape[1] == 0:
        return np.array([]).reshape(6, 0)
    x_vec, y_vec, z_vec = naive_xyz(AOAInput)
    x, y, z = x_vec * R, y_vec * R, z_vec * R
    pointCloud = np.concatenate((x, y, z, V, energy, R))
    pointCloud = np.reshape(pointCloud, (6, -1))
    pointCloud = pointCloud[:, y_vec != 0]
    pointCloud = np.transpose(pointCloud, (1, 0))
    if pointCloudProcessCFG.EnergyThrMed:
        idx = np.argwhere(pointCloud[:, 4] > np.median(pointCloud[:, 4])).flatten()
        pointCloud = pointCloud[idx]
    if pointCloudProcessCFG.EnergyThrPer95:
        idx = np.argwhere(pointCloud[:, 4] > np.percentile(pointCloud[:, 4], q=95)).flatten()
        pointCloud = pointCloud[idx]
    if pointCloudProcessCFG.NoStaticPoints:
        idx = np.argwhere(pointCloud[:, 3] != 0).flatten()
        pointCloud = pointCloud[idx]
    if pointCloudProcessCFG.ConstNoPCD:
        pointCloud = reg_data(pointCloud, 128)  

    return pointCloud


# def frame2pointcloud(dopplerResult, pointCloudProcessCFG):
#     dopplerResultSumAllAntenna = np.sum(dopplerResult, axis=(0, 1))
#     if pointCloudProcessCFG.dopplerToLog:
#         dopplerResultInDB = np.log10(np.absolute(dopplerResultSumAllAntenna))
#     else:
#         dopplerResultInDB = np.absolute(dopplerResultSumAllAntenna)

#     if pointCloudProcessCFG.RangeCut:  # filter out the bins which are too close or too far from radar
#         dopplerResultInDB[:, :25] = -100
#         dopplerResultInDB[:, 125:] = -100

#     cfarResult = np.zeros(dopplerResultInDB.shape, bool)
#     if pointCloudProcessCFG.EnergyTop128:
#         top_size = 128
#         energyThre128 = np.partition(dopplerResultInDB.ravel(), 128 * 256 - top_size - 1)[128 * 256 - top_size - 1]
#         cfarResult[dopplerResultInDB > energyThre128] = True

#     det_peaks_indices = np.argwhere(cfarResult == True)
#     R = det_peaks_indices[:, 1].astype(np.float64)
#     V = (det_peaks_indices[:, 0] - FrameConfig().numDopplerBins // 2).astype(np.float64)
#     if pointCloudProcessCFG.outputInMeter:
#         R *= cfg.RANGE_RESOLUTION
#         V *= cfg.DOPPLER_RESOLUTION
#     energy = dopplerResultInDB[cfarResult == True]

#     AOAInput = rangeResult[:, :, cfarResult == True]
#     AOAInput = AOAInput.reshape(12, -1)

#     if AOAInput.shape[1] == 0:
#         return np.array([]).reshape(6, 0)
#     x_vec, y_vec, z_vec = naive_xyz(AOAInput)

#     x, y, z = x_vec * R, y_vec * R, z_vec * R
#     pointCloud = np.concatenate((x, y, z, V, energy, R))
#     pointCloud = np.reshape(pointCloud, (6, -1))
#     pointCloud = pointCloud[:, y_vec != 0]
#     pointCloud = np.transpose(pointCloud, (1, 0))

#     if pointCloudProcessCFG.EnergyThrMed:
#         idx = np.argwhere(pointCloud[:, 4] > np.median(pointCloud[:, 4])).flatten()
#         pointCloud = pointCloud[idx]

#     if pointCloudProcessCFG.ConstNoPCD:
#         pointCloud = reg_data(pointCloud,
#                               128)  # if the points number is greater than 128, just randomly sample 128 points; if the points number is less than 128, randomly duplicate some points

#     return pointCloud


def reg_data(data, pc_size):  #
    pc_tmp = np.zeros((pc_size, 6), dtype=np.float32)
    pc_no = data.shape[0]
    if pc_no < pc_size:
        fill_list = np.random.choice(pc_size, size=pc_no, replace=False)
        fill_set = set(fill_list)
        pc_tmp[fill_list] = data
        dupl_list = [x for x in range(pc_size) if x not in fill_set]
        dupl_pc = np.random.choice(pc_no, size=len(dupl_list), replace=True)
        pc_tmp[dupl_list] = data[dupl_pc]
    else:
        pc_list = np.random.choice(pc_no, size=pc_size, replace=False)
        pc_tmp = data[pc_list]
    return 


if __name__ == '__main__':
    raw_poincloud_data_for_plot = []
    bin_filename = sys.argv[1]
    if len(sys.argv) > 2:
        total_frame_number = int(sys.argv[2])
    else:
        total_frame_number = 3000
    pointCloudProcessCFG = PointCloudProcessCFG()
    shift_arr = cfg.MMWAVE_RADAR_LOC
    bin_reader = RawDataReader(bin_filename)
    frame_no = 0
    for frame_no in range(total_frame_number):
        bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
        np_frame = bin2np_frame(bin_frame)
        frameConfig = pointCloudProcessCFG.frameConfig
        reshapedFrame = frameReshape(np_frame, frameConfig)
        rangeResult = rangeFFT(reshapedFrame, frameConfig)
        if pointCloudProcessCFG.enableStaticClutterRemoval:
            rangeResult = clutter_removal(rangeResult, axis=2)

        dopplerResult = dopplerFFT(rangeResult, frameConfig)
        pointCloud = frame2pointcloud(dopplerResult, rangeResult, pointCloudProcessCFG)
        frame_no += 1
        # print('Frame %d:' % (frame_no), pointCloud.shape, pointCloud[0])
        # if frame_no == 50:
        # phi_list = []
        # for j in range(pointCloud.shape[0]-1):
            # phi_list.append(calc_phi(pointCloud[j], pointCloud[j+1]))
        # print(np.argmax(phi_list))
        # find_phi(phi_list, '', '')
        if frame_no==100:
            clusters = sdetect.static_clusters(pointCloud)
        # print(f"Frame = {frame_no}, clusters = {clusters}")
    
        raw_poincloud_data_for_plot.append(pointCloud)
    bin_reader.close()