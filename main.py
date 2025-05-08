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

import sys
import gc
import numpy as np
import mmwave.dsp as dsp
import mmwave.clustering as clu
from mmwave.dataloader import DCA1000
from demo.visualizer.visualize import ellipse_visualize
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

plt.close('all')

#Filename
fname = "bot_2025-02-11_19_47_42_5objectsday1__only_sensor"
# fname = "2024-11-04_16-42-23_surrounding3"

# QOL settings
loadData = True
loadData_surr = False

numFrames = 301
numADCSamples = 256
numTxAntennas = 3
numRxAntennas = 4
numLoopsPerFrame = 182
numChirpsPerFrame = numTxAntennas * numLoopsPerFrame
numVirtAntennas = numTxAntennas * numRxAntennas
frameTime = 150 #ms
movieFPS = 1000/frameTime

numRangeBins = numADCSamples
numDopplerBins = numLoopsPerFrame
numAngleBins = 64

caponAngleRes = 1 #degrees
caponAngleRange = 90
numCaponAngleBins = (caponAngleRange * 2) // caponAngleRes + 1
rangeBinStartProcess = 0
rangeBinEndProcess = 10
numRangeBinsProcessed = rangeBinEndProcess - rangeBinStartProcess + 1

range_resolution, bandwidth = dsp.range_resolution(numADCSamples, dig_out_sample_rate=4400, freq_slope_const=60.012)
max_range = dsp.max_range(dig_out_sample_rate=4400, freq_slope_const=60.012)

doppler_resolution = dsp.doppler_resolution(bandwidth)

targetRangeBin = 8
targetAzimuthBin = 74

plotRangeAzimuth = False
plotAzimuth1D = False
plotRangeAzimuthTimeSeries = True
plotRangeDopp = False  
plot2DscatterXY = True  
plot2DscatterXZ = False  
plot3Dscatter = False  
plotCustomPlt = False

plotSurroundingRemoved = False
plotMakeMovie = False
makeMovieTitle = " "
makeMovieDirectory = "./range_angle.mp4"

visTrigger = plot2DscatterXY + plot2DscatterXZ + plot3Dscatter + plotRangeDopp + plotCustomPlt + plotRangeAzimuth + plotAzimuth1D + plotRangeAzimuthTimeSeries
assert visTrigger < 2, "Can only choose to plot one type of plot at once"

singFrameView = False
singFrameNumber = 40

#For plotting AxesImage objects (imshow)
def update(frame, data, img):
    
    # Add colorbar only on the first frame
    if frame > 0:
        img.set_data(data[:,:, frame])
    
    return [img]

def movieMaker(fig, ims, title, save_dir):
    # Set up formatting for the Range Azimuth heatmap movies
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

    plt.title(title)
    print('Done')
    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=False)
    print('Check')
    im_ani.save(save_dir, writer=writer)
    print('Complete')


if __name__ == '__main__':
    ims = []
    max_size = 0
    ra_timeSeries = []

    # (1) Reading in adc data
    if loadData:
        adc_data = np.fromfile(f"./dataset/{fname}.bin", dtype=np.uint16)
        adc_data = adc_data.reshape(numFrames, -1)
        adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame,
                                       num_rx=numRxAntennas, num_samples=numADCSamples)
        print(f'adc_data shape: {adc_data.shape}')
        print("Data Loaded!")
        
    if loadData_surr:
        adc_data_surr = np.fromfile(f"./dataset/{fname_surr}.bin", dtype=np.uint16)
        adc_data_surr = adc_data_surr.reshape(numFrames, -1)
        adc_data_surr = np.apply_along_axis(DCA1000.organize, 1, adc_data_surr, num_chirps=numChirpsPerFrame,
                                       num_rx=numRxAntennas, num_samples=numADCSamples)
        print("Surrounding Data Loaded!")
        
    # (1.5) Required Plot Declarations
    if plot2DscatterXY or plot2DscatterXZ:
        fig, axes = plt.subplots(1, 2)
    elif plot3Dscatter and plotMakeMovie:
        fig = plt.figure()
        nice = Axes3D(fig)
    elif plot3Dscatter:
        fig = plt.figure()
    elif plotRangeDopp:
        fig = plt.figure()
    elif plotRangeAzimuth:
        # Set up y and x ticks only once
        y_ticks = list(range(0, numCaponAngleBins * caponAngleRes, 15))
        y_labels = [str(j - caponAngleRange // caponAngleRes) for j in y_ticks]
        x_ticks = list(range(0, numRangeBinsProcessed, int(np.ceil(0.1*numRangeBinsProcessed))))
        x_labels = [str(j+rangeBinStartProcess) for j in x_ticks]
        
        if loadData_surr:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
            ax1.set_yticks(y_ticks)
            ax1.set_yticklabels(y_labels)
            ax1.set_xticks(x_ticks)
            ax1.set_xticklabels(x_labels, rotation=45, ha="right")
            ax2.set_yticks(y_ticks)
            ax2.set_yticklabels(y_labels)
            ax2.set_xticks(x_ticks)
            ax2.set_xticklabels(x_labels, rotation=45, ha="right")
            ax1.set_title("back radar")
            ax2.set_title("surrounding")
        else:
            fig, ax= plt.subplots()
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
    elif plotAzimuth1D:
        x_ticks = list(range(0, numCaponAngleBins, 10))
        x_labels = [str((j - caponAngleRange//caponAngleRes)*caponAngleRes) for j in x_ticks] 
        if loadData_surr:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5)) 
            for ax in [ax1, ax2]:
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_labels)
            ax1.set_title("back radar")
            ax2.set_title("surrounding")
        else:
            fig, ax = plt.subplots()
            plt.xticks(ticks=x_ticks, labels=x_labels)
    elif plotRangeAzimuthTimeSeries:
        fig, ax = plt.subplots()
    elif plotCustomPlt:
        print("Using Custom Plotting")

    # (1.6) Optional single frame view
    # if singFrameView:
    #     dataCube = np.zeros((1, numChirpsPerFrame, 4, 128), dtype=complex)
    #     dataCube[0, :, :, :] = adc_data[numFrames-1]
    # else:
    #     dataCube = adc_data
    dataCube = adc_data

    range_azimuth = np.zeros((numCaponAngleBins, numRangeBinsProcessed))
    range_azimuth_all_frames = np.zeros((numCaponAngleBins, numRangeBinsProcessed, adc_data.shape[0]))
    num_vec, steering_vec = dsp.gen_steering_vec(ang_est_range=caponAngleRange, ang_est_resolution=caponAngleRes, num_ant=numVirtAntennas)
    
    # ---- Load surrounding range azimuth
    if loadData_surr:
        range_azimuth_surr = np.zeros((numCaponAngleBins, numRangeBinsProcessed))
    
    for i, frame in enumerate(dataCube):
        # (2) Range Processing
        from mmwave.dsp.utils import Window
        
        if loadData_surr:
            frame_surr = adc_data_surr[i,:,:,:]
            radar_cube_surr = dsp.range_processing(frame_surr, window_type_1d=Window.HANNING)

        radar_cube = dsp.range_processing(frame, window_type_1d=Window.HANNING)
        assert radar_cube.shape == (
        numChirpsPerFrame, numRxAntennas, numADCSamples), "[ERROR] Radar cube is not the correct shape!"
        
        #Capon beamforming
        beamWeights   = np.zeros((numVirtAntennas, numRangeBinsProcessed), dtype=np.complex_)
        radar_cube_aoa = np.concatenate((radar_cube[0::3, ...], radar_cube[1::3, ...], radar_cube[2::3, ...]), axis=1)
        
        if loadData_surr:
            radar_cube_aoa_surr = np.concatenate((radar_cube_surr[0::3, ...], radar_cube_surr[1::3, ...], radar_cube_surr[2::3, ...]), axis=1)
        
        # Note that when replacing with generic doppler estimation functions, radarCube is interleaved and
        # has doppler at the last dimension.
        for k in range(numRangeBinsProcessed):
            j = k + rangeBinStartProcess
            range_azimuth[:,k], beamWeights[:,k] = dsp.aoa_capon(radar_cube_aoa[:, :, j].T, steering_vec, magnitude=True)
            
            # --- Store range azimuth for surrounding
            if loadData_surr:
                range_azimuth_surr[:,k], _ = dsp.aoa_capon(radar_cube_aoa_surr[:, :, j].T, steering_vec, magnitude=True)
                
        # --- Append timeseries values
        if plotRangeAzimuthTimeSeries:
            ra_timeSeries.append(range_azimuth[targetAzimuthBin, targetRangeBin])
            
        # --- Store the range_azimuths for plotting later 
        if plotRangeAzimuth and plotMakeMovie:
            if rangeBinStartProcess == 0 and rangeBinEndProcess>5:
                range_azimuth[:, :5] = 0
            
            if loadData_surr:
                range_azimuth_surr[:,:5]=0
                range_azimuth = np.absolute(range_azimuth - range_azimuth_surr)
            
            range_azimuth_all_frames[:,:,i] = range_azimuth
        
        # --- Plot range azimuth
        if plotRangeAzimuth and not plotMakeMovie:
            if rangeBinStartProcess == 0 and rangeBinEndProcess>5:
                range_azimuth[:, :5] = 0
            
            # --- Plot surrounding-removed range-azimuth
            if loadData_surr:
                if rangeBinStartProcess == 0 and rangeBinEndProcess>5:
                    range_azimuth_surr[:,:5]=0
                # range_azimuth = np.absolute(range_azimuth - range_azimuth_surr)
            
            if loadData_surr:
                for ax in [ax1, ax2]:
                    for coll in ax.collections:
                        coll.remove() 
            else:
                for coll in ax.collections:
                    coll.remove()  
                    
            fig.suptitle(f"Range-Azimuth plot {i}")
            
            if i==0:                
                if loadData_surr:
                    im1 = ax1.imshow(range_azimuth, aspect="auto", interpolation="nearest", animated=True)
                    im2 = ax2.imshow(range_azimuth_surr, aspect="auto", interpolation="nearest", animated=True)
                    cbar1 = fig.colorbar(im1,ax=ax1, orientation='vertical')
                    cbar2 = fig.colorbar(im2,ax=ax2, orientation='vertical')
                else:
                    img = ax.imshow(range_azimuth, aspect="auto", interpolation="nearest", animated=True)
                    ax.set_xlabel("Range bin")
                    ax.set_ylabel("Angle")
                    fig.colorbar(img, ax=ax, orientation="vertical")
            else:
                if loadData_surr:
                    im1.set_data(range_azimuth)
                    im2.set_data(range_azimuth_surr)
                else:
                    img.set_data(range_azimuth)
                    
            if singFrameView and i == singFrameNumber:
                # Save the current figure and exit
                plt.savefig(f"results\\imgs\\ra_{fname}_{i}.pdf", dpi=300)
                plt.close('all')
                gc.collect()
                sys.exit()
            
            if not singFrameView:    
                plt.pause(frameTime/1000)   
        
        # --- Plot range azimuth 1D for particular range bin
        if plotAzimuth1D:
            
            if plotMakeMovie:
                ims.append(plt.plot(range_azimuth[:, int(targetRangeBin)-rangeBinStartProcess], 'blue'))
                continue
            
            if loadData_surr:
                line1, = ax1.plot(range_azimuth[:, targetRangeBin], 'blue')
                line2, = ax2.plot(range_azimuth_surr[:, targetRangeBin], 'blue')
            else:
                plt.plot(range_azimuth[:, targetRangeBin])
                plt.xticks(ticks=x_ticks, labels=x_labels)
                plt.xlabel("Angle (degrees)")
                plt.ylabel("Power from capon beamforming")
            
            if singFrameView and i == singFrameNumber:
                # Save the current figure and exit
                plt.savefig(f"results\\imgs\\azimuth1d_{fname}_{i}.pdf", dpi=300)
                plt.close('all')
                gc.collect()
                sys.exit()
            
            if not singFrameView:
                plt.pause(frameTime/1000)
            
            if loadData_surr:
                line1.remove()
                line2.remove()
            else:
                plt.clf()

        # (3) Doppler Processing 
        det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=3, clutter_removal_enabled=False, window_type_2d=Window.HAMMING)
        # print(radar_cube.shape)
        
        
        # --- Show output
        if plotRangeDopp:
            det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)
            if plotMakeMovie:
                ims.append((plt.imshow(det_matrix_vis / det_matrix_vis.max()),))
            else:
                # print(det_matrix_vis.shape)
                plt.imshow(det_matrix_vis / det_matrix_vis.max())
                plt.title("Range-Doppler plot " + str(i))
                plt.pause(0.05)
                plt.clf()
                
        # (4) Object Detection
        # --- CFAR, SNR is calculated as well.
        fft2d_sum = det_matrix.astype(np.int64)
        thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=dsp.ca_,
                                                                  axis=0,
                                                                  arr=fft2d_sum.T,
                                                                  l_bound=1.5,
                                                                  guard_len=4,
                                                                  noise_len=16)

        thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=dsp.ca_,
                                                              axis=0,
                                                              arr=fft2d_sum,
                                                              l_bound=2.5,
                                                              guard_len=4,
                                                              noise_len=16)

        thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T
        det_doppler_mask = (det_matrix > thresholdDoppler)
        det_range_mask = (det_matrix > thresholdRange)

        # Get indices of detected peaks
        full_mask = (det_doppler_mask & det_range_mask)
        det_peaks_indices = np.argwhere(full_mask == True)

        # peakVals and SNR calculation
        peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
        snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]

        dtype_location = '(' + str(numTxAntennas) + ',)<f4'
        dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                                   'formats': ['<i4', '<i4', '<f4', dtype_location, '<f4']})
        detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
        detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
        detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
        detObj2DRaw['peakVal'] = peakVals.flatten()
        detObj2DRaw['SNR'] = snr.flatten()

        # Further peak pruning. This increases the point cloud density but helps avoid having too many detections around one object.
        detObj2DRaw = dsp.prune_to_peaks(detObj2DRaw, det_matrix, numDopplerBins, reserve_neighbor=True)

        # --- Peak Grouping
        detObj2D = dsp.peak_grouping_along_doppler(detObj2DRaw, det_matrix, numDopplerBins)
        SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16.0]])
        peakValThresholds2 = np.array([[4, 275], [1, 400], [500, 0]])
        detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, numRangeBins, 0.5, range_resolution)

        azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]

        x, y, z = dsp.naive_xyz(azimuthInput.T)
        xyzVecN = np.zeros((3, x.shape[0]))
        xyzVecN[0] = x * range_resolution * detObj2D['rangeIdx']
        xyzVecN[1] = y * range_resolution * detObj2D['rangeIdx']
        xyzVecN[2] = z * range_resolution * detObj2D['rangeIdx']

        Psi, Theta, Ranges, xyzVec = dsp.beamforming_naive_mixed_xyz(azimuthInput, detObj2D['rangeIdx'],
                                                                     range_resolution, method='Bartlett')

        # (5) 3D-Clustering
        # detObj2D must be fully populated and completely accurate right here
        numDetObjs = detObj2D.shape[0]
        dtf = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                        'formats': ['<f4', '<f4', '<f4', dtype_location, '<f4']})
        detObj2D_f = detObj2D.astype(dtf)
        detObj2D_f = detObj2D_f.view(np.float32).reshape(-1, 7)

        # Fully populate detObj2D_f with correct info
        for i, currRange in enumerate(Ranges):
            if i >= (detObj2D_f.shape[0]):
                # copy last row
                detObj2D_f = np.insert(detObj2D_f, i, detObj2D_f[i - 1], axis=0)
            if currRange == detObj2D_f[i][0]:
                detObj2D_f[i][3] = xyzVec[0][i]
                detObj2D_f[i][4] = xyzVec[1][i]
                detObj2D_f[i][5] = xyzVec[2][i]
            else:  # Copy then populate
                detObj2D_f = np.insert(detObj2D_f, i, detObj2D_f[i - 1], axis=0)
                detObj2D_f[i][3] = xyzVec[0][i]
                detObj2D_f[i][4] = xyzVec[1][i]
                detObj2D_f[i][5] = xyzVec[2][i]

                # radar_dbscan(epsilon, vfactor, weight, numPoints)
        #        cluster = radar_dbscan(detObj2D_f, 1.7, 3.0, 1.69 * 1.7, 3, useElevation=True)
        if len(detObj2D_f) > 0:
            cluster = clu.radar_dbscan(detObj2D_f, 0, doppler_resolution, use_elevation=True)

            cluster_np = np.array(cluster['size']).flatten()
            if cluster_np.size != 0:
                if max(cluster_np) > max_size:
                    max_size = max(cluster_np)

        # (6) Visualization
        if plotRangeDopp:
            continue
        
        if plotRangeAzimuth:
            continue
        if plotAzimuth1D:
            continue
        
        if plotRangeAzimuthTimeSeries:
            continue

        if plot2DscatterXY or plot2DscatterXZ:

            if plot2DscatterXY:
                xyzVec = xyzVec[:, (np.abs(xyzVec[2]) < 1.5)]
                xyzVecN = xyzVecN[:, (np.abs(xyzVecN[2]) < 1.5)]
                axes[0].set_ylim(bottom=0, top=10)
                axes[0].set_ylabel('Range')
                axes[0].set_xlim(left=-4, right=4)
                axes[0].set_xlabel('Azimuth')
                axes[0].grid(b=True)

                axes[1].set_ylim(bottom=0, top=10)
                axes[1].set_xlim(left=-4, right=4)
                axes[1].set_xlabel('Azimuth')
                axes[1].grid(b=True)

            elif plot2DscatterXZ:
                axes[0].set_ylim(bottom=-5, top=5)
                axes[0].set_ylabel('Elevation')
                axes[0].set_xlim(left=-4, right=4)
                axes[0].set_xlabel('Azimuth')
                axes[0].grid(b=True)

                axes[1].set_ylim(bottom=-5, top=5)
                axes[1].set_xlim(left=-4, right=4)
                axes[1].set_xlabel('Azimuth')
                axes[1].grid(b=True)

            if plotMakeMovie and plot2DscatterXY:
                ims.append((axes[0].scatter(xyzVec[0], xyzVec[1], c='r', marker='o', s=2),
                            axes[1].scatter(xyzVecN[0], xyzVecN[1], c='b', marker='o', s=2)))
            elif plotMakeMovie and plot2DscatterXZ:
                ims.append((axes[0].scatter(xyzVec[0], xyzVec[2], c='r', marker='o', s=2),
                            axes[1].scatter(xyzVecN[0], xyzVecN[2], c='b', marker='o', s=2)))
            elif plot2DscatterXY:
                axes[0].scatter(xyzVec[0], xyzVec[1], c='r', marker='o', s=3)
                axes[1].scatter(xyzVecN[0], xyzVecN[1], c='b', marker='o', s=3)
                plt.pause(0.1)
                axes[0].clear()
                axes[1].clear()
            elif plot2DscatterXZ:
                axes[0].scatter(xyzVec[0], xyzVec[2], c='r', marker='o', s=3)
                axes[1].scatter(xyzVecN[0], xyzVecN[2], c='b', marker='o', s=3)
                plt.pause(0.1)
                axes[0].clear()
                axes[1].clear()
        elif plot3Dscatter and plotMakeMovie:
            nice.set_zlim3d(bottom=-5, top=5)
            nice.set_ylim(bottom=0, top=10)
            nice.set_xlim(left=-4, right=4)
            nice.set_xlabel('X Label')
            nice.set_ylabel('Y Label')
            nice.set_zlabel('Z Label')

            ims.append((nice.scatter(xyzVec[0], xyzVec[1], xyzVec[2], c='r', marker='o', s=2),))

        # elif plot3Dscatter:
        #     if singFrameView:
        #         ellipse_visualize(fig, cluster, detObj2D_f[:, 3:6])
        #     else:
        #         ellipse_visualize(fig, cluster, detObj2D_f[:, 3:6])
        #         plt.pause(0.1)
        #         plt.clf()
        else:
            sys.exit("Unknown plot options.")

    if plotRangeAzimuthTimeSeries:
        plt.plot(ra_timeSeries)
        plt.title(f"Power time series for (range, azimuth) = ({targetRangeBin}, {targetAzimuthBin})")
        plt.xlabel("Frame number")
        plt.ylabel("Power")
        plt.savefig(f"results\\imgs\\ra_timeseries_{fname}_{i}.pdf", dpi=300)
    
    if visTrigger and plotMakeMovie:
        if not plotRangeAzimuth:
            print("Making movie...")
            movieMaker(fig, ims, makeMovieTitle, makeMovieDirectory)
        else:
            print("Making movie for Range Azimuth...")
            Writer = animation.writers['pillow']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800) 
            img = ax.imshow(range_azimuth_all_frames[:,:,0], aspect="auto", interpolation="nearest", animated=True)
            fig.colorbar(img, ax=ax, orientation="vertical")
            ani = animation.FuncAnimation(fig, update, frames=range_azimuth_all_frames.shape[-1], interval=50, blit=True, fargs=(range_azimuth_all_frames,img))
            ani.save(f"results\\gifs\\ra_{fname}.gif", writer=writer)
            print("Complete")
