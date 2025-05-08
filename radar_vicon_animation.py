import os
import re
from datetime import datetime
import numpy as np
import scatter_animation as vicon
import matplotlib.pyplot as plt
import static_vs_dynamic_animation as radar
import pandas as pd
import matplotlib.animation as animation

#rcParams formatting
plt.rcParams.update({
    'font.size': 24,
    #"text.usetex": True,
    "font.family": "Helvetica"})
# plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

radar_filename = 'bot_2025-03-07-18:02:15.586184_day2_four_static_two_dynamic_2___only_sensor.bin'
vicon_filename = "day2_four_static_two_dynamic_2_03072025180214.305770_Trajectories_100.csv"
radar_end_frame = 150

def extract_radar_time(filename):
    pattern = r'bot_(\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}.\d{6})_'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None

def extract_vicon_time(filename):
    key = "_Trajectories_100"
    pos = filename.find(key)
    if pos == -1 or pos < 21:
        return None
    return filename[pos-21:pos]

def convert_to_datetime(radar_str, vicon_str):
    radar_dt = datetime.strptime(radar_str, "%Y-%m-%d-%H:%M:%S.%f")
    vicon_dt = datetime.strptime(vicon_str, "%m%d%Y%H%M%S.%f")
    return radar_dt, vicon_dt

radar_time, vicon_time = convert_to_datetime(extract_radar_time(radar_filename), extract_vicon_time(vicon_filename))

vicon_filename = f"vicon/{vicon_filename}"
df = vicon.process_vicon(vicon_filename)
df["timestamp"] = [vicon_time + pd.Timedelta(milliseconds=10 * i) for i in range(len(df))]
vicon_static_pcd = []
vicon_dynamic_pcd = []
vicon_df = df.copy()
vicon_df= vicon_df.drop('timestamp', axis=1)
for frame_num in range(1, int(len(vicon_df)/10)):
    _, vicon_static, vicon_dynamic = vicon.finding_bot(frame_num, vicon_df)
    vicon_static_pcd.append(vicon_static)
    vicon_dynamic_pcd.append(vicon_dynamic) 
    # vicon_pcd.append(vicon.finding_bot(frame_num, vicon_df))
vicon_pcd_df = pd.DataFrame({'vicon_static': vicon_static_pcd, 'vicon_dynamic': vicon_dynamic_pcd})
vicon_pcd_df['timestamp'] = [vicon_time + pd.Timedelta(milliseconds=100 * i) for i in range(len(vicon_pcd_df))]

radar_static_pcd, radar_dynamic_pcd = radar.export_points_framewise(radar_filename, radar_end_frame+1)
radar_df = pd.DataFrame({'radar_static': radar_static_pcd, 'radar_dynamic': radar_dynamic_pcd})
radar_df["timestamp"] = [radar_time + pd.Timedelta(milliseconds=200 * i) for i in range(len(radar_df))]

radar_vicon_df = pd.merge_asof(radar_df, vicon_pcd_df, on='timestamp', direction='nearest')

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))

# Create the initial scatter plot
static_scatter = ax.scatter([], [], c='b', label='Radar Static', s=50)
dynamic_scatter = ax.scatter([], [], c='g', label='Radar Dynamic', s=50)
vicon_static_scatter = ax.scatter([], [], c='r', label='Vicon Static', s=50)
vicon_dynamic_scatter = ax.scatter([], [], c='tab:orange', label='Vicon Dynamic', s=50)

ax.set_xlim(-3, 3)  # Set appropriate limits for X axis
ax.set_ylim(-0.25, 5)  # Set appropriate limits for Y axis
ax.set_xlabel('X (m)') #, fontsize = 24)
ax.set_ylabel('Y (m)') #, fontsize = 24)
ax.legend(loc=(0, 1.05), ncol=2) #,bbox_to_anchor=(-0.03, 1.5), ) #, bbox_to_anchor=(0.01, 1), ) #, fontsize=18) (0,1)
plt.tight_layout()

def update(frame):
    static_points = radar_vicon_df['radar_static'][frame][:,:2]
    dynamic_points = radar_vicon_df['radar_dynamic'][frame][:,:2]
    vicon_static_points = radar_vicon_df['vicon_static'][frame] / 1000
    vicon_dynamic_points = radar_vicon_df['vicon_dynamic'][frame] / 1000
     
    # ax.set_title(f'Frame No: {frame+1}')
     
    if len(static_points) == 0 or len(dynamic_points) == 0 or len(vicon_static_points)==0 or len(vicon_dynamic_points)==0:
        return static_scatter, dynamic_scatter, vicon_static_scatter, vicon_dynamic_scatter
    
    # Update scatter plots
    static_scatter.set_offsets(static_points)
    dynamic_scatter.set_offsets(dynamic_points)
    vicon_static_scatter.set_offsets(vicon_static_points)
    vicon_dynamic_scatter.set_offsets(vicon_dynamic_points)
    
    return static_scatter, dynamic_scatter, vicon_static_scatter, vicon_dynamic_scatter

# Create the animation
# ani = animation.FuncAnimation(
#     fig, update, frames=len(radar_vicon_df), interval=200, blit=True
# )

# # Save the animation
# ani.save('radar_vicon_animation.gif', writer='imagemagick', fps=5)
# plt.show()

output_dir = "radar_vicon_frames"
# for frame in range(len(radar_vicon_df)):
    # if frame+1 == 24:
frame = 23
update(frame)  # Call he existing update function
plt.savefig(os.path.join(output_dir, f"frame_{(frame+1):04d}.png"))  # Save each frame