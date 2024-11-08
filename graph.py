import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

df = pd.read_csv('2021_12_08_17_31_26_933bcb2ee1c266e0.csv')

latitude = df['latitude']
longitude = df['longitude']
timestamp_offset = df['timestamp_offset']

plt.figure(figsize=(10, 6))
scatter = plt.scatter(longitude, latitude, c=timestamp_offset, cmap='viridis', s=50, alpha=0.8)

plt.colorbar(scatter, label='Timestamp Offset')
plt.ticklabel_format(style='plain', axis='both', useOffset=False)

plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.show()



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(min(df['acc_x']), max(df['acc_x']))
ax.set_ylim(min(df['acc_y']), max(df['acc_y']))
ax.set_zlim(min(df['acc_z']), max(df['acc_z']))

ax.set_xlabel('Acceleration X')
ax.set_ylabel('Acceleration Y')
ax.set_zlabel('Acceleration Z')

point, = ax.plot([], [], [], 'o', color='blue')

def update_acc(num):
    point.set_data(df['acc_x'][num], df['acc_y'][num])
    point.set_3d_properties(df['acc_z'][num])
    return point,

ani = FuncAnimation(fig, update_acc, frames=len(df['timestamp_offset']), interval=50, blit=True, repeat=False)

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(min(df['gyro_x']), max(df['gyro_x']))
ax.set_ylim(min(df['gyro_y']), max(df['gyro_y']))
ax.set_zlim(min(df['gyro_z']), max(df['gyro_z']))

ax.set_xlabel('Gyroscope X')
ax.set_ylabel('Gyroscope Y')
ax.set_zlabel('Gyroscope Z')

point, = ax.plot([], [], [], 'o', color='blue')

def update_acc(num):
    point.set_data(df['gyro_x'][num], df['gyro_y'][num])
    point.set_3d_properties(df['gyro_z'][num])
    return point,

ani = FuncAnimation(fig, update_acc, frames=len(df['timestamp_offset']), interval=50, blit=True, repeat=False)

plt.show()




plt.figure(figsize=(10, 6))
plt.plot(df['timestamp_offset'], df['compass'], label='compass', color='b')
plt.xlabel('Timestamp Offset')
plt.ylabel('Compass (degree)')
plt.legend()
plt.grid(True)
plt.show()