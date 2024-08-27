import numpy as np
import matplotlib.pyplot as plt




plt.ion()

fig2, ax2 = plt.subplots()

array = np.zeros(shape=(24, 32), dtype=np.uint8)
axim2 = ax2.imshow(array, vmin=20, vmax=40)

fig2.subplots_adjust(right=0.85)
cbar_ax = fig2.add_axes([0.88, 0.15, 0.04, 0.7])
fig2.colorbar(axim2, cax=cbar_ax)

del array

import serial
ser = serial.Serial("/dev/tty.usbmodem1101", 115200)

print (1)
while True:
    line = ser.readline()
    if not line.strip():  # evaluates to true when an "empty" line is received
        print ("Empty line")
        pass
    else:
        # print (f"Line: {line}")
        line = line.decode("utf-8")[:-4]
        # print (line)
        parts = [float(x) for x in line.split(", ")]
        # print (len(parts))
        if len(parts) != 768:
            print (f"Skipping line: {line}")
            continue
        mat = np.array(parts).reshape(24, 32)
        axim2.set_data(mat)
        fig2.canvas.flush_events()
