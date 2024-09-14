#!/usr/bin/env python3
"""
Example usage of the ODrive python library to monitor and control ODrive devices
"""

from __future__ import print_function

import numpy as np
import odrive
import odrive.enums
from odrive.utils import dump_errors
import time
import math

# Find a connected ODrive (this will block until you connect one)
print("finding an odrive...")
my_drive = odrive.find_any()

# breakpoint()
# Calibrate motor and wait for it to finish
print("starting calibration...")
my_drive.axis0.requested_state = odrive.enums.AXIS_STATE_FULL_CALIBRATION_SEQUENCE
while my_drive.axis0.current_state != odrive.enums.AXIS_STATE_IDLE:
    time.sleep(0.1)

my_drive.axis0.requested_state = odrive.enums.AXIS_STATE_CLOSED_LOOP_CONTROL

# To read a value, simply read the property
print("Bus voltage is " + str(my_drive.vbus_voltage) + "V")

dump_errors(my_drive, True)

# Or to change a value, just assign to the property
my_drive.axis0.controller.input_pos = -0.8 # this should probably be a lot lower



print("Position setpoint is " + str(my_drive.axis0.controller.pos_setpoint))
# quit()
# And this is how function calls are done: #1.19, -1.5
for i in [1,2,3,4]:
    print('voltage on GPIO{} is {} Volt'.format(i, my_drive.get_adc_voltage(i)))

speed = 3 # turns per second, 0.1 to like 10



# A sine wave to test
t0 = time.monotonic()
while True:
    setpoint = math.sin((time.monotonic() - t0)*speed) * 1.2
    # setpoint = np.clip(setpoint, -)
    
    print("goto ", setpoint)
    my_drive.axis0.controller.input_pos = setpoint
    my_drive.axis0.watchdog_feed()
    time.sleep(0.001)
    # dump_errors(my_drive, True)
    # quit()

# Some more things you can try:

# Write to a read-only property:
my_drive.vbus_voltage = 11.0  # fails with `AttributeError: can't set attribute`

# Assign an incompatible value:
my_drive.motor0.pos_setpoint = "I like trains"  # fails with `ValueError: could not convert string to float`