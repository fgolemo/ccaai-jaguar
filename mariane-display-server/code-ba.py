import os
import time
import displayio
import board
# import nonblocking_serialinput as nb_serialin
import usb_cdc

files = [x for x in os.listdir("/images") if ".bmp" in x and not x.startswith(".")]
print ("files", files) 

file_idx = 0

import supervisor
supervisor.runtime.autoreload = False 

while True:
    image_file = open("/images/"+files[file_idx], "rb")
    image = displayio.OnDiskBitmap(image_file)
    image_sprite = displayio.TileGrid(image, pixel_shader=getattr(image, 'pixel_shader', displayio.ColorConverter()))

    group = displayio.Group()
    group.x = 0
    group.y = 30
    group.append(image_sprite)
    board.DISPLAY.root_group = group
    time.sleep(6)

    # scan for new files
    files2 = [x for x in os.listdir("/images") if ".bmp" in x and not x.startswith(".")]
    if len(files2) != len(files):
        print ("new files", files2) 
        file_idx = -1
        files = files2

    file_idx += 1
    if file_idx >= len(files):
        file_idx = 0

# def read_serial(serial):
    
#     available = serial.in_waiting
#     buffer = []
#     while available:
#         raw = serial.read(available)
#         # try to decode and if it's a linebreak, return the line
#         try:
#             print ("raw", raw)
#             # text = raw.decode("utf-8")
#             # if text == "\n":
            
#             buffer.append(raw)
            
#         except UnicodeDecodeError:
#             print("Error decoding serial input", raw)
#         # text = raw.decode("utf-8")
#         available = serial.in_waiting
#         if not available:
#             break
#     return buffer


# serial = usb_cdc.console
# while True:
#     test = read_serial(serial)
#     if len (test)>0:
#         print ("test", test)
    # if test is not None:
    #     print ("test")
    # if buffer.endswith("\n"):
    #     # strip line end
    #     input_line = buffer[:-1]
    #     # clear buffer
    #     buffer = ""
    #     # handle input
    #     handle_your_input(input_line)


# while True:
#     my_input.update()
#     input_string = my_input.input()
#     if input_string is not None:
#         print ("got msg", input_string)
#         # my_input.print("cool, I got "+input_string)
        # check if 

# SPDX-FileCopyrightText: 2023 Anne Barela for Adafruit Industries
#
# SPDX-License-Identifier: MIT
#
# gifio demo for the Adafruit PyPortal - single file
#
# Documentation:
#   https://docs.circuitpython.org/en/latest/shared-bindings/gifio/
# Updated 4/5/2023
#
# import time
# import gc
# import board
# import gifio
# import displayio
# import adafruit_touchscreen

# display = board.DISPLAY
# splash = displayio.Group()
# display.root_group = splash

# # Pyportal has a touchscreen, a touch stops the display
# WIDTH = board.DISPLAY.width
# HEIGHT = board.DISPLAY.height
# ts = adafruit_touchscreen.Touchscreen(board.TOUCH_XL, board.TOUCH_XR,
#                                       board.TOUCH_YD, board.TOUCH_YU,
#                                       calibration=((5200, 59000), (5800, 57000)),
#                                       size=(WIDTH, HEIGHT))
# # odg = gifio.OnDiskGif('/gif2.gif')
# odg = gifio.OnDiskGif('/goose.gif')

# start = time.monotonic()
# next_delay = odg.next_frame()  # Load the first frame
# end = time.monotonic()
# call_delay = end - start  



# # Depending on your display the next line may need Colorspace.RGB565
# #   instead of Colorspace.RGB565_SWAPPED
# group = displayio.Group()
# group.x = 50
# group.y = 0
# face = displayio.TileGrid(odg.bitmap,
#                           pixel_shader=displayio.ColorConverter
#                           (input_colorspace=displayio.Colorspace.RGB565_SWAPPED))
# group.append(face)
# board.DISPLAY.root_group = group
# # splash.append(face)
# board.DISPLAY.refresh()

# # Play the GIF file until screen is touched
# while True:
#     # time.sleep(max(0, next_delay - call_delay))
#     time.sleep(0.1)
#     next_delay = odg.next_frame()
#     if ts.touch_point is not None:
#         break
# # End while
# # Clean up memory
# odg.deinit()
# odg = None
# gc.collect()