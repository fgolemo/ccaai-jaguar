import io
from serial import Serial
import os

demofile = os.path.expanduser("~/Downloads/dream 1.png") 

# convert to bmp

from PIL import Image

img = Image.open(demofile)
if len(img.split()) == 4:
    # prevent IOError: cannot write mode RGBA as BMP
    r, g, b, a = img.split()
    img = Image.merge("RGB", (r, g, b))

# image is now in memory as a BMP file
# have to resize
#base_width = 320
# wpercent = (base_width / float(img.size[0]))
# hsize = int((float(img.size[1]) * float(wpercent)))
# img = img.resize((base_width, hsize), Image.Resampling.LANCZOS)

# we're assuming fixed size
img = img.resize((320, 180), Image.Resampling.LANCZOS)

img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='BMP')
img_byte_arr = img_byte_arr.getvalue()


# open serial port and send image
ser = Serial('/dev/tty.usbmodem1101', 115200, timeout=1) #or whatever
ser.write(img_byte_arr) #send file
# ser.write(b"test") #send file
# ser.write(b"\n") #send message indicating file transmission complete

# ser = Serial('/dev/tty.usbmodem1101', 115200, timeout=1) #or whatever 
# ser.write(open("some_file.txt","rb").read()) #send file
# ser.write("\n<<EOF>>\n") #send message indicating file transmission complete
