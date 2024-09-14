import serial

# windows
# ser = serial.Serial('COM1', 115200) or ser = serial.Serial(port='COM4')


with serial.Serial('/dev/tty.usbmodem1101', 115200, timeout=1) as ser:
    # x = ser.read()          # read one byte
    # s = ser.read(10)        # read up to ten bytes (timeout)
    ser.write(b'hello')     # write a string
    line = ser.readline()   # read a '\n' terminated line
