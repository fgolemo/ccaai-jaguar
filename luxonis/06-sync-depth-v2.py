import depthai as dai
import numpy as np
import cv2
from datetime import timedelta

pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
color = pipeline.create(dai.node.ColorCamera)
stereo = pipeline.create(dai.node.StereoDepth)
sync = pipeline.create(dai.node.Sync)

xoutGrp = pipeline.create(dai.node.XLinkOut)

xoutGrp.setStreamName("xout")

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

color.setCamera("color")

sync.setSyncThreshold(timedelta(milliseconds=50))

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

stereo.disparity.link(sync.inputs["disparity"])
color.video.link(sync.inputs["video"])

sync.out.link(xoutGrp.input)

disparityMultiplier = 255.0 / stereo.initialConfig.getMaxDisparity()
with dai.Device(pipeline) as device:
    # queue = device.getOutputQueue("xout", 10, False)
    queue = device.getOutputQueue(name="xout", maxSize=1, blocking=False)
    while True:
        msgGrp = queue.get()
        # for name, msg in msgGrp:
        #     frame = msg.getCvFrame()
        #     if name == "disparity":
        #         frame = (frame * disparityMultiplier).astype(np.uint8)
        #         frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        #     cv2.imshow(name, frame)

        
        frameRgb = msgGrp["video"].getCvFrame()
        cv2.imshow("rgb", frameRgb)

        frameDisp = msgGrp["disparity"].getFrame()
        # Optional, extend range 0..95 -> 0..255, for a better visualisation
        frameDisp = (frameDisp * disparityMultiplier).astype(np.uint8)
        # Optional, apply false colorization
        frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_HOT)
        frameDisp = np.ascontiguousarray(frameDisp)
        cv2.imshow("depth", frameDisp)

        # frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(frameRgb, 0.4, frameDisp, 0.6, 0)
        cv2.imshow("blend", blended)
        frameRgb = None
        frameDisp = None

        key = cv2.waitKey(-1)

        if key == ord("q"):
            break


