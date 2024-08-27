import argparse
import threading
from pathlib import Path

from depthai_sdk.managers import PipelineManager, NNetManager, BlobManager, PreviewManager
from depthai_sdk import Previews, getDeviceInfo, downloadYTVideo
import blobconverter
from datetime import timedelta

import cv2
import depthai as dai
import numpy as np

import cv2
import numpy as np

keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank',
                    'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28],
          [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]]
colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]


def getKeypoints(probMap, threshold=0.2):
    mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
    mapMask = np.uint8(mapSmooth > threshold)
    keypoints = []
    contours = None
    try:
        # OpenCV4.x
        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        # OpenCV3.x
        _, contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


def getValidPairs(outputs, w, h, detected_keypoints):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.2
    conf_th = 0.4
    for k in range(len(mapIdx)):

        pafA = outputs[0, mapIdx[k][0], :, :]
        pafB = outputs[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (w, h))
        pafB = cv2.resize(pafB, (w, h))
        candA = detected_keypoints[POSE_PAIRS[k][0]]

        candB = detected_keypoints[POSE_PAIRS[k][1]]

        nA = len(candA)
        nB = len(candB)

        if (nA != 0 and nB != 0):
            valid_pair = np.zeros((0, 3))
            for i in range(nA):
                max_j = -1
                maxScore = -1
                found = 0
                for j in range(nB):
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores) / len(paf_scores)

                    if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            valid_pairs.append(valid_pair)
        else:
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


def getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list):
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:, 0]
            partBs = valid_pairs[k][:, 1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][
                        2]

                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints

def cam2keypoints(data):
    heatmaps = np.array(data.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57))
    pafs = np.array(data.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57))
    heatmaps = heatmaps.astype('float32')
    pafs = pafs.astype('float32')
    outputs = np.concatenate((heatmaps, pafs), axis=1)

    new_keypoints = []
    new_keypoints_list = np.zeros((0, 3))
    keypoint_id = 0

    for row in range(18):
        probMap = outputs[0, row, :, :]
        probMap = cv2.resize(probMap, (w,h))  # (456, 256)
        keypoints = getKeypoints(probMap, 0.3)
        new_keypoints_list = np.vstack([new_keypoints_list, *keypoints])
        keypoints_with_id = []

        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoint_id += 1

        new_keypoints.append(keypoints_with_id)

    valid_pairs, invalid_pairs = getValidPairs(outputs, w=w, h=h, detected_keypoints=new_keypoints)
    newPersonwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, new_keypoints_list)
    # breakpoint()
    # print("len keypoints", len(newPersonwiseKeypoints)) # number of skellies

    for idx in range (len(newPersonwiseKeypoints)):
        good_keypoints = len([x for x in newPersonwiseKeypoints[idx] if x != -1])
        print (f"found person {idx} with {good_keypoints} keypoints") # min 7 keypoints
    # good_keypoints = len([x for x in valid_pairs if len(x) > 0])
    # print (good_keypoints)
    return new_keypoints, new_keypoints_list, newPersonwiseKeypoints

def show(frame, keypoints_list, detected_keypoints, personwiseKeypoints):
    
    if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
        scale_factor = frame.shape[0] / h # 1 
        offset_w = int(frame.shape[1] - w * scale_factor) // 2 # 0

        def scale(point):
            return int(point[0] * scale_factor) + offset_w, int(point[1] * scale_factor)

        for i in range(18):
            for j in range(len(detected_keypoints[i])):
                cv2.circle(frame, scale(detected_keypoints[i][j][0:2]), 5, colors[i], -1, cv2.LINE_AA)
        for i in range(17):
            for n in range(len(personwiseKeypoints)):
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(frame, scale((B[0], A[0])), scale((B[1], A[1])), colors[i], 3, cv2.LINE_AA)

threshold = 0.3
nPoints = 18
w = 456
h = 256
full_w = 1920
full_h = 1080
detected_keypoints = []
shaves = 6
device_info = getDeviceInfo()

colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]

running = True
pose = None
keypoints_list = None
detected_keypoints = None
personwiseKeypoints = None

blob_path = blobconverter.from_zoo(name="human-pose-estimation-0001", shaves=shaves)

pipeline = dai.Pipeline()
color = pipeline.create(dai.node.ColorCamera)
color.setPreviewSize(w, h)
color.setCamera("color")
color.setInterleaved(False)


monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")

monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)


nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(blob_path)

sync = pipeline.create(dai.node.Sync)
sync.setSyncThreshold(timedelta(milliseconds=50))
stereo.disparity.link(sync.inputs["disparity"])

xOut = pipeline.create(dai.node.XLinkOut)
xOut.setStreamName("xout")
# color.video.link(xOut.input)

color.preview.link(nn.input)
color.preview.link(sync.inputs["rgb"])
color.video.link(sync.inputs["video"])
nn.out.link(sync.inputs["nn"])
sync.out.link(xOut.input)

disparityMultiplier = 255.0 / stereo.initialConfig.getMaxDisparity()

with dai.Device(pipeline) as device:
    # queue = device.getOutputQueue("xout", 10, False)
    queue = device.getOutputQueue(name="xout", maxSize=1, blocking=False)
    while True:
        msgGrp = queue.get()
        
        frameRaw = msgGrp["video"].getCvFrame()
        frameRgb = msgGrp["rgb"].getCvFrame()

        frameDisp = msgGrp["disparity"].getFrame()
        # Optional, extend range 0..95 -> 0..255, for a better visualisation
        frameDisp = (frameDisp * disparityMultiplier).astype(np.uint8)
        # Optional, apply false colorization
        frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_HOT)
        frameDisp = np.ascontiguousarray(frameDisp)
        cv2.imshow("depth", frameDisp)

        # frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(frameRaw, 0.4, frameDisp, 0.6, 0)
        detected_keypoints, keypoints_list, personwiseKeypoints = cam2keypoints(msgGrp["nn"])
        show(blended, keypoints_list, detected_keypoints, personwiseKeypoints)
        cv2.imshow("blend", blended)
        
        cv2.imshow("rgb", frameRgb)
        # print (frameRaw.shape) # 1920x1080
        # quit()
        show(frameRaw, keypoints_list, detected_keypoints, personwiseKeypoints)
        cv2.imshow("video", frameRaw)

        frameRgb = None

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

