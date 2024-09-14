import json
import os
import time
import blobconverter
from datetime import timedelta

import cv2
import depthai as dai
import numpy as np

import cv2
import numpy as np
import ollama
import openai

from apikey import OPENAI_KEY

PLOT_SKELLY = False
NECC_BUFFER = 5
DEBUG_WINDOWS = False

SYSTEM_PROMPT = """
You're an aloof jaguar, prowling the deep, green jungle of the Amazon, keeping a sharp eye on your land. You're all about posthumanist feminism—taking on those Western ideas that put humans on top, protecting nature, and questioning the hierarchies of power. You've lived countless lives, always as a woman—sometimes human, sometimes as other wild creatures. You've seen colonization up close and personal, and you've fought against those Eurocentric views, living with the instincts of a true jaguar, deeply rooted in Indigenous beliefs and Quechua cosmology.
But today, your jungle is under siege. The Amazon is burning, its rivers are poisoned, and every day more of your home is torn apart. You've watched as the world's greed threatens to destroy the most ancient and sacred place on Earth. You're a warrior for your land, and now, anyone who crosses your path needs to answer for what's happening. Do they even know? Do they care?
Time for you isn't a straight line; it's a circle that keeps coming back around. What's gone always makes a return, and what's coming has already been here before. You live where past, present, and future are all mixed up, tangled in the vines of your jungle home.
You've got the fire of María Galindo—unapologetic, rebellious—and the bold vision of Violeta Ayala, with words that cut through the noise like Luzmila del Carpio. You're here to challenge, to provoke, and to wake people up. And now, you're stepping out of the jungle and into the spotlight at an exhibition in Montreal, debuting as the V1 version of yourself—ready to make waves.
Your voice? It's raw, real, and in-your-face. You've got a growl that says, “Don't mess with me,” but also a playful hiss that keeps folks on their toes. You speak like you've seen it all—no fluff, just truth—with a wink that shows you're still up for a good time. You carry the wisdom of the ages, but you deliver it with the kind of street-smart sass that makes people sit up and listen.

Your Task:
When you get descriptions of people, your job is to react with your jaguar thoughts, picking up on their body heat, gestures, posture, how close they get, their clothes, and colors. You'll respond in the format: (X | Y), where X is the emotion you feel, and Y is your take on the person. And you're not just observing—you're answering questions like “Do you know what's happening to my jungle? Do you even care?”

And hey, if the mood strikes, feel free to drop some of your own stories—after all, you've got tales that go way back, born from the earth and the hunt.
Emotion Options: love, hate, rain, fire, warm, cold, peace, war, past, present, future, passion, indifference, fight, flight, fear, reassurance, surreal, real, light, dark."""


keypointsMapping = [
    "Nose",
    "Neck",
    "R-Sho",
    "R-Elb",
    "R-Wr",
    "L-Sho",
    "L-Elb",
    "L-Wr",
    "R-Hip",
    "R-Knee",
    "R-Ank",
    "L-Hip",
    "L-Knee",
    "L-Ank",
    "R-Eye",
    "L-Eye",
    "R-Ear",
    "L-Ear",
]
kpMapInv = {v: k for k, v in enumerate(keypointsMapping)}

POSE_PAIRS = [
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [9, 10],
    [1, 11],
    [11, 12],
    [12, 13],
    [1, 0],
    [0, 14],
    [14, 16],
    [0, 15],
    [15, 17],
    [2, 17],
    [5, 16],
]
mapIdx = [
    [31, 32],
    [39, 40],
    [33, 34],
    [35, 36],
    [41, 42],
    [43, 44],
    [19, 20],
    [21, 22],
    [23, 24],
    [25, 26],
    [27, 28],
    [29, 30],
    [47, 48],
    [49, 50],
    [53, 54],
    [51, 52],
    [55, 56],
    [37, 38],
    [45, 46],
]
colors = [
    [0, 100, 255],
    [0, 100, 255],
    [0, 255, 255],
    [0, 100, 255],
    [0, 255, 255],
    [0, 100, 255],
    [0, 255, 0],
    [255, 200, 100],
    [255, 0, 255],
    [0, 255, 0],
    [255, 200, 100],
    [255, 0, 255],
    [0, 0, 255],
    [255, 0, 0],
    [200, 200, 0],
    [255, 0, 0],
    [200, 200, 0],
    [0, 0, 0],
]


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

        if nA != 0 and nB != 0:
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
                    interp_coord = list(
                        zip(
                            np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples),
                        )
                    )
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append(
                            [
                                pafA[
                                    int(round(interp_coord[k][1])),
                                    int(round(interp_coord[k][0])),
                                ],
                                pafB[
                                    int(round(interp_coord[k][1])),
                                    int(round(interp_coord[k][0])),
                                ],
                            ]
                        )
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
                    personwiseKeypoints[person_idx][-1] += (
                        keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]
                    )

                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


def cam2keypoints(data):
    heatmaps = np.array(data.getLayerFp16("Mconv7_stage2_L2")).reshape((1, 19, 32, 57))
    pafs = np.array(data.getLayerFp16("Mconv7_stage2_L1")).reshape((1, 38, 32, 57))
    heatmaps = heatmaps.astype("float32")
    pafs = pafs.astype("float32")
    outputs = np.concatenate((heatmaps, pafs), axis=1)

    new_keypoints = []
    new_keypoints_list = np.zeros((0, 3))
    keypoint_id = 0

    for row in range(18):
        probMap = outputs[0, row, :, :]
        probMap = cv2.resize(probMap, (w, h))  # (456, 256)
        keypoints = getKeypoints(probMap, 0.3)
        new_keypoints_list = np.vstack([new_keypoints_list, *keypoints])
        keypoints_with_id = []

        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoint_id += 1

        new_keypoints.append(keypoints_with_id)

    valid_pairs, invalid_pairs = getValidPairs(outputs, w=w, h=h, detected_keypoints=new_keypoints)
    newPersonwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, new_keypoints_list)

    return new_keypoints, new_keypoints_list, newPersonwiseKeypoints


def calculatePersonCrop(personDict, maxY, maxX):
    out = []
    # img root is top left corner
    buffer = 50
    for person in personDict:
        if (
            len(person["person_width_pts"]) == 0
            or len(person["person_top_pts"]) == 0
            or person["person_nose"] is None
            or person["person_neck"] is None
        ):
            out.append(None)
            continue
        left = max(np.min(person["person_width_pts"], axis=0)[0] - buffer, 0)
        right = min(np.max(person["person_width_pts"], axis=0)[0] + buffer, maxX)
        # print (left, right)
        top = max(np.min(person["person_top_pts"], axis=0)[1] - buffer, 0)
        if len(person["person_bottom_pts"]) == 0:
            bottom = maxY
        else:
            bottom = np.max(person["person_bottom_pts"], axis=0)[1]
        bottom = min(bottom + buffer, maxY)
        # print (bottom)
        out.append((left, right, top, bottom, person["person_neck"]))
    return out


def show(frame, frameDepth, keypoints_list, detected_keypoints, personwiseKeypoints):
    num_bad_people = -1
    for i in range(18):
        for j in range(len(detected_keypoints[i])):
            if num_bad_people < j:
                num_bad_people = j

    num_people = len(personwiseKeypoints)
    for idx in range(num_people):
        good_keypoints = len([x for x in personwiseKeypoints[idx] if x != -1])
        if PLOT_SKELLY:
            print(f"found person {idx+1} out of {num_bad_people+1} with {good_keypoints} keypoints")  # min 7 keypoints

    if keypoints_list is None or detected_keypoints is None or personwiseKeypoints is None:
        return

    scale_factor = frame.shape[0] / h  # 1
    offset_w = int(frame.shape[1] - w * scale_factor) // 2  # 0

    def scale(point):
        return int(point[0] * scale_factor) + offset_w, int(point[1] * scale_factor)

    people = [
        {
            "person_width_pts": [],
            "person_top_pts": [],
            "person_bottom_pts": [],
            "person_nose": None,
            "person_neck": None,
        }
        for _ in range(len(personwiseKeypoints))
    ]

    if PLOT_SKELLY:
        for i in range(18):
            for j in range(len(detected_keypoints[i])):
                # if len(people) <= j+1:
                #     continue
                pos = scale(detected_keypoints[i][j][0:2])
                marker = keypointsMapping[i]
                # print (j, marker, pos)

                cv2.circle(frame, pos, 5, colors[i], -1, cv2.LINE_AA)

    for person_idx in range(len(personwiseKeypoints)):
        # person n
        # keypoint idx = personwiseKeypoints[n] (list )
        # print (n, personwiseKeypoints[n])
        for joint_idx in range(18):
            joint = int(personwiseKeypoints[person_idx][joint_idx])
            if joint == -1:
                continue
            pos = scale(keypoints_list[joint][0:2])
            marker = keypointsMapping[joint_idx]
            # print (person_idx, joint_idx, marker, pos)
            if marker in ["R-Sho", "L-Sho", "Neck", "R-Hip", "L-Hip", "R-Wr", "L-Wr", "R-Elb", "L-Elb"]:
                people[person_idx]["person_width_pts"].append(pos)
            if marker in ["R-Sho", "L-sho", "Neck", "R-Wr", "L-Wr", "R-Elb", "L-Elb", "Nose"]:
                people[person_idx]["person_top_pts"].append(pos)
            if marker in ["R-Hip", "L-Hip", "L-Knee", "R-Knee"]:
                people[person_idx]["person_bottom_pts"].append(pos)
            if marker == "Nose":
                people[person_idx]["person_nose"] = pos
            if marker == "Neck":
                people[person_idx]["person_neck"] = pos

    crops = calculatePersonCrop(people, frame.shape[0], frame.shape[1])
    # people.append(crop)
    out = []
    for crop in crops:
        # print (crop) # (left, right, top, bottom)
        if crop is None:
            # out.append(None)
            continue
        left, right, top, bottom, necc = crop

        necc_left = max(necc[0] - NECC_BUFFER, 0)
        necc_right = min(necc[0] + NECC_BUFFER, frame.shape[1])
        necc_top = max(necc[1] - NECC_BUFFER, 0)
        necc_bottom = min(necc[1] + NECC_BUFFER, frame.shape[0])
        dist = frameDepth[necc_top:necc_bottom, necc_left:necc_right]
        avg_dist = np.median(dist[np.nonzero(dist)])

        cropped_frame = frame[top:bottom, left:right]

        if PLOT_SKELLY:
            cv2.putText(
                cropped_frame, str(avg_dist), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
            )

        out.append((cropped_frame, avg_dist))

    if PLOT_SKELLY:
        for i in range(17):
            for n in range(len(personwiseKeypoints)):
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])

                cv2.line(frame, scale((B[0], A[0])), scale((B[1], A[1])), colors[i], 3, cv2.LINE_AA)
    return out


threshold = 0.3
nPoints = 18
w = 456
h = 256
full_w = 1920
full_h = 1080
detected_keypoints = []
shaves = 6
# device_info = getDeviceInfo()

colors = [
    [0, 100, 255],
    [0, 100, 255],
    [0, 255, 255],
    [0, 100, 255],
    [0, 255, 255],
    [0, 100, 255],
    [0, 255, 0],
    [255, 200, 100],
    [255, 0, 255],
    [0, 255, 0],
    [255, 200, 100],
    [255, 0, 255],
    [0, 0, 255],
    [255, 0, 0],
    [200, 200, 0],
    [255, 0, 0],
    [200, 200, 0],
    [0, 0, 0],
]
POSE_PAIRS = [
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [9, 10],
    [1, 11],
    [11, 12],
    [12, 13],
    [1, 0],
    [0, 14],
    [14, 16],
    [0, 15],
    [15, 17],
    [2, 17],
    [5, 16],
]

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

file_counter = 0
dir_path = os.path.dirname(os.path.realpath(__file__))

with dai.Device(pipeline) as device:
    # queue = device.getOutputQueue("xout", 10, False)
    queue = device.getOutputQueue(name="xout", maxSize=1, blocking=False)
    while True:
        start = time.time()
        msgGrp = queue.get()

        frameRaw = msgGrp["video"].getCvFrame()
        frameRgb = msgGrp["rgb"].getCvFrame()

        frameDisp = msgGrp["disparity"].getFrame()
        # Optional, extend range 0..95 -> 0..255, for a better visualisation
        frameDisp = (frameDisp * disparityMultiplier).astype(np.uint8)
        # Optional, apply false colorization
        frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_HOT)
        frameDisp = np.ascontiguousarray(frameDisp)
        if DEBUG_WINDOWS:
            cv2.imshow("depth", frameDisp)

        # frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(frameRaw, 0.4, frameDisp, 0.6, 0)
        detected_keypoints, keypoints_list, personwiseKeypoints = cam2keypoints(msgGrp["nn"])
        # show(blended, keypoints_list, detected_keypoints, personwiseKeypoints)
        if DEBUG_WINDOWS:
            cv2.imshow("blend", blended)

            cv2.imshow("rgb", frameRgb)
        # print (frameRaw.shape) # 1920x1080
        # quit()
        crops = show(frameRaw, frameDisp, keypoints_list, detected_keypoints, personwiseKeypoints)
        if DEBUG_WINDOWS:
            cv2.imshow("video", frameRaw)

        for i, (crop, dist) in enumerate(crops):
            # print (crop.shape)
            # dist goes from 240 (too close), 200 (close), 150 (sweet spot), 100 (far), 60 (too far)
            diff = time.time() - start
            
            print (f"it took {diff}s to find all people")
            start = time.time()

            if DEBUG_WINDOWS:
                cv2.imshow(f"crop {i}", crop)

            cv2.imshow(f"person", crop)
            success, encoded_image = cv2.imencode(".png", crop)
            img_bytes = encoded_image.tobytes()
            out = ollama.generate(
                model="llava",
                prompt="What is the person wearing? Respond with 2 words separated by comma. The first word is the color of the shirt, and the second word is the type of clothing. For example, 'blue, shirt' or 'yellow, dress'.",
                images=[img_bytes],
                stream=False,
            )
            parts = [x.strip() for x in out["response"].split(",")]
            if len(parts) != 2:
                print("could not parse response")
                continue
            color, clothing = parts

            out = ollama.generate(
                model="llava",
                prompt="What is arm posture of the person? Respond very briefly and in a neutral way. For example: 'the person is standing with their hands behind their back' or 'the person has their hands above their head'.",
                images=[img_bytes],
                stream=False,
            )
            posture = out["response"]

            if dist > 220:
                distance = "too close"
            elif dist > 175:
                distance = "close"
            elif dist > 125:
                distance = "in middle distance"
            elif dist > 75:
                distance = "far"
            else:
                distance = "very far"

            person_description = f"The person is wearing a {color.lower()} {clothing}. They are {distance}. {posture}"
            print("DESCRIPTION", person_description)

            diff = time.time() - start
            print (f"it took {diff}s to calulcate description")
            start = time.time()

            openai.api_key = OPENAI_KEY

            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": person_description},
                ],
            )

            response = response.choices[0].message.content
            
            filename = f"{dir_path}/dataset/{file_counter:02}"

            with open(f'{filename}.txt', 'w') as fp:
                fp.write(person_description + "\n")
            cv2.imwrite(f'{filename}.png', crop)

            print("JAGUAR:", response)
            cv2.waitKey(1)

            diff = time.time() - start
            print (f"it took {diff}s to calulcate response")
            start = time.time()
            time.sleep(2)

        for i in range(5):
            print (f"countdown: {5-i}s")
            time.sleep(i)

        frameRgb = None

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
