import blobconverter
model_path = blobconverter.from_zoo(
        name="human-pose-estimation-0007",
        zoo_type="intel",
        shaves=6
    )

print (model_path)
# Note that the models in the DepthAI Model Zoo are exported such that they expect images with values in the [0-255] range, 
# BGR (Blue Green Red) color order, and CHW (Channel Height Width) channel layout

# Image, name: image, shape: 1, 3, 448, 448 in the B, C, H, W format, where:

# human-pose-estimation-0007

from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('yolov7tiny_coco_416x416', color)
    nn.config_nn(resize_mode='crop')
    oak.visualize([nn, nn.out.passthrough], fps=True)
    oak.start(blocking=True)