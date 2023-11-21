from roboflow import Roboflow
rf = Roboflow(api_key="iASLXL1zT7I5Xe4TdpEV")
project = rf.workspace().project("spring-detection-9x7be")
model = project.version(1).model

# infer on a local image
print(model.predict(r"E:\imagenes paintline\resources\full\llena1.63279390335083.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("your_image.jpg", confidence=80, overlap=80).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
"""
from roboflowoak import RoboflowOak
import cv2
import time
import numpy as np

if __name__ == '__main__':
    # instantiating an object (rf) with the RoboflowOak module
    rf = RoboflowOak(model="spring-detection-9x7be", confidence=0.6, overlap=0.8,
    version="1", api_key="iASLXL1zT7I5Xe4TdpEV", rgb=True,
    depth=True, device=None, blocking=True)
    # Running our model and displaying the video output with detections
    while True:
        t0 = time.time()
        # The rf.detect() function runs the model inference
        result, frame, raw_frame, depth = rf.detect()
        predictions = result["predictions"]
        #{
        #    predictions:
        #    [ {
        #        x: (middle),
        #        y:(middle),
        #        width:
        #        height:
        #        depth: ###->
        #        confidence:
        #        class:
        #        mask: {
        #    ]
        #}
        #frame - frame after preprocs, with predictions
        #raw_frame - original frame from your OAK
        #depth - depth map for raw_frame, center-rectified to the center camera

        # timing: for benchmarking purposes
        t = time.time()-t0
        print("INFERENCE TIME IN MS ", 1/t)
        print("PREDICTIONS ", [p.json() for p in predictions])

        # setting parameters for depth calculation
        max_depth = np.amax(depth)
        cv2.imshow("depth", depth/max_depth)
        # displaying the video feed as successive frames
        cv2.imshow("frame", frame)

        # how to close the OAK inference window / stop inference: CTRL+q or CTRL+c
        if cv2.waitKey(1) == ord('q'):
            break
"""