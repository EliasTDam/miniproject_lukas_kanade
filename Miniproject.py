import cv2 as cv
import numpy as np
import argparse

class OpticalFlow():

    def __init__(self, video_path="", n_features=10):

        self.n_features = n_features
        self.video_path = video_path
        self.video = cv.VideoCapture(video_path)
        self.initial_corners = []
        self.previous_frame = None
        self.disparity_vectors = []

    def cornerDetection(self):
        ret, first_frame = self.video.read()
        self.previous_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
        if ret:

            # Convert image to grayscale
            grayImage = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

            # Input parameters for corner detection function
            input_parameters = dict(
                maxCorners=self.n_features,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7
            )

            # Detect corners in image
            features = cv.goodFeaturesToTrack(grayImage, **input_parameters)

            print("Found corners")
            self.initial_corners = features

        else:
            raise Exception("Couldn't find the first frame")

    def runOpticalFlow(self, method="LukasKanade", radius=5):

        if self.previous_frame is not None:

            # Initialize variables and get initial values
            ret, current_frame = self.video.read()
            previous_vectors = []

            if method == "LukasKanade":

                previous_vectors = self.initial_corners.reshape(self.n_features, 2)

                while(ret):
                    current_frame = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)

                    # Run optical flow, and update variables
                    new_vectors = self.lukasKanade(current_frame, self.previous_frame, previous_vectors, radius)

                    new_vectors = new_vectors.reshape(self.n_features, 2)
                    self.disparity_vectors.append(new_vectors)
                    previous_vectors = new_vectors

                    self.previous_frame = current_frame.copy()
                    ret, current_frame = self.video.read()

            elif method == "OpenCV":

                previous_vectors = self.initial_corners

                while(ret):
                    current_frame = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)

                    # Run optical flow, and update variables
                    lk_params = dict(winSize=(15, 15),
                                     maxLevel=2,
                                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

                    new_vectors, st, err = cv.calcOpticalFlowPyrLK(
                        self.previous_frame,
                        current_frame,
                        previous_vectors,
                        None,
                        **lk_params
                    )

                    previous_vectors = new_vectors
                    new_vectors = new_vectors.reshape(self.n_features, 2)
                    self.disparity_vectors.append(new_vectors)

                    self.previous_frame = current_frame.copy()
                    ret, current_frame = self.video.read()

            else:
                raise Exception(f"{method} method is not supported")

            self.disparity_vectors = np.stack(self.disparity_vectors, axis=0)

        else:
            raise Exception("Couldn't find the first frame - Run corner detection first")

    def visualize(self):
        self.video = cv.VideoCapture(self.video_path)

        self.video.read()
        ret, frame = self.video.read()
        frame_number = 1

        mask = np.zeros_like(frame)
        color = np.random.randint(0, 255, (100, 3))

        while(ret):
            try:
                # draw the tracks
                for i in range(np.shape(self.disparity_vectors)[1]):
                    a = self.disparity_vectors[frame_number, i, 0]
                    b = self.disparity_vectors[frame_number, i, 1]
                    c, d = self.disparity_vectors[frame_number-1, i, 0], self.disparity_vectors[frame_number-1, i, 1]
                    mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                    frame = cv.circle(frame, (int(self.initial_corners[i, 0, 0]), int(self.initial_corners[i, 0, 1])), 5, color[i].tolist(), -1)
                img = cv.add(frame, mask)

                cv.imshow('frame', img)
                cv.waitKey(30)

                ret, frame = self.video.read()
                frame_number += 1
            except Exception as e:
                print("Visualization finished")
                break

    def lukasKanade(self, current_frame, previous_frame, previous_points, radius):

        disparity_vectors = np.empty((10, 2))

        for i in range(self.n_features):

            # Step 1: Extract templates for current and previous image
            slicex = [int(previous_points[i, 0]-radius), int(previous_points[i, 0]+radius+1)]
            slicey = [int(previous_points[i, 1]-radius), int(previous_points[i, 1]+radius+1)]
            template_curr = current_frame[slicex[0]:slicex[1], slicey[0]:slicey[1]]
            template_prev = previous_frame[slicex[0]:slicex[1], slicey[0]:slicey[1]]

            # Step 2: Get the T Matrix - The difference in intensities represented as a vector
            t_matrix = np.subtract(template_prev, template_curr).flatten()

            # Step 2: Get the gradients for the previous image
            template_curr = current_frame[slicex[0]-1:slicex[1]+1, slicey[0]-1:slicey[1]+1]
            template_prev = previous_frame[slicex[0]-1:slicex[1]+1, slicey[0]-1:slicey[1]+1]

            cv.imshow("Template_curr", template_curr)
            cv.waitKey(0)

            gradient_currx = cv.Sobel(template_curr, -1, 1, 0)
            gradient_curry = cv.Sobel(template_curr, -1, 0, 1)
            gradient_prevx = cv.Sobel(template_prev, -1, 1, 0)
            gradient_prevy = cv.Sobel(template_prev, -1, 0, 1)

            print(f"Shape template: {template_curr.shape}")
            print(f"Shape gradient: {gradient_currx.shape}")
            print("--------------------------------------")







def main():
    of = OpticalFlow("slow_traffic_small.mp4")
    of.cornerDetection()
    of.runOpticalFlow("LukasKanade")
    of.visualize()

main()
