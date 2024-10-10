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

    def runOpticalFlow(self, method="LukasKanade"):

        if self.previous_frame is not None:

            # Initialize variables and get initial values
            ret, current_frame = self.video.read()
            previous_vectors = []
            print(ret)

            if method == "LukasKanade":

                while(ret):
                    current_frame = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)

                    # Run optical flow, and update variables
                    new_vectors = lukasKanade(current_frame, self.previous_frame, previous_vectors)

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

            # draw the tracks
            #print(f"Shape of disp: {np.shape(self.disparity_vectors)[1]}")
            for i in range(np.shape(self.disparity_vectors)[1]):
                #print(f"Type of disparity: {type(self.disparity_vectors[0])}")
                a = self.disparity_vectors[frame_number, i, 0]
                b = self.disparity_vectors[frame_number, i, 1]
                c, d = self.disparity_vectors[frame_number-1, i, 0], self.disparity_vectors[frame_number-1, i, 1]
                #if i == 0:
                    #print(f"A: {a}, B: {b}, C: {c}, D: {d}")
                mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv.circle(frame, (int(self.initial_corners[i, 0, 0]), int(self.initial_corners[i, 0, 1])), 5, color[i].tolist(), -1)
            img = cv.add(frame, mask)
            #print("---------------------")

            cv.imshow('frame', img)
            cv.waitKey(30)

            ret, frame = self.video.read()
            frame_number += 1

    def lukasKanade(self):
        # Convert to grayscale
        pass

def main():
    of = OpticalFlow("slow_traffic_small.mp4")
    of.cornerDetection()
    of.runOpticalFlow("OpenCV")
    of.visualize()
    print(np.shape(of.disparity_vectors))

main()
