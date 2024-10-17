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
            self.n_features=len(features)
            
            self.initial_corners = features

        else:
            raise Exception("Couldn't find the first frame")

    def runOpticalFlow(self, method="LukasKanade", radius=4):

        if self.previous_frame is not None:

            # Initialize variables and get initial values
            for i in range(0,120):
                ret, current_frame = self.video.read()
            ret, current_frame = self.video.read()
            previous_vectors = []

            if method == "LukasKanade":

                previous_vectors = self.initial_corners.reshape(self.n_features, 2)

                counter = 0
                while(ret):
                    print(counter)
                    current_frame = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)

                    # Run optical flow, and update variables
                    new_vectors = self.lukasKanade(current_frame, self.previous_frame, previous_vectors, radius)

                    new_point = np.add(previous_vectors, new_vectors)


                    # Handle values outside frame limits
                    for i in range(np.shape(new_point)[0]):
                        maxy = np.shape(current_frame)[0]-radius-1
                        maxx = np.shape(current_frame)[1]-radius-1
                        if new_point[i, 0] >= maxx or new_point[i, 1] >= maxy or new_point[i, 0] <= (radius+1) or new_point[i, 1] <= (radius+1):
                            new_point[i] = np.subtract(new_point[i], new_vectors[i])

                    #new_vectors = new_vectors.reshape(self.n_features, 2)
                    self.disparity_vectors.append(new_point)
                    previous_vectors = new_point

                    #print(new_point)

                    self.previous_frame = current_frame.copy()
                    ret, current_frame = self.video.read()
                    counter += 1

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
        for i in range(0,120):
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
                    frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
                img = cv.add(frame, mask)

                cv.imshow('frame', img)
                cv.waitKey(0)

                ret, frame = self.video.read()
                frame_number += 1
            except Exception as e:
                print("Visualization finished")
                break

    def lukasKanade(self, c_frame, p_frame, previous_points, radius):

        current_frame = c_frame.copy()
        previous_frame = p_frame.copy()

        disparity_vectors = np.empty((10, 2),dtype=np.float32)
        temp_prevx = cv.Sobel(current_frame, -1, 1, 0,ksize=-1)
        temp_prevy = cv.Sobel(current_frame, -1, 0, 1,ksize=-1)

        
        cv.waitKey(0)


        for i in range(self.n_features):

            # Step 1: Extract templates for current and previous image
            slicey = [int(previous_points[i, 0]-radius), int(previous_points[i, 0]+radius+1)]
            slicex = [int(previous_points[i, 1]-radius), int(previous_points[i, 1]+radius+1)]
            template_curr = current_frame[slicex[0]:slicex[1], slicey[0]:slicey[1]]
            template_prev = previous_frame[slicex[0]:slicex[1], slicey[0]:slicey[1]]

            #print(template_prev)
            #cv.imshow("Template prev", template_prev)
            #cv.waitKey(0)

            # Step 2: Get the T Matrix - The difference in intensities represented as a vector
            t_matrix = np.subtract(template_prev, template_curr).flatten()
            print(np.shape(np.subtract(template_prev, template_curr).flatten()))
            # Step 3: Get the gradients for the previous image
            #template_curr = current_frame[slicex[0]-1:slicex[1]+1, slicey[0]-1:slicey[1]+1]
            #template_prev = previous_frame[slicex[0]-1:slicex[1]+1, slicey[0]-1:slicey[1]+1]

            gradient_prevx = temp_prevx[slicex[0]:slicex[1], slicey[0]:slicey[1]].flatten()
            gradient_prevy = temp_prevy[slicex[0]:slicex[1], slicey[0]:slicey[1]].flatten()

            S = np.column_stack((gradient_prevx, gradient_prevy))

            # Step 4: Least squares solution
            u, v = np.linalg.lstsq(S, t_matrix)[0]

            disparity_vectors[i] = [u, v]

        print("Frame complete")

        return disparity_vectors




def main():
    of = OpticalFlow("dog.mp4")
    of.cornerDetection()
    of.runOpticalFlow("OpenCV")
    of.visualize()

main()
