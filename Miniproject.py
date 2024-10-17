import cv2 as cv
import numpy as np
import argparse

class OpticalFlow():

    def __init__(self, video_path="", n_features=15):

        self.n_features = n_features
        self.video_path = video_path
        self.video = cv.VideoCapture(video_path)
        self.initial_corners = []
        self.previous_frame = None
        self.disparity_vectors = []

    def cornerDetection(self):
        ret, first_frame = self.video.read()
        #first_frame = first_frame[171:226,151:192]
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
            self.video.read()
            ret, current_frame = self.video.read()
            #current_frame = current_frame[171:226,151:192]

            previous_vectors = []

            if method == "LukasKanade":

                previous_vectors = self.initial_corners.reshape(self.n_features, 2)

                counter = 0
                while(ret):
                    #print(counter)
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
                    previous_vectors = new_point.copy()

                    

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

        disparity_vectors = np.empty((self.n_features, 2),dtype=np.float32)

        blurred = cv.GaussianBlur(previous_frame,(3,3),0)
        

        temp_prevx = cv.Sobel(previous_frame, cv.CV_64F, 1, 0,ksize=3)
        temp_prevy = cv.Sobel(previous_frame, cv.CV_64F, 0, 1,ksize=3)

        blur_prevx = cv.Sobel(blurred, cv.CV_64F, 1, 0,ksize=3)
        blur_prevy = cv.Sobel(blurred, cv.CV_64F, 0, 1,ksize=3)

        # Threshold gradients
        # _, blur_prevx = cv.threshold(np.abs(blur_prevx), 45, 255, cv.THRESH_TOZERO)
        # _, blur_prevy = cv.threshold(np.abs(blur_prevy), 45, 255, cv.THRESH_TOZERO)

        # _, temp_prevx = cv.threshold(np.abs(temp_prevx), 10, 255, cv.THRESH_TOZERO)
        # _, temp_prevy = cv.threshold(np.abs(temp_prevy), 10, 255, cv.THRESH_TOZERO)



        gradients = np.add(temp_prevx,temp_prevy)
        gradients_blurred = np.add(blur_prevx,blur_prevy)

        # cv.imshow('grad',gradients)
        # cv.imshow('grad_blurred',gradients_blurred)
        # cv.waitKey(0)

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
            #t_matrix = np.subtract(template_prev, template_curr).flatten()
            t_matrix = template_prev.astype(np.float32)-template_curr.astype(np.float32)
            t_matrix2 = template_curr.astype(np.float32)-template_prev.astype(np.float32)
            t_matrix = t_matrix.flatten()
            t_matrix2 = t_matrix2.flatten()
            print(t_matrix)
            print('----------')
            print(t_matrix2)
            input()
            
            
            # Step 3: Get the gradients for the previous image
            #template_curr = current_frame[slicex[0]-1:slicex[1]+1, slicey[0]-1:slicey[1]+1]
            #template_prev = previous_frame[slicex[0]-1:slicex[1]+1, slicey[0]-1:slicey[1]+1]

            gradient_prevx = temp_prevx[slicex[0]:slicex[1], slicey[0]:slicey[1]].flatten()
            gradient_prevy = temp_prevy[slicex[0]:slicex[1], slicey[0]:slicey[1]].flatten()

            S = np.column_stack((gradient_prevx, gradient_prevy))
            

            # Step 4: Least squares solution
            u, v = np.linalg.lstsq(S, t_matrix,rcond=-1)[0]
            

            disparity_vectors[i] = [u, v]
            
            
        #print(disparity_vectors)
        #input()

        return disparity_vectors

    def lukasKanade2(self, c_frame, p_frame, previous_points, radius):
        current_frame = c_frame.copy()
        previous_frame = p_frame.copy()

        disparity_vectors = np.empty((self.n_features, 2), dtype=np.float32)

        
        temp_prevx = cv.Sobel(previous_frame, cv.CV_64F, 1, 0, ksize=5)
        temp_prevy = cv.Sobel(previous_frame, cv.CV_64F, 0, 1, ksize=5)

        blurred = cv.blur(previous_frame,(3,3))
        blur_prevx = cv.Sobel(blurred, -1, 1, 0,ksize=5)
        blur_prevy = cv.Sobel(blurred, -1, 0, 1,ksize=5)

        gradients_blurred = np.add(blur_prevx,blur_prevy)
        gradients = np.add(temp_prevx,temp_prevy)
        
        cv.imshow('grad',gradients)
        #cv.imshow('grad_blurred',gradients_blurred)
        cv.waitKey(0)

        for i in range(self.n_features):
            slicey = [int(previous_points[i, 0] - radius), int(previous_points[i, 0] + radius + 1)]
            slicex = [int(previous_points[i, 1] - radius), int(previous_points[i, 1] + radius + 1)]

            if (slicey[0] < 0 or slicey[1] >= current_frame.shape[1] or
                slicex[0] < 0 or slicex[1] >= current_frame.shape[0]):
                disparity_vectors[i] = [0, 0]  # Out of bounds, no movement
                continue

            template_curr = current_frame[slicex[0]:slicex[1], slicey[0]:slicey[1]]
            template_prev = previous_frame[slicex[0]:slicex[1], slicey[0]:slicey[1]]

            t_matrix = (template_prev - template_curr).flatten()
            gradient_prevx = temp_prevx[slicex[0]:slicex[1], slicey[0]:slicey[1]].flatten()
            gradient_prevy = temp_prevy[slicex[0]:slicex[1], slicey[0]:slicey[1]].flatten()

            S = np.column_stack((gradient_prevx, gradient_prevy))

            try:
                u, v = np.linalg.lstsq(S, t_matrix, rcond=None)[0]
                disparity_vectors[i] = [u, v]
            except np.linalg.LinAlgError:
                print("least squares error")
                disparity_vectors[i] = [0, 0]

        return disparity_vectors

    def lukasKanadePyramid(self, c_frame, p_frame, previous_points, radius, levels=3):
        # Create Gaussian pyramids for the previous and current frames
        previous_pyramid = [p_frame]
        current_pyramid = [c_frame]

        for level in range(1, levels):
            previous_pyramid.append(cv.pyrDown(previous_pyramid[level - 1]))
            current_pyramid.append(cv.pyrDown(current_pyramid[level - 1]))

        # Start at the coarsest level
        disparity_vectors = np.zeros((self.n_features, 2), dtype=np.float32)
        for level in reversed(range(levels)):
            scale_factor = 2 ** level
            current_frame = current_pyramid[level]
            previous_frame = previous_pyramid[level]
            
            # Scale the points to match the current pyramid level
            scaled_points = previous_points / scale_factor

            temp_prevx = cv.Sobel(previous_frame, cv.CV_64F, 1, 0, ksize=5)
            temp_prevy = cv.Sobel(previous_frame, cv.CV_64F, 0, 1, ksize=5)

            for i in range(self.n_features):
                slicey = [int(scaled_points[i, 0] - radius), int(scaled_points[i, 0] + radius + 1)]
                slicex = [int(scaled_points[i, 1] - radius), int(scaled_points[i, 1] + radius + 1)]

                if (slicey[0] < 0 or slicey[1] >= current_frame.shape[1] or
                    slicex[0] < 0 or slicex[1] >= current_frame.shape[0]):
                    continue  # Skip this point if it's out of bounds

                template_curr = current_frame[slicex[0]:slicex[1], slicey[0]:slicey[1]]
                template_prev = previous_frame[slicex[0]:slicex[1], slicey[0]:slicey[1]]

                t_matrix = (template_prev - template_curr).flatten()
                gradient_prevx = temp_prevx[slicex[0]:slicex[1], slicey[0]:slicey[1]].flatten()
                gradient_prevy = temp_prevy[slicex[0]:slicex[1], slicey[0]:slicey[1]].flatten()

                S = np.column_stack((gradient_prevx, gradient_prevy))

                try:
                    u, v = np.linalg.lstsq(S, t_matrix, rcond=None)[0]
                    # Update disparity vectors scaled to the current pyramid level
                    disparity_vectors[i] = [u, v]
                except np.linalg.LinAlgError:
                    disparity_vectors[i] = [0, 0]  # If solving fails, set flow to zero

            # Update points with new estimates and upscale for the next level
            if level > 0:
                previous_points = scaled_points + disparity_vectors
                previous_points *= 2  # Scale up to the next pyramid level

        # At the finest level, return the final disparity vectors
        return disparity_vectors



def main():
    of = OpticalFlow("slow_left.mp4")
    of.cornerDetection()
    of.runOpticalFlow("LukasKanade")
    of.visualize()

main()
