import cv2 as cv
import numpy as np
import argparse
import pprint

parser = argparse.ArgumentParser(description="Test")
parser.add_argument('method', type=str, help='Method used to solve the optical flow')
parser.add_argument('--skip', action='store_true', help='Flag to skip the first N frames of the video (default: False)')
args = parser.parse_args()

DEBUG = False

class OpticalFlow:
    def __init__(self, video_path="", n_features=100):
        self.n_features = n_features
        self.video_path = video_path
        self.video = cv.VideoCapture(video_path)
        self.initial_corners = []
        self.previous_frame = None
        self.disparity_vectors = []
    
    def cornerDetection(self):
        ret, first_frame = self.video.read()
        if not ret:
            raise Exception("Couldn't find the first frame")
        
        self.previous_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
        
        # Input parameters for corner detection function
        input_parameters = dict(
            maxCorners=self.n_features,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Detect corners in image
        self.initial_corners = cv.goodFeaturesToTrack(self.previous_frame, **input_parameters)
        self.n_features = len(self.initial_corners)
        
        if self.initial_corners is None:
            raise Exception("Couldn't detect any corners.")
        print("Found corners")

    def runOpticalFlow(self, method="LukasKanade", radius=3):
        if self.previous_frame is None:
            raise Exception("Run corner detection first")
        
        # Initialize variables
        ret, current_frame = self.video.read()
        if not ret:
            raise Exception("No frames to read from video")

        previous_vectors = self.initial_corners.reshape(self.n_features, 2)

        if method == "LukasKanade":
            while ret:
                current_frame_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
                new_vectors = self.lukasKanade(current_frame_gray, self.previous_frame, previous_vectors)

                self.disparity_vectors.append(new_vectors)
                previous_vectors = new_vectors

                self.previous_frame = current_frame_gray.copy()
                ret, current_frame = self.video.read()

        elif method == "OpenCV":
            lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

            while ret:
                current_frame_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
                new_vectors, st, err = cv.calcOpticalFlowPyrLK(self.previous_frame, current_frame_gray, previous_vectors, None, **lk_params)
                previous_vectors = new_vectors

                self.disparity_vectors.append(new_vectors)
                self.previous_frame = current_frame_gray.copy()
                ret, current_frame = self.video.read()

        else:
            raise Exception(f"{method} method is not supported")

        self.disparity_vectors = np.stack(self.disparity_vectors, axis=0)

    def visualize(self, output_video_path="output_with_optical_flow.mp4"):
        self.video = cv.VideoCapture(self.video_path)
        
        # Get video properties
        width = int(self.video.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = self.video.get(cv.CAP_PROP_FPS)

        # Initialize VideoWriter to save the output video
        fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        out = cv.VideoWriter(output_video_path, fourcc, fps, (width, height))

        ret, frame = self.video.read()
        mask = np.zeros_like(frame)
        color = np.random.randint(0, 255, (self.n_features, 3))

        frame_number = 0

        while True:
            if frame_number >= len(self.disparity_vectors):
                print("End of video")
                break
            if frame_number < 0:
                frame_number = 0

            # Reset mask and read the corresponding frame
            self.video.set(cv.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.video.read()
            if not ret:
                print("Cannot read frame")
                break

            # Draw the flow vectors for this frame
            for i, corner in enumerate(self.initial_corners):
                a, b = self.disparity_vectors[frame_number][i]
                c, d = self.disparity_vectors[frame_number - 1][i] if frame_number > 0 else (a, b)

                mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

            img = cv.add(frame, mask)
            
            # Show the frame with optical flow
            cv.imshow('Optical Flow Visualization', img)

            # Write the frame to the output video file
            out.write(img)

            # Wait for a key press
            key = cv.waitKey(0) & 0xFF

            # Quit visualization when 'q' is pressed
            if key == ord('q'):
                break

            # Move forward with right arrow key (→)
            elif key == ord('d'):  # Right arrow key
                frame_number += 1

            # Move backward with left arrow key (←)
            elif key == ord('a'):  # Left arrow key
                frame_number -= 1

        cv.destroyAllWindows()


    def lukasKanade(self, current_frame, previous_frame, previous_points, radius=3):
        disparity_vectors = np.empty_like(previous_points)

        for i in range(self.n_features):
            point_x, point_y = previous_points[i]

            # Extract region around the point for current and previous frames
            x_start, x_end = int(point_x - radius), int(point_x + radius + 1)
            y_start, y_end = int(point_y - radius), int(point_y + radius + 1)

            template_curr = current_frame[y_start:y_end, x_start:x_end]
            template_prev = previous_frame[y_start:y_end, x_start:x_end]

            if template_curr.shape != template_prev.shape or template_curr.size == 0:
                disparity_vectors[i] = previous_points[i]
                continue

            # Get image gradients
            gradient_prev_x = cv.Sobel(template_prev, cv.CV_64F, 1, 0)
            gradient_prev_y = cv.Sobel(template_prev, cv.CV_64F, 0, 1)

            # Flatten and create the matrix
            S_Matrix = np.column_stack((gradient_prev_x.flatten(), gradient_prev_y.flatten()))

            # Calculate the temporal difference (template difference)
            t_matrix = (template_prev - template_curr).flatten()

            # Solve for displacement (optical flow vector)
            uv, _, _, _ = np.linalg.lstsq(S_Matrix, t_matrix, rcond=None)

            # Update the position of the point
            disparity_vectors[i] = previous_points[i] + uv

        if DEBUG:    
            pprint.pprint(disparity_vectors)
            input("-------------------")
        return disparity_vectors


def main():

    method = args.method

    of = OpticalFlow("slow_traffic_small.mp4")
    of.cornerDetection()
    of.runOpticalFlow(method)
    of.visualize(output_video_path=method + ".mp4")


if __name__ == "__main__":
    main()
