import cv2 as cv
from matplotlib.pylab import grid
from more_itertools import first
import numpy as np
import argparse
import math

import pprint

parser = argparse.ArgumentParser(description="Lucas-Kanade Optical Flow")
parser.add_argument('--method', type=str, 
        help='Method used to solve the optical flow')
parser.add_argument('--video', type=str, 
        help='Video used to test the method')
parser.add_argument('--skip', action='store_true', 
        help='Flag to skip the first N frames of the video (default: False)')
parser.add_argument('--save', action='store_true', 
        help='Save the result in an output with the method name (default: False)')
parser.add_argument('--radius', type=int, default=3, 
        help='Radius for Lucas-Kanade window size (default: 3)')
parser.add_argument('--debug', action='store_true', help='Enables debug')
parser.add_argument('--manual', action='store_true', help='Enables manual controls for visualization')

args = parser.parse_args()


class OpticalFlow:
    def __init__(self, video_path="", n_features=16, debug=False):
        """Initialize the OpticalFlow class with video path and number of features."""

        self.n_features = n_features
        self.video_path = video_path
        self.video = cv.VideoCapture(video_path)
        
        if not self.video.isOpened():
            raise Exception(f"Error opening video file: {video_path}")
        
        self.initial_corners = []
        self.prev_frame = None
        self.disparity_vectors = []

        self.templates = []

        self.running = True
        self.debug = debug
    
    def reset(self, video_path="", n_features=16, debug=False):
        self.n_features = n_features
        self.video_path = video_path
        self.video = cv.VideoCapture(video_path)
        
        if not self.video.isOpened():
            raise Exception(f"Error opening video file: {video_path}")
        
        self.initial_corners = []
        self.prev_frame = None
        self.disparity_vectors = []

        self.running = True
        self.debug = debug

    def debug_image(self, image, title):

        cv.imshow(title, image)
        key = cv.waitKey(0) 
        
        if key == ord("q") or key == ord("Q"):
            self.running = False
            cv.destroyAllWindows()
            exit()

        if key == ord("d") or key == ord("D"):
            self.debug = False


    def cornerDetection(self):
        """Detect corners in the first frame of the video."""

        ret, first_frame = self.video.read()

        if not ret:
            raise Exception("Couldn't find the first frame")
        
        self.prev_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

        # Input parameters for corner detection function
        input_parameters = dict(
            maxCorners=self.n_features,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7)
        
        # Detect corners in image
        self.initial_corners = cv.goodFeaturesToTrack(self.prev_frame, **input_parameters)

        if self.initial_corners is None:
            raise Exception("Couldn't detect any corners.")
        
        print(f"Found {self.n_features} corners")

        self.n_features = len(self.initial_corners)

    def display_templates(self, spacing, templates):
        n_images = len(templates)
    
        # Calculate grid size (square-like layout)
        grid_size = math.ceil(math.sqrt(n_images))  # Create grid rows and cols

        # Image dimensions (assumes all images are the same size)
        
        img_height = len(templates[0])
        img_width = len(templates[0])

        # Create a blank canvas for the grid
        # Calculate the total canvas size including the spacing (for grid lines)
        grid_image = np.zeros(((img_height + spacing) * grid_size - spacing, 
                           (img_width + spacing) * grid_size - spacing), dtype=np.uint8) + 255  # White background for grid lines

            # Loop through images and place them in the grid
        for idx, img in enumerate(templates):
            row = idx // grid_size
            col = idx % grid_size
            y = row * (img_height + spacing)
            x = col * (img_width + spacing)
            grid_image[y:y+img_height, x:x+img_width] = img
            
        return grid_image

    def lucasKanade(self, crnt_frame, prev_frame, prev_points, radius=3):
        disparity_vectors = np.empty_like(prev_points)

        # Get image gradients
        gradient_prev_x = cv.Sobel(prev_frame, cv.CV_64F, 1, 0, ksize=-1)
        gradient_prev_y = cv.Sobel(prev_frame, cv.CV_64F, 0, 1, ksize=-1)

        # Just for debug purposes
        if self.debug:
            self.templates = []
            self.debug_image(gradient_prev_x + gradient_prev_y, "Gradient")
        
        

        # Set a gradient threshold to filter out weak gradients (noise)
        gradient_threshold = 1e-3
        
        for i in range(self.n_features):
            pt_x, pt_y = prev_points[i]

            # Extract region around the point for current and previous frames
            x_st, x_end = int(pt_x - radius), int(pt_x + radius + 1)
            y_st, y_end = int(pt_y - radius), int(pt_y + radius + 1)

            # Ensure the region is within bounds of the frame
            if (x_st < 0 or y_st < 0 or x_end >= crnt_frame.shape[1] 
                or y_end >= crnt_frame.shape[0]):
                disparity_vectors[i] = prev_points[i]
                continue

            # Extract image patches for the current and previous frames
            template_curr = crnt_frame[y_st:y_end, x_st:x_end]
            template_prev = prev_frame[y_st:y_end, x_st:x_end]
            
            if self.debug:
            #     self.debug_image(template_curr, "Template for point: "  + str(i))
                len(template_curr)
                new_size = (int(len(template_curr) * 5), int(len(template_curr) * 5))
        
                # Resize the image
                resized_img = cv.resize(template_curr, new_size, interpolation=cv.INTER_LINEAR)
                
                self.templates.append(resized_img)
            

            # Extract gradient patches
            template_gradient_x = gradient_prev_x[y_st:y_end, x_st:x_end]
            template_gradient_y = gradient_prev_y[y_st:y_end, x_st:x_end]

            # if self.debug:
            #     self.debug_image(template_gradient_x + template_gradient_y, "Template gradient for point: " + str(i))



            # Apply gradient threshold to filter out regions with less gradient
            gradient_magnitude = np.sqrt(template_gradient_x ** 2 + template_gradient_y ** 2)
            if np.mean(gradient_magnitude) < gradient_threshold:
                disparity_vectors[i] = prev_points[i]  
                # No reliable gradient, skip this point
                continue

            # Flatten and create the matrix
            S_Matrix = np.column_stack((template_gradient_x.flatten(), template_gradient_y.flatten()))
            

            # Calculate the temporal difference (template difference)
            t_matrix = (template_prev - template_curr).flatten()

            # Solve for displacement (optical flow vector)
            uv, _, _, _ = np.linalg.lstsq(S_Matrix, t_matrix, rcond=None)

            # Update the position of the point
            disparity_vectors[i] = prev_points[i] + uv

        if self.debug:
            templates = self.display_templates(10, self.templates)
            self.debug_image(templates, "All templates")

        return disparity_vectors

    def runOpticalFlow(self, method="LukasKanade", radius=3):
        """Run optical flow algorithm (Lucas-Kanade or OpenCV method)."""
        
        if self.prev_frame is None:
            raise Exception("Run corner detection first")
        
        # Initialize variables
        ret, crnt_frame = self.video.read()
        
        if not ret:
            raise Exception("No frames to read from video")

        prev_vectors = self.initial_corners.reshape(self.n_features, 2)

        if method == "LucasKanade":
            while ret:
                crnt_frame_gray = cv.cvtColor(crnt_frame, cv.COLOR_BGR2GRAY)
                crnt_vectors = self.lucasKanade(crnt_frame_gray, self.prev_frame, prev_vectors, radius=radius)

                self.disparity_vectors.append(crnt_vectors)
                prev_vectors = crnt_vectors

                self.prev_frame = crnt_frame_gray.copy()
                ret, crnt_frame = self.video.read()

        elif method == "OpenCV":
            lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

            while ret:
                crnt_frame_gray = cv.cvtColor(crnt_frame, cv.COLOR_BGR2GRAY)
                crnt_vectors, st, err = cv.calcOpticalFlowPyrLK(self.prev_frame, crnt_frame_gray, prev_vectors, None, **lk_params)
                prev_vectors = crnt_vectors

                self.disparity_vectors.append(crnt_vectors)
                self.prev_frame = crnt_frame_gray.copy()
                ret, crnt_frame = self.video.read()

        else:
            raise Exception(f"{method} method is not supported")

        self.disparity_vectors = np.stack(self.disparity_vectors, axis=0)
        return self.disparity_vectors

    def save_result_video(self, color, output_video_path=""):
        """Visualize and save the video with the optical flow vectors."""
    
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
        color = color

        frame_number = 0

        while True:
            if frame_number >= len(self.disparity_vectors):
                print("End of video")
                break
            if frame_number < 0:
                frame_number = 0

            self.video.set(cv.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = self.video.read()
            if not ret:
                print("End of the video")
                break

            # Draw the flow vectors for this frame
            for i, corner in enumerate(self.initial_corners):
                a, b = self.disparity_vectors[frame_number][i]
                c, d = self.disparity_vectors[frame_number - 1][i] if frame_number > 0 else (a, b)

                mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color, 2)
                frame = cv.circle(frame, (int(a), int(b)), 5, color, -1)

            img = cv.add(frame, mask)
            
            out.write(img)

            frame_number += 1

    def visualize(self, disp_a, disp_b, manual_control=False):
        """Visualize and compare two different disparity vectors"""
        # In our case will be OpenCV vs ROB7 Lucas Kanade

        self.video = cv.VideoCapture(self.video_path)

        frame_number = 0
        ret, frame = self.video.read()

        if not ret or frame is None:
            print("Cannot read the first frame for visualization")
            exit()
        
        mask_a = np.zeros_like(frame)
        mask_b = np.zeros_like(frame)
      
        color_a = (0, 0, 255)  # Red for Lucas-Kanade
        color_b = (255, 0, 0)  # Blue for OpenCV

        while True:
            if frame_number >= len(disp_a):
                print("End of video")
                break

            # Read the corresponding frame
            self.video.set(cv.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.video.read()
            if not ret:
                print("Cannot read frame")
                break

            # Draw the flow vectors for both methods (if method == "both")
          
            for i in range(np.shape(disp_b)[1]):

                # ROB7 vectors 
                a, b = disp_a[frame_number][i]
                c, d = disp_a[frame_number - 1][i] if frame_number > 0 else (a, b) 
                mask_a = cv.line(mask_a, (int(a), int(b)), (int(c), int(d)), color_a, 2)
                frame = cv.circle(frame, (int(a), int(b)), 5, color_a, -1)

                # # OpenCV vectors 
                a, b = disp_b[frame_number][i]
                c, d = disp_b[frame_number - 1][i] if frame_number > 0 else (a, b)
                mask_b = cv.line(mask_b, (int(a), int(b)), (int(c), int(d)), color_b, 2) 
                frame = cv.circle(frame, (int(a), int(b)), 5, color_b, -1)

            img = cv.add(frame, mask_a)
            img = cv.add(img, mask_b)
            
            # Show the frame with optical flow
            cv.imshow('Optical Flow Visualization', img)

            # Wait for a key press
            key = cv.waitKey(1) & 0xFF  # Wait for 30 ms for real-time effect
            # Quit visualization when 'q' is pressed
            if key == ord('q'):
                break

            if manual_control:   
                # Move forward with right arrow key (→)
                if key == ord('d'):
                    frame_number += 1

                # Move backward with left arrow key (←)
                elif key == ord('a'):
                    frame_number -= 1  
            else:
                frame_number += 1

def main():
    """Main function to execute the OpticalFlow process."""
    method = args.method
    video = args.video
    radius = args.radius
    save = args.save
    debug = args.debug
    manual = args.manual


    print("Running OpenCV - Optical Flow")
    of = OpticalFlow(video, debug=debug)
    of.cornerDetection()
    method = "OpenCV"
    ocv_disp_vector = of.runOpticalFlow(method, radius=radius)
    if save:
        of.save_result_video(color=(0, 0, 255), output_video_path="outputs/" + method + ".mp4")
    # 
    print("* ----------------------------------------- *")

    print("Running ROB7 - Optical Flow")    
    of.reset(video, debug=debug)
    of.cornerDetection()
    method = "LucasKanade"
    rob7_disp_vector = of.runOpticalFlow("LucasKanade", radius=radius)
    if save:
        of.save_result_video(color=(255, 0, 0), output_video_path="outputs/" + method + ".mp4")
    print("* ----------------------------------------- *")

    print("Combining both results to compare")
    of.visualize(ocv_disp_vector, rob7_disp_vector, manual_control=manual)

   


    print("* ----------------------------------------- *")   

if __name__ == "__main__":
    main()
