## hand_landmarks.py ##
#
# various functions to get hand and arms landmarks from a video

# libraries
import cv2
import numpy as np

import mediapipe as mp
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# utility functions for inside the loop

def process_frame(frame, model):
    # change the frame to correct color scheme and apply id/tracking model to it    
    
    # convert frame to rgb
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # detect landmarks from image
    results = model.process(rgb_frame)
    return(results)

def annotate_frame(frame, results):
    # annotate a frame with the found landmarks
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing_styles.get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
    
def landmarks_to_array(results):
    # transform the results of mp holistic model into an array containing only landmarks of interest
    out_array = np.concatenate(
            (np.array([(lm.x, lm.y, lm.z, lm.visibility) for lm in results.pose_landmarks.landmark[7:17]]
                if results.pose_landmarks
                    else np.zeros(40)).flatten(),
            np.array([(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark]
                if results.left_hand_landmarks
                 else np.zeros(63)).flatten(),
            np.array([(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark]
                if results.right_hand_landmarks
                 else np.zeros(63)).flatten())
    )
    return(out_array)


def get_landmarks(file_in = 0, file_out = None, display = True):
    # run the mp holistic model onto a video file or using your webcam; optionnaly write out the landmarks
    # file_in: the file from which to read the video, or device number for livetracking
    # file_out: the file in which to write the landmarks. If none, landmarks are not saved
    # display: whether the video should be displayed (True) or not (False)

    # setting the tracking model
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.8) as hands:

        if display:
            # setting up window size
            cv2.namedWindow("Annotated hands", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Annotated hands", 1000, 1000)

        if file_out:
            # setting up the np array to save landmarks
            all_landmarks = None
        
        # reading video file
        cap = cv2.VideoCapture(file_in)
        while cap.isOpened():
            ret, frame = cap.read()
        
            # detect landmarks from image
            results = process_frame(frame, hands)
        
            # annotate and display frame
            if display:
                    annotate_frame(frame, results)
                    cv2.imshow("Annotated hands", cv2.flip(frame, 1))
         
            # save append the new landmarks to all_landmarks array
            if file_out:
                landmarks = landmarks_to_array(results)
                try:
                    all_landmarks = np.vstack([all_landmarks, landmarks])
                except:
                    all_landmarks = landmarks
   

            # wait 40 ms or break if 'q' is pressed
            if cv2.waitKey(40) & 0xFF == ord('q'):
                break
            

    # release the capture object
    cap.release()
     
    # Close frames
    cv2.destroyAllWindows()

    # write out the landmarks
    if file_out:
        np.save(file_out, all_landmarks)




