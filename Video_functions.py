import cv2 as cv
import os

def save_video_frames(video_path, dir_name):
    """
    Saves all frames in a video as images in a numbered order
    into a destination directory.

    Args:
    video_path (str): Path to the video file.
    dir_name (str): Destination directory for the extracted frames.
    """

    # Create the destination directory if it does not exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Open the video file
    video = cv.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not video.isOpened():
        print("Error: Could not open video file.")
        return

    frame_number = 0
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    # Iterate through the video frames
    while True:
        ret, frame = video.read()

        # Break the loop if we reached the end of the video
        if not ret:
            break
        # Calculate the percentage of video played
        percentage_played = (frame_number / total_frames) * 100
        display_frame = frame.copy()
        # Overlay the text onto the frame
        font = cv.FONT_HERSHEY_SIMPLEX
        text = f'{percentage_played:.2f}%'
        position = (10, 30)
        font_scale = 1
        font_color = (0, 255, 0)  # Green
        line_type = 2
        cv.putText(display_frame, text, position, font, font_scale, font_color, line_type)

        # Display the current video frame
        cv.imshow('video', display_frame)

        # Save the current frame as an image
        frame_filename = os.path.join(dir_name, f'frame{frame_number:04d}.jpg')
        cv.imwrite(frame_filename, frame)
        frame_number += 1

        # Break the loop if the user presses the 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video resources and close the display window
    video.release()
    cv.destroyAllWindows()
# def choose_train_images(video, dir_name):