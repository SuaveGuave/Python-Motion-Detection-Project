from tkinter import *
from tkinter import filedialog
from tkVideoPlayer import TkinterVideo
import tkinter.messagebox
import cv2
import numpy as np
from typing import cast
import os

# global variable setup
my_label = None
current_player = None
motion_player = None
user_video_window = None
video_path = ""
fps = 15


def detect_motion(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        tkinter.messagebox.showinfo("Error", "Error opening video file for motion detection.")  # incase it cant open
        return None, None

    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, dsize=(600, 400))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_blur = cv2.GaussianBlur(frame_rgb, (5, 5), 0)  # applies Gaussian blur
            frames.append((frame, frame_blur))  # stores original and blurred frames
        else:
            break
    cap.release()

    diff_frames = []
    bounding_box_frames = []
    left = None

    for i, (original_frame, frame_blur) in enumerate(frames):
        if i == 0:
            left = np.float32(frame_blur)  # (left is previous frame)
        else:
            diff_frame = (np.float32(frame_blur) - left) ** 2
            diff_allchannel = np.sum(diff_frame, axis=2)

            if np.mean(diff_allchannel) > 10:
                # threshold the diff frame
                _, threshold_done = cv2.threshold(diff_allchannel, 25, 255, cv2.THRESH_BINARY)
                threshold_done = threshold_done.astype(np.uint8)
                # essentially, if the mean difference exceeds a threshold, binary thresholding is applied,
                # to identify the areas of significant motion

                # find contours
                contours, _ = cv2.findContours(threshold_done, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # draw bounding boxes on the original frame
                for contour in contours:
                    if cv2.contourArea(contour) > 500:  # filter small regions
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(original_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        # I honestly don't get how all this works, the documentation is not clear
                        # ,but it works, and it's 2am, so I am not looking into this further

                bounding_box_frames.append(original_frame)
                diff_frames.append(diff_allchannel)
            left = np.float32(frame_blur)
            # saves processed frame, then updates left for the next iteration

    return diff_frames, bounding_box_frames


def save_motion_video(frames, fps=15):
    # whole function basically saves the motion detected frames as a video
    motion_output_path = "video files for testing/motion_output_segment.mp4"

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # codec used for mp4 so that its optimised better
    target_resolution = (frames[0].shape[1], frames[0].shape[0])

    out = cv2.VideoWriter(
        motion_output_path,
        fourcc,
        fps,
        target_resolution,
        isColor=True
    )

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # converts back to BGR for writing

    out.release()
    return motion_output_path


def save_event_log(file_path, motion_detected):
    # self-explanatory, if motion detected, write it in log, if no motion, write that in log
    log_path = "event_log.txt"
    if motion_detected:
        log_entry = f"File: {file_path}\nMotion Detected\n\n"
    else:
        log_entry = f"File: {file_path}\nNo Motion Detected\n\n"

    with open(log_path, "a") as log_file:
        log_file.write(log_entry)

    tkinter.messagebox.showinfo("Event Log Updated", f"Motion event logged to {os.path.abspath(log_path)}")


def detected_motion_video():
    global video_path, motion_player
    if not video_path:
        tkinter.messagebox.showinfo("Error", "No video selected.")
        return

    diff_frames, bounding_box_frames = detect_motion(video_path)
    motion_detected = diff_frames is not None and len(diff_frames) > 0
    save_event_log(video_path, motion_detected)
    # checks for detected frames

    if not motion_detected:
        tkinter.messagebox.showinfo("Info", "No significant motion detected.")
        return

    motion_output_path = save_motion_video(bounding_box_frames, fps=fps)

    if motion_player:
        motion_player.destroy()

    motion_player = TkinterVideo(user_video_window, scaled=True)
    motion_player.load(motion_output_path)
    motion_player.pack(expand=True, fill="both")
    motion_player.play()


def play_selected_video():
    global my_label, current_player, user_video_window, video_path

    if my_label:
        my_label.destroy()
    if current_player:
        current_player.destroy()

    my_label = Label(user_video_window)
    my_label.pack()

    video_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv")]
    )
    if video_path:
        current_player = TkinterVideo(user_video_window, scaled=True)
        current_player.load(video_path)
        current_player.pack(expand=True, fill="both")
        current_player.play()
    else:
        tkinter.messagebox.showinfo("Error", "Please select an appropriate file")


def pause_video():
    global current_player
    if current_player is not None:
        player = cast(TkinterVideo, current_player)
        # cast used because it thinks .pause and .play could be None,
        # but they literally can't due to where this is accessed,
        # and the yellow errors were really visually annoying me
        player.pause()
    else:
        tkinter.messagebox.showinfo("Error", "No video loaded, please load video")


def play_video():
    global current_player
    if current_player is not None:
        player = cast(TkinterVideo, current_player)
        player.play()
    else:
        tkinter.messagebox.showinfo("Error", "No video loaded, please load video")


master = Tk()
master.title("Motion Detection Program")
master.geometry("600x600")

open_video_button = Button(master, text='Open Video', command=play_selected_video)
open_video_button.pack(pady=10)

play_button = Button(master, text='Play', command=play_video)
play_button.pack(pady=10)

pause_button = Button(master, text='Pause', command=pause_video)
pause_button.pack(pady=10)

detect_motion_button = Button(master, text='Detect Motion', command=detected_motion_video)
detect_motion_button.pack(pady=10)

master.mainloop()
