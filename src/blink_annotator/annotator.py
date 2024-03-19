#!/usr/bin/python3


import pickle
from collections import deque
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from blink_annotator.utils.ear import EAR, FaceDetectException
from blink_annotator.utils.eyeblink_entry import EyeBlinkEntry, MalformedEntryError


def create_annotations(blinks: list[bool]) -> list[EyeBlinkEntry]:
    annotations: list[EyeBlinkEntry] = []
    blink_count = 1
    idx = 0
    for blink in blinks:
        if not blink:
            blink_id = -1

            if idx > 0 and blinks[idx - 1]:
                blink_count += 1
        else:
            blink_id = blink_count
        annotations.append(EyeBlinkEntry(frame_id=idx, blink_id=blink_id))
        idx += 1

    return annotations


def write_annotations_file(file, annotations):
    with open(file, "w") as f:
        for annotation in annotations:
            f.write(f"{annotation}\n")


def save_blinks(blinks_file: Path, d: list):
    afile = open(blinks_file, "wb")
    pickle.dump(d, afile)
    afile.close()


def load_blinks(blinks_file: Path):
    afile = open(blinks_file, "rb")

    d = pickle.load(afile)
    afile.close()
    return d


def print_debug(blinks, frame_number):
    print(f"Frames {max(frame_number - 10, 0)} - {frame_number + 9}")
    print("---------------")
    for idx in range(max(frame_number - 10, 0), frame_number + 10):
        print(f"{idx}: {blinks[idx]}")
    pass


def time_output(mtime):
    secs, milli = divmod(int(mtime), 1000)
    min, secs = divmod(secs, 60)

    return f"{min:d}:{secs:02d}:{milli:02d}"


def osd(cap, frame, blinks, insert_mode):
    frame_number = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    frame_time = cap.get(cv.CAP_PROP_POS_MSEC)

    (height, width, _) = frame.shape

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)

    font_face = cv.FONT_HERSHEY_PLAIN
    font_scale = 1.5
    font_thickness = 2

    cv.rectangle(frame, (0, height - 80), (width, height), black, -1)

    text = "Insert Mode" if insert_mode else "View Mode"
    color = green if insert_mode else white
    size, _ = cv.getTextSize(
        text,
        fontFace=font_face,
        fontScale=font_scale,
        thickness=font_thickness,
    )
    text_width, _ = size
    cv.putText(
        frame,
        text,
        (width - (text_width + 5), height - 60),
        fontFace=font_face,
        fontScale=font_scale,
        thickness=font_thickness,
        color=color,
    )

    cv.putText(
        frame,
        f"Frame: {frame_number}",
        (5, height - 60),
        fontFace=font_face,
        fontScale=font_scale,
        thickness=font_thickness,
        color=white,
    )
    cv.putText(
        frame,
        f"Time: {time_output(frame_time)}",
        (5, height - 38),
        fontFace=font_face,
        fontScale=font_scale,
        thickness=font_thickness,
        color=white,
    )

    text_color = green if blinks[frame_number] else red
    cv.putText(
        frame,
        f"Blink: {blinks[frame_number]}",
        (5, height - 16),
        fontFace=font_face,
        fontScale=font_scale,
        thickness=font_thickness,
        color=text_color,
    )
    return frame


def draw_eye_marks(frame, facemarks):
    red = (0, 0, 255)
    for idx in range(36, 48):
        cv.circle(frame, (facemarks.part(idx).x, facemarks.part(idx).y), 2, red, -1)

    return frame


def plot(frame):
    # ear = EAR()
    x = 0
    # for i in range(100):
    x = x + 0.04
    y = np.sin(x)
    plt.scatter(x, y)
    plt.title("Real Time plot")
    plt.xlabel("x")
    plt.ylabel("sinx")
    # plt.pause(0.05)

    # plt.show()


def annotate(video_file: str, max_height: int = -1, start_frame: int = 0):
    ear = EAR()

    cap = cv.VideoCapture(video_file)
    if not cap.isOpened:
        print("--(!)Error opening video file")
        exit(0)

    width: float = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height: float = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    max_height = int(height) if max_height == -1 else max_height
    # fps: float = cap.get(cv.CAP_PROP_FPS)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    scale = 1.0
    if height > max_height:
        scale = max_height / height

    frame_number = start_frame
    video_path = Path(video_file)

    blinks_file = video_path.with_suffix(".blink")
    annotation_file = video_path.with_suffix(".tag")

    if blinks_file.exists():
        blinks = load_blinks(blinks_file)
    else:
        blinks = [None] * frame_count
        blinks[frame_number] = False

    insert_mode = False
    debug_mode = False

    fig = plt.figure()

    xs = deque(maxlen=13)
    ys = deque(maxlen=13)
    # plt.title("Dynamic Plot of sinx", fontsize=25)
    #
    # plt.xlabel("X", fontsize=18)
    # plt.ylabel("sinX", fontsize=18)

    # window = cv.namedWindow("Annotation Window")

    # line1, = plt.plot((0, 13), (0, 1), 'ko-')

    while frame_number < frame_count:
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_number - 1)
        _, frame = cap.read()

        if frame is None:
            print("--(!) Skipping frame!")
            break

        frame = cv.resize(frame, (int(width * scale), int(height * scale)))

        if frame_number > 0 and blinks[frame_number] is None:
            blinks[frame_number] = blinks[frame_number - 1]

        frame = osd(cap, frame, blinks, insert_mode)

        if debug_mode:
            try:
                eye_marks = ear.eye_marks(frame)
                frame = draw_eye_marks(frame, eye_marks)
            except FaceDetectException:
                print(f"No face detected at frame: {frame_number}")
                xs.append(frame_number)
                ys.append(None)

                frame_number += 1
                continue

        cv.imshow("Annotation window", frame)

        key = cv.waitKey(0)

        if key >= 0:
            c = chr(key)
            if c == "d":
                debug_mode = not debug_mode
            elif c == "s":
                annotations = create_annotations(blinks)
                write_annotations_file(annotation_file, annotations)
            elif c == "q":
                exit(0)

            if insert_mode:
                if c == "\x1b":  # Escape
                    save_blinks(blinks_file, blinks)  # Save before leaving insert mode
                    insert_mode = False
                elif c in [" ", "n"]:
                    blinks[frame_number + 1] = blinks[frame_number]
                    frame_number += 1
                elif c == "p":
                    frame_number = frame_number - 1 if frame_number >= 0 else 0
                elif c in ["\r", "b"]:
                    blinks[frame_number] = not blinks[frame_number]

                save_blinks(blinks_file, blinks)

            else:
                if c == "i":
                    insert_mode = True
                elif c in [" ", "n"]:
                    frame_number += 1
                elif c == "p":
                    frame_number = frame_number - 1 if frame_number >= 0 else 0
                elif c == "o":
                    frame_number += 10
                elif c == "a":
                    frame_number = frame_number - 10 if frame_number >= 10 else 0

        if debug_mode:
            print_debug(blinks, frame_number)
            try:
                l_ear, r_ear = ear.calc(frame)
                eye_marks = ear.eye_marks(frame)
                frame = draw_eye_marks(frame, eye_marks)

                fig.clear()

                if len(xs) < 13 or frame_number > xs[-1]:
                    xs.append(frame_number)
                    ys.append(l_ear)
                    (line1,) = plt.plot(xs, ys, "ko-")
                    line1.set_xdata(xs)
                    line1.set_ydata(ys)

                    fig.canvas.draw()

                    img = np.fromstring(
                        fig.canvas.tostring_rgb(), dtype=np.uint8, sep=""
                    )
                    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                    # img is rgb, convert to opencv's default bgr
                    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

                    cv.imshow("EAR", img)

            except FaceDetectException:
                print(f"No face detected at frame: {frame_number}")
                xs.append(frame_number)
                ys.append(None)

                frame_number += 1
