import re
from dataclasses import dataclass


class MalformedEntryError(Exception):
    pass


@dataclass
class EyeBlinkEntry:
    frame_id: int = 0
    blink_id: int = -1
    non_frontal_face: bool = False
    left_eye_fully_closed: bool = False
    left_eye_not_visible: bool = False
    right_eye_fully_closed: bool = False
    right_eye_not_visible: bool = False
    face: tuple[int, int, int, int] = (0, 0, 0, 0)
    left_eye_left: tuple[int, int] = (0, 0)
    left_eye_right: tuple[int, int] = (0, 0)
    right_eye_left: tuple[int, int] = (0, 0)
    right_eye_right: tuple[int, int] = (0, 0)

    def __repr__(self):
        return f"{self.frame_id}:{self.blink_id}:{'N' if self.non_frontal_face else 'X'}:{'C' if self.left_eye_fully_closed else 'X'}:{'N' if self.left_eye_not_visible else 'X'}:{'C' if self.right_eye_fully_closed else 'X'}:{'N' if self.right_eye_not_visible else 'X'}:{self.face[0]}:{self.face[1]}:{self.face[2]}:{self.face[3]}:{self.left_eye_left[0]}:{self.left_eye_left[1]}:{self.left_eye_right[0]}:{self.left_eye_right[1]}:{self.right_eye_left[0]}:{self.right_eye_left[1]}:{self.right_eye_right[0]}:{self.right_eye_right[1]}"

    def parse_line(self, line):
        line = line.rstrip("\n")
        entries = re.split(r":", line)

        if len(entries) < 19:
            raise MalformedEntryError(f"Malformed entry: {line}")

        self.frame_id = int(entries[0])
        self.blink_id = int(entries[1])
        self.non_frontal_face = True if entries[2] == "N" else False
        self.left_eye_fully_closed = True if entries[3] == "C" else False
        self.left_eye_not_visible = True if entries[4] == "N" else False
        self.right_eye_fully_closed = True if entries[5] == "C" else False
        self.right_eye_not_visible = True if entries[6] == "N" else False
        self.face = (
            (int(entries[7]), int(entries[8]),
             int(entries[9]), int(entries[10]))
            if (
                entries[7] != ""
                or entries[8] != ""
                or entries[9] != ""
                or entries[10] != ""
            )
            else None
        )
        self.left_eye_left = (
            (int(entries[11]), int(entries[12]))
            if (entries[11] != "" or entries[12] != "")
            else None
        )
        self.left_eye_right = (
            (int(entries[13]), int(entries[14]))
            if (entries[13] != "" or entries[14] != "")
            else None
        )
        self.right_eye_left = (
            (int(entries[15]) or None, int(entries[16]))
            if (entries[15] != "" or entries[16] != "")
            else None
        )
        self.right_eye_right = (
            (int(entries[17]), int(entries[18]))
            if (entries[17] != "" or entries[18] != "")
            else None
        )
