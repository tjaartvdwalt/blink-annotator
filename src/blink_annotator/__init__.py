# import typer
#
from blink_annotator.annotator import annotate
#
#
# def main():
#     typer.run(main)
#
from typing import Optional

import typer

app = typer.Typer()


@app.command()
def main(video_file: str, max_height: int = -1, start_frame: int = 0):
    annotate(video_file, max_height, start_frame)
