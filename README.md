# blink_annotator
A tool to for annotating blink videos

## Keyboard shortcuts

### View Mode

The app has a Vim like mode system, with 3 modes:

`View mode` can be used to view the video, but does not change any blink setting.

`Insert mode` allows the user to set the blink state for the current frame. By default it sets the blink state for the current frame, to that of the previous frame (This increases the speed of annotation, because you only have to toggle the state once for every blink cycle.)

`Debug mode` can be used together with the other modes, to print the eye aspect ratio and facial landmarks.

| Shortcut  | Action                     |
|-----------|----------------------------|
| i         | Switch to insert mode      |
| d         | Enable debug mode          |
| n, Space  | Skip to the next frame     |
| o         | Skip 10 frames forward     |
| p         | Skip to the previous Frame |
| a         | Skip 10 frames back        |
| s         | Save the annotations file  |
| q         | Quit                       |


### Insert Mode
| Shortcut | Action |
|----------|-------------------------------------------------|
| Esc      | Switch to view mode                             |
| d        | Enable debug mode                               |
| b, Enter | Toggle blink setting for the current frame      |
| n, Space | Skip to the next frame (save current state)     |
| p        | Skip to the previous frame (save current state) |
| s        | Save the annotations file                       |
| q        | Quit                                            |

## Output files

The internal blink state is saved in a `.blink` file in the same directory as the original video. (This is a Serialized version of the state object)

When pressing `s` a `.tag` file is created that saves the blink states in the format described by [Fogelton](https://www.blinkingmatters.com/research)
