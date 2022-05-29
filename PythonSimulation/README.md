# Python Simulation


## Dependencies

### mediapipe: "pip install mediapipe"
### tensorflow: "pip install tensorflow"
### numpy: "pip install numpy"
### openCV: "pip install opencv-python"


## Data Collection

### Unfortunately the data acquired for the ios model isn't compatible with models running on a computer (nor will any model built with the iOS data) so for a python simulation data will have to be acquired. There are some samples in **data_indiv** that can build a decent A-B Classifier.

1. open 'config_indiv.settings', set the variables how you want and save file:
    - parent_directory: directory where samples will be put named "data_indiv"
    - gesture: letter you want to make samples for (Note: make sure letter is CAPITALIZED)
    - number_of_vids: number of videos
    - video_source: May vary with your computer (normally 0), if you don't know then play around
    - setup_check: if setup_check is True, then data will not be saved to files

2. with all variables set, open a terminal and navigate to *Pipeline*
3. run the following command "python data_collection_indiv.py"

    
## Build Model

1. open a terminal and navigate to *Pipeline*
2. open "gesture_indiv.names" in a text editor
3. put your class labels (A,B,C, etc...) on individual lines and save
4. run the following command "python build_model_indiv.py"

## Test Model

1. open a terminal and navigate to *Pipeline*
2. build a test dataset, simple change parent_directory in 'config_indiv.settings' from "data_indiv" to "test_data_indiv" and build dataset the same way as the other one
3. run the following command "python test_model_indiv.py"

## Run Model

1. open a terminal and navigate to *Pipeline*
2. run the following command "python run_model_indiv.py"
3. when finished press ctrl+C in the terminal to stop

