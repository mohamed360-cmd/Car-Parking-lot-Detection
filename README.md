## Car Parking Lot Detection
# Overview
    This program detects available parking spaces in a parking lot video. It uses the YOLO algorithm to count the number of vehicles in each frame and determines the number of free parking slots.

# How It Works
    User Input: The program prompts you to enter the total number of parking slots.

# Note: For the provided video, enter 20 as the number of parking slots. If you use your own video, remember to rename it accordingly or modify the script.
    Video Processing: The program loads the video and uses the YOLO algorithm to analyze each frame, counting the number of vehicles.

    Calculation: It subtracts the number of vehicles from the total number of parking slots to determine the number of free spaces. It assumes that a parking space is considered free once a vehicle leaves.

# Author
    Mohamed Abdildaif