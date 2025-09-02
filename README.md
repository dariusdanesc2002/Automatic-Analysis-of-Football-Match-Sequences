# Automatic Analysis of Football Match Sequences

This project is a comprehensive football game analyzer that leverages computer vision and deep learning techniques to extract valuable insights from football match videos. The system is capable of detecting and tracking players, the ball, and referees, automatically assigning players to their respective teams, and calculating ball possession throughout the game. 

## Features

-   **Object Detection:** Utilizes a fine-tuned YOLOv8 model to accurately detect key entities on the football pitch, including players, goalkeepers, referees, and the football.
-   **Object Tracking:** Employs the ByteTrack algorithm for robust and efficient tracking of all detected objects across video frames.
-   **Automatic Team Assignment:** A sophisticated team assignment module that analyzes the dominant colors of the players' jerseys in the initial frame to automatically group them into their respective teams.
-   **Ball Possession Calculation:** Calculates and displays real-time ball possession statistics, giving a clear indication of team dominance.
-   **Data Analysis:** Includes scripts to analyze the distribution of classes in the training dataset, providing insights into the data used to train the object detection model.
  
## How It Works

The core of the project lies in a pipeline that processes the football match video frame by frame:

1.  **Model Loading:** A pre-trained YOLOv8 model is loaded, ready for object detection.
2.  **Video Processing:** The input video is read frame by frame.
3.  **Object Detection:** For each frame, the YOLOv8 model detects players, the ball, goalkeepers, and referees.
4.  **Object Tracking:** The detected objects are then passed to the ByteTrack algorithm, which assigns a unique ID to each object and tracks its movement across subsequent frames.
5.  **Team Assignment:** In the first frame of the video, the system identifies the players and analyzes the color of their jerseys to automatically assign them to 'Team A' or 'Team B'.
6.  **Ball Possession:** The system calculates ball possession by determining which team is in closest proximity to the ball at any given moment.
7.  **Annotation and Output:** The processed video is annotated with bounding boxes, labels, and tracking IDs for all detected objects, and the final video is saved to a specified output path.

