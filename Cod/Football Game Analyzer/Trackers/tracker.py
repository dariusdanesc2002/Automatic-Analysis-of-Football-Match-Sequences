from ultralytics import YOLO
import supervision as sv
import cv2
import os
import numpy as np


class Tracker:
    def __init__(self):
        self.model = YOLO('Models/best.pt')
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
        )

    def set_tracking(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)[0]

            detections = sv.Detections.from_ultralytics(results)

            tracked_objects = self.tracker.update_with_detections(detections)

            # # Annotate frame (optional)
            # labels = [
            #     f"ID: {tracker_id} {self.model.model.names[class_id]}"
            #     for _, _, _, _, class_id, tracker_id in tracked_objects
            # ]
            frame = self.box_annotator.annotate(
                scene=frame,
                detections=tracked_objects,
            )
            out.write(frame)
            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
