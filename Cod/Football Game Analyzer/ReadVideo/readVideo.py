import torch
import cv2
import numpy as np
import pandas as pd
from supervision import Position, Detections
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
from TeamAssigner import Assigner
from Possesion import Possession
from matplotlib import pyplot as plt


class ReadVideo:
    def __init__(self):
        self.video_path = 'C:\\Users\\dariu\\OneDrive\\Desktop\\Licenta\\Videoclipuri\\fcsb.mp4'
        self.output_path = 'C:\\Users\\dariu\\OneDrive\\Desktop\\Licenta\\Rezultate Videoclipuri\\fcsb.mp4'
        self.box_annotator = sv.BoxAnnotator(

        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['E0E0E0', 'D0E7FF']),
            text_color=sv.ColorPalette.from_hex(['#333333', '#1A237E'])

        )
        self.round_annotator = sv.RoundBoxAnnotator(
            thickness=3,
            roundness=1,

        )
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(

        )
        self.ellipse_annotator_goalkeeper = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['000000'])

        )
        self.ellipse_annotator_referee = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['ff0000'])

        )
        self.ellipse_annotator_teamA = sv.EllipseAnnotator(

        )
        self.ellipse_annotator_teamB = sv.EllipseAnnotator(

        )
        self.model = YOLO('Models/best.pt')
        self.tracker = sv.ByteTrack()
        self.tracker_ball = sv.ByteTrack()
        self.detections = None
        self.players_dict_xyxy = []
        self.first_frame_processed = False
        self.teamA = None
        self.teamB = None

    def process_video(self):

        video_info = sv.VideoInfo.from_video_path(video_path='C:\\Users\\dariu\\Downloads\\08fd33_4.mp4')
        frames_generator = sv.get_video_frames_generator(self.video_path)

        with sv.VideoSink(target_path=self.output_path, video_info=video_info) as sink:
            for frame in tqdm(frames_generator, total=video_info.total_frames):
                results = self.model.predict(frame)
                result = results[0]
                self.detections = sv.Detections.from_ultralytics(result)
                labels = [
                    f"{class_name} {confidence:.2f} "
                    for class_name, confidence
                    in zip(self.detections['class_name'], self.detections.confidence)
                ]
                annotated_detection = frame.copy()
                annotated_detection = self.box_annotator.annotate(scene=annotated_detection, detections=self.detections)
                annotated_detection = self.label_annotator.annotate(scene=annotated_detection, detections=self.detections, labels=labels)

                sink.write_frame(annotated_detection)

    def getPlayers_dict_xyxy(self):
        for detection in self.detections:
            if detection[5].get('class_name') == 'player':
                self.players_dict_xyxy.append({'player': detection[0]})

        return self.players_dict_xyxy

    def assignTeamA(self, detections, team):

        players = []
        i = 0
        for detection in detections:
            if detection[5]['colour']['colour'][0] == team[0] and detection[5]['colour']['colour'][1] == team[1] and detection[5]['colour']['colour'][2] == team[2]:
                players.append(detections[i][0])
            i += 1

        return Detections.merge(players)

    def assignTeamB(self, detections, team):

        players = []
        i = 0
        for detection in detections:
            if detection[5]['colour']['colour'][0] == team[0] and detection[5]['colour']['colour'][1] == team[1] and detection[5]['colour']['colour'][2] == team[2]:
                players.append(detections[i][0])
            i += 1

        return Detections.merge(players)


    def extrapolation(self, ball_positions, ball_timestamps):
        delta_t = abs(ball_timestamps[1] - ball_timestamps[0])
        x1 = ball_positions[0]
        x2 = ball_positions[2]
        x3 = ball_positions[4]
        y1 = ball_positions[1]
        y2 = ball_positions[3]
        y3 = ball_positions[5]
        v_x1 = (x2 - x1) / delta_t
        v_x2 = (x3 - x2) / delta_t
        a_x = (v_x2 - v_x1) / delta_t
        v_y1 = (y2 - y1) / delta_t
        v_y2 = (y3 - y2) / delta_t
        a_y = (v_y2 - v_y1) / delta_t

        x_pred = x3 + v_x2 * delta_t + 1/2 * a_x * delta_t**2
        y_pred = y3 + v_y2 * delta_t + 1/2 * a_y * delta_t**2

        return  x_pred, y_pred

    def predict_position(self, positions, timestamps):
        degree = 3
        t_future = 0.16
        coeffs = np.polyfit(timestamps, positions, deg=degree)
        position_future = np.polyval(coeffs, t_future)
        return position_future

    def box_area(self,  x_pred_top_left, y_pred_top_left, x_pred_bottom_right, y_pred_bottom_right):
        return [abs((y_pred_bottom_right - y_pred_top_left) * (x_pred_bottom_right - x_pred_top_left))]

    def getCoordsTeam(self, player_detections_team):
        distances = []
        for coords in player_detections_team:
            distances.append(coords[0])

        return distances

    def process_video_with_tracker(self):
        video_info = sv.VideoInfo.from_video_path(self.video_path)
        frames_generator = sv.get_video_frames_generator(self.video_path)
        possession = Possession()
        ball_positions = []
        timestamps = []
        frame_index = 0
        with sv.VideoSink(target_path=self.output_path, video_info=video_info) as sink:
            for frame in tqdm(frames_generator, total=video_info.total_frames):
                timestamp = frame_index / video_info.fps
                results = self.model.predict(frame)
                result = results[0]
                self.detections = sv.Detections.from_ultralytics(result)
                annotated_detection = frame.copy()
                ball_detections = self.detections[self.detections.class_id == 0]
                timestamps.append(timestamp)
                frame_index += 1
                if ball_detections.xyxy.size > 0:
                    ball_positions.append(ball_detections.xyxy)
                else:
                    ball_position_top_left = [ball_positions[-3][0][0], ball_positions[-3][0][1], ball_positions[-2][0][0], ball_positions[-2][0][1], ball_positions[-1][0][0], ball_positions[-1][0][1]]
                    ball_position_bottom_right = [ball_positions[-3][0][2], ball_positions[-3][0][3], ball_positions[-2][0][2], ball_positions[-2][0][3], ball_positions[-1][0][2], ball_positions[-1][0][3]]

                    ball_timestamps_top_left = [timestamps[-2], timestamps[-1]]
                    ball_timestamps_bottom_right = [timestamps[-2], timestamps[-1]]

                    x_pred_top_left, y_pred_top_left = self.extrapolation(ball_position_top_left, ball_timestamps_top_left)
                    x_pred_bottom_right, y_pred_bottom_right = self.extrapolation(ball_position_bottom_right, ball_timestamps_bottom_right)

                    ball_positions.append([[x_pred_top_left, y_pred_top_left, x_pred_bottom_right, y_pred_bottom_right]])
                    # ball_detections.area = list(map(float, self.box_area(x_pred_top_left, y_pred_top_left, x_pred_bottom_right, y_pred_bottom_right)))
                    area_vals = [float(a) for a in self.box_area(
                        x_pred_top_left,
                        y_pred_top_left,
                        x_pred_bottom_right,
                        y_pred_bottom_right
                    )]
                    # __setitem__ pune 'area' și 'box_area' în dict-ul intern de metadate
                    ball_detections.__setitem__('area', area_vals)
                    ball_detections.__setitem__('box_area', area_vals)
                    # ball_detections.box_area = ball_detections.area
                    ball_detections.xyxy = np.array([[
                        float(x_pred_top_left),
                        float(y_pred_top_left),
                        float(x_pred_bottom_right),
                        float(y_pred_bottom_right)
                    ]])
                    ball_detections.class_id = np.array([0])

                goalkeeper_detections = self.detections[self.detections.class_id == 1]
                player_detections = self.detections[self.detections.class_id == 2]
                # print(type(player_detections))
                referee_detections = self.detections[self.detections.class_id == 3]
                detections = self.tracker.update_with_detections(player_detections)
                # detections = self.tracker.update_with_detections(goalkeeper_detections)

                # Aici vom implementa partea de segmentare pe echipe.(assigner)

                self.getPlayers_dict_xyxy()
                assigner = Assigner()

                if not self.first_frame_processed:
                    self.teamA, self.teamB = assigner.detectColours(frame, self.players_dict_xyxy)
                    self.first_frame_processed = True

                colours = assigner.assignPlayersToTeam(frame, self.players_dict_xyxy, self.teamA, self.teamB)

                labels = [
                    f"{class_id}:{tracker_id} {confidence:.2f} "
                    for class_id, tracker_id, confidence
                    in zip(detections['class_name'], detections.tracker_id, detections.confidence)
                ]


                detections.__setitem__('colour', colours)
                player_detections.__setitem__('colour', colours)
                player_detections_teamA = detections.empty()
                player_detections_teamA = self.assignTeamA(detections, self.teamA)
                player_detections_teamB = detections.empty()
                player_detections_teamB = self.assignTeamB(detections, self.teamB)
                intRGB = map(int, self.teamA)
                r, g, b = intRGB
                ellipse_annotator_teamA = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex([sv.Color(r=r,g=g,b=b).as_hex()]))
                intRGB = map(int, self.teamB)
                r, g, b = intRGB
                ellipse_annotator_teamB = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex([sv.Color(r=r,g=g,b=b).as_hex()]))


                # for player in player_detections_teamB:
                #     x1, y1, x2, y2 = player[0]
                #     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                #     plt.imshow(frame[y1:y2, x1:x2])
                #     plt.show()
                #     plt.close()

                annotated_detection = self.round_annotator.annotate(scene=frame, detections=ball_detections)
                annotated_detection = ellipse_annotator_teamA.annotate(scene=frame, detections=player_detections_teamA)
                annotated_detection = ellipse_annotator_teamB.annotate(scene=frame, detections=player_detections_teamB)
                annotated_detection = self.ellipse_annotator_referee.annotate(scene=frame, detections=referee_detections)
                annotated_detection = self.ellipse_annotator_goalkeeper.annotate(scene=frame, detections=goalkeeper_detections)
                annotated_detection = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)
                coords_A = self.getCoordsTeam(player_detections_teamA)
                coords_B = self.getCoordsTeam(player_detections_teamB)

                if ball_detections.xyxy.size > 0:
                    possession.calculatePossession(ball_detections, coords_A, coords_B)
                else:
                    possession.calculatePossessionWithoutBall()

                possession.increaseNumberFrames()
                possession.getPossession()

                self.players_dict_xyxy.clear()
                colours.clear()
                sink.write_frame(annotated_detection)














