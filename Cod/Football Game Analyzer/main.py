import torch
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO
import supervision as sv
from AnalyzeData import Analyze
from Trackers import Tracker
from ReadVideo import ReadVideo
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

def main():
    analyze = Analyze()
    analyze.function()
    analyze.plot()
    if torch.cuda.is_available():
        device = 'cuda'
        print("Lucram pe GPU")
    else:
        device = 'cpu'
        print("Lucram pe CPU")

    read_video = ReadVideo()
    read_video.process_video_with_tracker()
    players_dict_xyxy = read_video.getPlayers_dict_xyxy()

    frames_generator = sv.get_video_frames_generator('C:\\Users\\dariu\\Downloads\\08fd33_4.mp4')
    # for frame in frames_generator:
    #     for player in players_dict_xyxy:
    #         for idx, (name, coords) in enumerate(player.items()):
    #             x1, y1, x2, y2 = coords
    #             x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    #             cv2.imwrite('C:\\Users\\dariu\\OneDrive\\Desktop\\Licenta\\Cod\\Football Game Analyzer\\Output\\img.jpg', frame[y1:y2, x1:x2])
    #             break
    #         break
    #     break

    # tracker = Tracker().set_tracking('Input Videos/video_meci1.mp4', 'C:/Users/dariu/OneDrive/Desktop/Licenta/Rezultate Videoclipuri')


if __name__ == "__main__":
    main()
    # box_annotator = sv.BoxAnnotator()
    # text_label = sv.LabelAnnotator()
    # video_info = sv.VideoInfo.from_video_path('Input Videos/08fd33_4.mp4')
    # print(video_info)
    # frames = sv.get_video_frames_generator('Input Videos/08fd33_4.mp4')
    # frame = next(frames)
    # model = YOLO('Models/best.pt')
    # results = model.predict(frame)
    # result = results[0]
    #
    # detections = sv.Detections.from_ultralytics(result)
    # labels = [
    #     f"{class_name} {confidence:.2f} "
    #     for class_name, confidence
    #     in zip(detections['class_name'], detections.confidence)
    # ]
    # annotated_detection = frame.copy()
    # annotated_detection = box_annotator.annotate(scene=annotated_detection, detections=detections)
    # annotated_detection = text_label.annotate(scene=annotated_detection, detections=detections, labels=labels)
    # sv.plot_image(annotated_detection)


    # main()
    # if torch.cuda.is_available():
    #   device = 'cuda'
    #   print("Lucram pe GPU")
    # else:
    #   device = 'cpu'
    #   print("Lucram pe CPU")
    #
    # video_frames = cv2.VideoCapture('soccer_match.mp4')
    #
    # model = YOLO('Models/best.pt').to(device)
    #
    # results = model.predict('Input Videos/video_meci1.mp4', save=True)
    # # iau primul frame din videoclip
    # result = results[0]
    #
    # for box in result.boxes:
    #   class_id = result.names[box.cls[0].item()]
    #   coordinates = box.xyxy[0].tolist()
    #   coordinates = [round(x) for x in coordinates]
    #   prob = box.conf[0].item()
    #   print("Object type is:", class_id)
    #   print("Object coordinates are:", coordinates)
    #   print("Object probability is:", prob)
    #   print("/n")
    # import required libraries



    # img = cv2.imread('C:\\Users\\dariu\\Downloads\\car.jpg')
    #
    # # Convert BGR to HSV
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # plt.imshow(hsv)
    # plt.show()
    #
    # # define range of blue color in HSV
    # lower_yellow = np.array([15, 50, 180])
    # upper_yellow = np.array([40, 255, 255])
    #
    # # Create a mask. Threshold the HSV image to get only yellow colors
    # mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    #
    # # Bitwise-AND mask and original image
    # result = cv2.bitwise_and(img, img, mask=mask)
    #
    # # display the mask and masked image
    # cv2.imshow('Mask', mask)
    # cv2.waitKey(0)
    # cv2.imshow('Masked Image', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
