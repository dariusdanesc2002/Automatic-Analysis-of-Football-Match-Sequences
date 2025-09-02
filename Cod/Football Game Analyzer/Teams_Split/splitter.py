import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

class Spliter:
    def __init__(self):
        self.ID_PLAYER = 2
        self.players_dict_complete = {'team_id': [], 'player': [], 'colour_player': []}
        self.teamA = None
        self.teamB = None

    # def firsSegmentation(self, image_path):
    #     img = cv2.imread(image_path)
    #     height, width = img.shape[:2]
    #     new_img = img[0:int(height/2), :]
    #     new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    #     plt.imshow(new_img)
    #     plt.show()
    #     new_img_reshape = new_img.reshape(new_img.shape[0]*new_img.shape[1], 3)
    #     kmeans = KMeans(n_clusters=2, random_state=0)
    #     kmeans.fit(new_img_reshape)
    #     labels = kmeans.labels_
    #     segmentationImage = labels.reshape(new_img.shape[0], new_img.shape[1])
    #     player_cropped = []
    #     plt.imshow(segmentationImage)
    #     plt.show()
    #     player_mask = np.where(segmentationImage == 1, 1, 0).astype(np.uint8)
    #     player_extracted = cv2.bitwise_and(new_img, new_img, mask=player_mask)
    #     player_extracted_reshape = player_extracted.reshape(player_extracted.shape[0] * player_extracted.shape[1], 3)
    #     kmeans.fit(player_extracted_reshape)
    #     labels = kmeans.labels_
    #     segmentationImage = labels.reshape(new_img.shape[0], new_img.shape[1])
    #     plt.imshow(segmentationImage)
    #     plt.show()
    #     plt.imshow(player_extracted)
    #     plt.show()
    def firstSegmentation(self, player_img):
        new_player_img = cv2.cvtColor(player_img, cv2.COLOR_BGR2RGB)
        h,w = new_player_img.shape[:2]
        #Luam doar jumatate din jucator,pentru ca ne intereseaza doar culoarea tricoului sau
        new_player_img = new_player_img[0:int(h/2), :]
        new_player_img_reshape = new_player_img.reshape(new_player_img.shape[0]*new_player_img.shape[1], 3)
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(new_player_img_reshape)
        return kmeans.cluster_centers_[1]

    def playerColour(self, frame, players_dict_xyxy):
        #aici punem toate culorile  identificate pe jucatori
        players_colours = []
        for player in players_dict_xyxy:
            for _, (name, coords) in enumerate(player.items()):
                x1, y1, x2, y2 = coords
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                player_img = frame[y1:y2, x1:x2]
                player_colour = self.firstSegmentation(player_img)
                players_colours.append(player_colour)

        return players_colours

    def assignPlayerColours(self, players_colours, players_dict_xyxy):

        for colour, player in zip(players_colours, players_dict_xyxy):
            self.players_dict_complete['colour_player'].append(colour)
            self.players_dict_complete['player'].append(player)

        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(players_colours)
        self.teamA = kmeans.cluster_centers_[0]
        self.teamB = kmeans.cluster_centers_[1]

        for player, colour in zip(self.players_dict_complete['player'], self.players_dict_complete['colour_player']):
            team_id = kmeans.predict(colour.reshape(1, -1))[0]
            self.players_dict_complete['team_id'].append(team_id)






