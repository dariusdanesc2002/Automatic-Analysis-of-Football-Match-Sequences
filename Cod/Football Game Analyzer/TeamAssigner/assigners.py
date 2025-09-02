import cv2
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np


class Assigner:
    def __init__(self):
        self.playersColours = []
        # self.teamA = None
        # self.teamB = None

    @staticmethod
    def getCroppedImg(self, player_img):

        player_img = cv2.cvtColor(player_img, cv2.COLOR_BGR2LAB)
        h, w = player_img.shape[:2]
        x1 = int(w * 0.15)
        x2 = int(w * 0.85)
        y1 = int(h * 0.15)
        y2 = int(h * 0.50)
        cropped_player_img = player_img[y1:y2, x1:x2]
        return cropped_player_img


    def getPlayerColour(self, player_img):
        """
               Get the colour of the player

               Parameters
               ----------
               player_img : numpy.ndarray
                  Image to detect the colour

               Returns
               -------
               Returns in self.playersColours all the
               jersey colours detected
        """
        player_img = cv2.cvtColor(player_img, cv2.COLOR_BGR2RGB)
        h, w = player_img.shape[:2]
        x1 = int(w * 0.15)
        x2 = int(w * 0.85)
        y1 = int(h * 0.15)
        y2 = int(h * 0.50)
        cropped_player_img = player_img[y1:y2, x1:x2]
        # cropped_player_img = player_img[0:int(h/2), :]
        # plt.imshow(cropped_player_img)
        # plt.show()
        cropped_player_img_reshaped = cropped_player_img.reshape(cropped_player_img.shape[0] * cropped_player_img.shape[1], 3)
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(cropped_player_img_reshaped)
        #Cum stiu cand sa iau 0 sau cand sa iau 1??
        labels = kmeans.labels_
        segmentationImage = labels.reshape(cropped_player_img.shape[0], cropped_player_img.shape[1])
        corners = [segmentationImage[0, 0], segmentationImage[0, -1], segmentationImage[-1, 0], segmentationImage[-1, -1]]
        background = max(corners, key=corners.count)
        # plt.imshow(segmentationImage)
        # plt.show()
        self.playersColours.append({'colour': kmeans.cluster_centers_[1 - background]})
        # print(kmeans.cluster_centers_[1 - background])

    def detectColours(self, frame, players_dict_xyxy):
        """
               Get the colours of teamA and teamB

               Parameters
               ----------
               frame : numpy.ndarray
                  Image to detect the colour
               players_dict_xyxy : list of dict
                  Dictionary of players coordinates

               Returns
               -------
                Return the colours of teamA and teamB
        """
        for player in players_dict_xyxy:
            for _, (name, position) in enumerate(player.items()):
                x1, y1, x2, y2 = position
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                player_img = frame[y1:y2, x1:x2]
                self.getPlayerColour(player_img)

        colours = [color['colour'] for color in self.playersColours]
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(colours)
        teamA = kmeans.cluster_centers_[0]
        teamB = kmeans.cluster_centers_[1]
        return teamA, teamB
        # print(kmeans.cluster_centers_)

    def checkTeam(self, player_img, team):
        """
               Function that returns the convolution
               between the mask and the colour of teamA/teamB

               Parameters
               ----------
               player_img : numpy.ndarray
                  Image of the player
               team : numpy.ndarray
                  Colour of either teamA or teamB

               Returns
               -------
                Return the result of the convolution
        """

        lower = np.clip(team - 40, 0, 255)
        upper = np.clip(team + 40, 0, 255)
        mask = cv2.inRange(player_img, lower, upper)
        result = cv2.bitwise_and(player_img, player_img, mask=mask)
        return result


    @staticmethod
    def countNumberOfZeros(result):
        """
               Static function that counts the number of zeros

               Parameters
               ----------
               result : numpy.ndarray
                  Image of the convolution return by checkTeam

               Returns
               -------
                Returns a counter with the number of zeros from result w
        """
        counter = 0
        x, y, z = result.shape
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    if result[i, j, k] == 0:
                        counter += 1
        return counter

    def assignPlayersToTeam(self, frame, players_dict_xyxy, teamA, teamB):
        """
               Important function that returns the list of colours of all players in the frame

               Parameters
               ----------
               frame : numpy.ndarray
                  Image to detect the colour
               players_dict_xyxy : list of dict
                  Dictionary of players coordinates
                teamA : list of numbers
                  Colour of teamA
                teamB : list of numbers
                  Colour of teamB
               Returns
               -------
                Returns a list of colours, each colour represent the colour of
                every player in the frame
        """
        colours = []
        for player in players_dict_xyxy:
            for _, (name, position) in enumerate(player.items()):
                x1, y1, x2, y2 = position
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                player_img = frame[y1:y2, x1:x2]
                player_img = cv2.cvtColor(player_img, cv2.COLOR_BGR2RGB)
                h, w = player_img.shape[:2]
                x1 = int(w * 0.15)
                x2 = int(w * 0.85)
                y1 = int(h * 0.15)
                y2 = int(h * 0.50)
                player_img = player_img[y1:y2, x1:x2]
                # plt.imshow(player_img)
                # plt.show()
                resultA = self.checkTeam(player_img, teamA)
                resultB = self.checkTeam(player_img, teamB)
                counterA = self.countNumberOfZeros(resultA)
                counterB = self.countNumberOfZeros(resultB)
                if counterA > counterB:
                    colours.append({'colour': teamB})
                else:
                    colours.append({'colour': teamA})

                # print(counterA, counterB)
                # plt.imshow(resultA)
                # plt.show()
                # plt.imshow(resultB)
                # plt.show()
        return colours






