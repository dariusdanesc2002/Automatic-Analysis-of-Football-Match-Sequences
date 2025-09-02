import math
from collections import Counter

class Possession:
    def __init__(self):
        # Amount of consecutive frames new team has to have the ball in order to change possession
        self.possesion_counter_threshold = 30
        # Distance in pixels from player to ball in order to consider a player has the ball
        self.ball_distance_threshold = 10
        self.timeA = 0
        self.timeB = 0
        self.whoHasPossession = 'B'
        self.inertia = 5
        self.numberFrames = 0
        self.previousBallPossessions = []

    def increaseNumberFrames(self):
        self.numberFrames += 1

    def getCenterOfBall(self, ballCoords):
        xCenter = (ballCoords[0] + ballCoords[2]) / 2
        yCenter = (ballCoords[1] + ballCoords[3]) / 2
        return xCenter, yCenter

    def getDistanceBall2Player(self, ballCoords, playerCoords):
        xb, yb = ballCoords
        xp, yp = playerCoords
        return math.sqrt((xb - xp) ** 2 + (yb - yp) ** 2)

    def getDistanceBall2Players(self, coordsBall, coordsTeam):
        euclidianDistances = []
        centerBall = self.getCenterOfBall(coordsBall.xyxy[0])
        for player in coordsTeam:
            euclidianDistances.append(self.getDistanceBall2Player(centerBall, [player[0], player[1]]))
            euclidianDistances.append(self.getDistanceBall2Player(centerBall, [player[2], player[3]]))

        return min(euclidianDistances)

    def calculatePossession(self, coordsBall, coords_A, coords_B):
        self.previousBallPossessions = [self.whoHasPossession]
        euclidianDistancesA = self.getDistanceBall2Players(coordsBall, coords_A)
        euclidianDistancesB = self.getDistanceBall2Players(coordsBall, coords_B)

        if euclidianDistancesA <= self.ball_distance_threshold or euclidianDistancesB <= self.ball_distance_threshold:
            if euclidianDistancesA > euclidianDistancesB:
                if self.whoHasPossession == 'A' and (self.numberFrames % self.possesion_counter_threshold == 0):
                    self.whoHasPossession = 'B'
                    self.timeB += 1
                else:
                    self.timeA += 1
                self.previousBallPossessions.append(self.whoHasPossession)
            elif euclidianDistancesA < euclidianDistancesB:
                if self.whoHasPossession == 'B' and (self.numberFrames % self.possesion_counter_threshold == 0):
                    self.whoHasPossession = 'A'
                    self.timeA += 1
                else:
                    self.timeB += 1

                self.previousBallPossessions.append(self.whoHasPossession)

        else:
            previousRecordedTeams = self.previousBallPossessions[-5:]
            counter = Counter(previousRecordedTeams)
            most_common_letter, count = counter.most_common(1)[0]
            self.previousBallPossessions.append(most_common_letter)
            if most_common_letter == 'A':
                self.timeA += 1
            if most_common_letter == 'B':
                self.timeB += 1

    def calculatePossessionWithoutBall(self):
        previousRecordedTeams = self.previousBallPossessions[-5:]
        counter = Counter(previousRecordedTeams)
        most_common_letter, count = counter.most_common(1)[0]
        self.previousBallPossessions.append(most_common_letter)
        if most_common_letter == 'A':
            self.timeA += 1
        if most_common_letter == 'B':
            self.timeB += 1

    def getPossession(self):
        print(f'Possession Team A is: {(self.timeA / self.numberFrames) * 100}')
        print(f'Possession Team B is: {(self.timeB / self.numberFrames) * 100}')
        print(self.numberFrames)