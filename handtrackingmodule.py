'''
mediapipe is ML modelling package that allows to configure and recognise
many landmarks such as face or hands.
'''
# creating a module out of this
import cv2
import mediapipe as mp
import time


class handdetector():
    # default parameters already in place
    def __init__(self, mode=False, maxhands=2, detectionconfidence=0.5, trackconfidence=0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.detectionconfidence = detectionconfidence
        self.trackconfidence = trackconfidence

        self.mphands = mp.solutions.hands
        # creating a mediapipe hands object, first does detection => then tracking
        self.hands = self.mphands.Hands(self.mode, self.maxhands, self.detectionconfidence,
                                        self.trackconfidence)  # default parameters already given to object
        # drawing landmarks
        self.mpdraw = mp.solutions.drawing_utils

    def findhands(self, frame, draw=True):
        # send our rgb image to object(only uses rgb) , hence conversion necessary
        imgrgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Processes an RGB image and returns the hand landmarks and handedness
        # of each detected hand.
        self.results = self.hands.process(imgrgb)
        # checking for multiple hands using a for loop

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:  # for each hand
                if draw:
                    self.mpdraw.draw_landmarks(frame, handlms, self.mphands.HAND_CONNECTIONS)
        return frame

    def findposition(self,frame,handnumber=0,draw=True):

        lmlist=[]

        #if any hands were detected or not
        if self.results.multi_hand_landmarks:
            #which hand we are talking about , 1 particular hand
            myhand=self.results.multi_hand_landmarks[handnumber]
            for id, lm in enumerate(myhand.landmark):
                # print(id, lm)
                h, w, c = frame.shape
                # center position for each landmark
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(frame,(cx,cy),5,(255,0,0),cv2.FILLED)
        return lmlist

