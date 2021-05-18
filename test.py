import cv2
import handtrackingmodule as htm
import numpy as np
import autopy

cap = cv2.VideoCapture(0)

###########################
wcam, hcam = 640, 480
wscr, hscr = autopy.screen.size()
frameR = 50  # frame reduction

plocx, plocy = 0, 0
clocx, clocy = 0, 0
smoothening = 7
###########################

cap.set(3, wcam)
cap.set(4, hcam)

detector = htm.handdetector(maxhands=1)

tipid = [4, 8, 12, 16, 20]
# thumb , index, middle , ring , pinky

while True:
    _, frame = cap.read()
    '''
    1.detect hands and landmarks
    2.check which hand is up (our of index and middle)
    3.get location of tip of index and middle
    4.smoothen the values acc to smoothning algorithm
    5.if only index up :(moving mode)
        set interpolation values acc to window screen resolution
        move mouse 
    6.if both index and middle mouse if up :(selection mode i.e. clicking mode)
        using autopy select the position
    '''
    # findinga hand landmarks

    frame = detector.findhands(frame)
    lmlist = detector.findposition(frame, draw=False)
    # print(lmslist)

    cv2.rectangle(frame, (frameR, frameR), (wcam - frameR, hcam - frameR),
                  (255, 0, 0), 2)

    # finding finger up
    if len(lmlist) != 0:
        fingers = []
        # for thumb , x position of tip and mid is checked
        if lmlist[tipid[0]][1] > lmlist[tipid[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # for each finger , y position of tip and mid is checked
        for id in range(1, 5):
            if lmlist[tipid[id]][2] < lmlist[tipid[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        if fingers[1] == 1 and fingers[2] == 0:
            # MOVING MODE
            # getting coordinates of tip of index finger
            cx, cy = lmlist[4][1], lmlist[4][2]
            # converting coordinates acc to screen size

            x3 = np.interp(cx, (frameR, wcam - frameR), (0, wscr))
            y3 = np.interp(cy, (frameR, hcam - frameR), (0, hscr))

            # smoothening
            clocx = plocx + (x3 - plocx) / smoothening
            clocy = plocy + (y3 - plocy) / smoothening

            # moving mouse to that position
            autopy.mouse.move(wscr - x3, y3)
            plocx, plocy = clocx, clocy

        if fingers[1] == 1 and fingers[2] == 1:
            # CLICKING MODE
            # code to find distance bw tip of index and middle finger
            indexcx, indexcy = lmlist[8][1], lmlist[8][2]
            middlecx, middlecy = lmlist[12][1], lmlist[12][2]

            midx = (indexcx + middlecx) // 2
            midy = (indexcy + middlecy) // 2

            length = ((((indexcx - middlecx) ** 2) + ((indexcy - middlecy) ** 2)) ** 0.5)
            cv2.line(frame, (indexcx, indexcy), (middlecx, middlecy),
                     (0, 255, 0), 5)
            cv2.circle(frame, (midx, midy), 10, (255, 0, 0), cv2.FILLED)
            if length <= 45:
                cv2.circle(frame, (midx, midy), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (midx, midy), 12, (0, 0, 0), 2)
                autopy.mouse.click()

    cv2.imshow("live", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
