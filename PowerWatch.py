import cv2
import numpy as np


RED   = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE  = (255, 0, 0)

def lerp(a, b, t):
    return cv2.addWeighted(a, 1-t, b, t, 0)

def overlap(aTL, aBR, bTL, bBR):
    if aTL[0] < bBR[0]:
        return False
    if aTL[1] < bBR[1]:
        return False
    if aBR[0] < bTL[0]:
        return False
    if aBR[1] < bTL[1]:
        return False
    return True

def runPowerWatch():
    cv2.namedWindow("preview")
    cap = cv2.VideoCapture(0)

    lastGray = None
    while cap.isOpened():
        success, frameBgr = cap.read()
        if success:
            currGray = cv2.cvtColor(frameBgr, cv2.COLOR_BGR2GRAY)
            currGray = cv2.GaussianBlur(currGray, (21, 21), 0)
            if lastGray is None:
                lastGray = currGray

            deltaGray = cv2.absdiff(lastGray, currGray)
            thresh = cv2.threshold(deltaGray, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            lastGray = currGray

            display = frameBgr

            # compute trigger window
            screenH, screenW, _ = display.shape
            triggerX0, triggerY0 = screenW // 4, screenH * 2 // 4
            triggerX1, triggerY1 = triggerX0 + screenW // 2, triggerY0 + screenH // 4

            triggerBL = (triggerY0, triggerX0)
            triggerTR = (triggerY1, triggerX1)


            triggered = False

            # loop over the contours
            for c in cnts:
                if cv2.contourArea(c) < 500:
                    continue
                (contourX0, contourY0, contourW, contourH) = cv2.boundingRect(c)
                contourX1, contourY1 = contourX0 + contourW, contourY0 + contourH
                contourBL = (contourY0, contourX0)
                contourTR = (contourY1, contourX1)
                cv2.rectangle(display, (contourX0, contourY0), (contourX1, contourY1), GREEN, 2)

                if not triggered and overlap(triggerTR, triggerBL, contourTR, contourBL):
                    triggered = True

            cv2.rectangle(display, (triggerX0, triggerY0), (triggerX1, triggerY1), RED if triggered else BLUE, 2)

            cv2.imshow("preview", display)

            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break
        else:
            break
    cap.release()
    cv2.destroyWindow("preview")



runPowerWatch()