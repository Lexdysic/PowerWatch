import cv2
import numpy as np
import time

# State
kStateInit = 0
kStateOut  = 1
kStateIn   = 2

# Key
kKeyDown   = 2621440
kKeyUp     = 2490368
kKeyRight  = 2555904
kKeyLeft   = 2424832
kKeySpace  = 32
kKeyEscape = 27

# Time
kSecondsPerMinute = 60
kMinutesPerHour = 60
kSecondsPerHour = kSecondsPerMinute * kMinutesPerHour

# Color
kRed   = (0, 0, 255)
kGreen = (0, 255, 0)
kBlue  = (255, 0, 0)

# Power Meter
kWattHoursPerRev = 7.2

# Config
kMinInsideTime = 1.0
kWindowed = True

def lerp(a, b, t):
    return cv2.addWeighted(a, 1-t, b, t, 0)

def overlap(aTL, aBR, bTL, bBR):
    if aTL[0] > bBR[0]:
        return False
    if aTL[1] > bBR[1]:
        return False
    if aBR[0] < bTL[0]:
        return False
    if aBR[1] < bTL[1]:
        return False
    return True

def runPowerWatch():
    if kWindowed:
        cv2.namedWindow("preview")
    cap = cv2.VideoCapture(0)

    state = kStateInit
    lastInsideTime = time.clock()
    lastKnownPower = None
    lastTriggerTime = None
    lastGray = None
    displayIndex = 0
    while cap.isOpened():
        success, frameBgr = cap.read()
        if not success:
            break

        # Process the frame to obtain contours
        currGray = cv2.cvtColor(frameBgr, cv2.COLOR_BGR2GRAY)
        currGray = cv2.GaussianBlur(currGray, (21, 21), 0)
        if lastGray is None:
            lastGray = currGray

        deltaGray    = cv2.absdiff(lastGray, currGray)
        threshGray   = cv2.threshold(deltaGray, 30, 255, cv2.THRESH_BINARY)[1]
        dialatedGray = cv2.dilate(threshGray, None, iterations=2)

        contours, _ = cv2.findContours(dialatedGray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lastGray = currGray


        # Set up the image that will end up being displayed to screen
        images      = [frameBgr, currGray, deltaGray, threshGray, dialatedGray]
        imageCount  = len(images)
        display     = images[displayIndex]
        showOverlay = display is frameBgr

        # compute trigger window
        screenH, screenW     = display.shape[0], display.shape[1]
        triggerW, triggerH   = screenW  * 1 // 3, screenH * 1 // 4
        triggerX0, triggerY0 = screenW * 1 // 3, screenH * 2 // 4
        triggerX1, triggerY1 = triggerX0 + triggerW, triggerY0 + triggerH
        triggerTL            = (triggerY0, triggerX0)
        triggerBR            = (triggerY1, triggerX1)


        # Determine if any of the contours overlap the trigger window
        triggered = False
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            contourX0, contourY0, contourW, contourH = cv2.boundingRect(contour)
            contourX1, contourY1 = contourX0 + contourW, contourY0 + contourH
            contourTL = (contourY0, contourX0)
            contourBR = (contourY1, contourX1)

            if showOverlay:
                cv2.rectangle(display, (contourX0, contourY0), (contourX1, contourY1), kGreen, 2)

            if not triggered and overlap(triggerTL, triggerBR, contourTL, contourBR):
                triggered = True
        if showOverlay:
            cv2.rectangle(display, (triggerX0, triggerY0), (triggerX1, triggerY1), kRed if triggered else kBlue, 2)


        # Process the state machine
        currTime = time.clock()
        if state == kStateInit:
            if triggered:
                state = kStateIn
        elif state == kStateIn:
            if not triggered and (currTime - lastInsideTime) > kMinInsideTime:
                state = kStateOut
        elif state == kStateOut:
            if triggered:
                state = kStateIn
                lastInsideTime = currTime

                currTriggerTime = currTime
                if lastTriggerTime:
                    secondsPerRev = currTriggerTime - lastTriggerTime
                    lastKnownPower = kWattHoursPerRev * kSecondsPerHour / secondsPerRev
                    print("Triggered Power = {}".format(lastKnownPower))
                lastTriggerTime = currTriggerTime


        # Print the current state
        if showOverlay:
            stateValue = "Init" if state == kStateInit else "In" if state == kStateIn else "Out"
            text = "State: {} | Power Usage: {} Watts".format(stateValue, lastKnownPower)
            cv2.putText(display, text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, kRed, 2)

        # Display the image
        if kWindowed:
            cv2.imshow("preview", display)

        # Process key press
        key = cv2.waitKey(20)
        if key == kKeyEscape:
            break
        elif key == kKeyUp:
            displayIndex = (displayIndex + imageCount - 1) % imageCount
        elif key == kKeyDown:
            displayIndex = (displayIndex + 1) % imageCount
        
    cap.release()
    if kWindowed:
        cv2.destroyWindow("preview")



runPowerWatch()