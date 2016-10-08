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
kMinContourArea = 500
kThreshold = 30
kBlurSize = 25

class Rect():
    def __init__(self, x, y, w, h):
        self.x0 = x
        self.y0 = y
        self.x1 = self.x0 + w
        self.y1 = self.y0 + h


    def overlap(a, b):
        if a.y0 > b.y1:
            return False
        if a.x0 > b.x1:
            return False
        if a.y1 < b.y0:
            return False
        if a.x1 < b.x0:
            return False
        return True


class ImageProcessor():
    def __init__(self):
        self.lastBlurGray   = None
        self.frameBgr       = None
        self.frameGray      = None
        self.blurGray       = None
        self.deltaGray      = None
        self.thresholdGray  = None
        self.dialatedGray   = None

    def Update(self, frameBgr):
        self.frameBgr  = frameBgr
        self.frameGray = cv2.cvtColor(self.frameBgr, cv2.COLOR_BGR2GRAY)
        self.blurGray  = cv2.GaussianBlur(self.frameGray, (kBlurSize, kBlurSize), 0)
        if self.lastBlurGray is None:
            self.lastBlurGray = self.blurGray

        self.deltaGray     = cv2.absdiff(self.lastBlurGray, self.blurGray)
        self.thresholdGray = cv2.threshold(self.deltaGray, kThreshold, 255, cv2.THRESH_BINARY)[1]
        self.dialatedGray  = cv2.dilate(self.thresholdGray, None, iterations=2)

        contours, _ = cv2.findContours(self.dialatedGray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.lastBlurGray = self.blurGray

        return contours

class Display():
    def __init__(self, windowName = None):
        self.windowName = windowName
        self.image      = None
        self.hasOverlay = False
        if self.windowName:
            cv2.namedWindow(self.windowName)

    def __del__(self):
        if self.windowName:
            cv2.destroyWindow(self.windowName)

    def update(self, image, hasOverlay):
        self.image      = image
        self.hasOverlay = hasOverlay

    def rectangle(self, rect, color):
        if self.hasOverlay:
            cv2.rectangle(self.image, (rect.x0, rect.y0), (rect.x1, rect.y1), color, 2)

    def text(self, text, pos, color):
        if self.hasOverlay:
            cv2.putText(self.image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def present(self):
        if self.windowName:
            cv2.imshow(self.windowName, self.image)

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]



def runPowerWatch():
    display = Display("Power Watch" if kWindowed else None)
    caputre = cv2.VideoCapture(0)

    state = kStateInit
    lastInsideTime = time.clock()
    lastKnownPower = 0
    lastTriggerTime = None
    displayIndex = 0

    processor = ImageProcessor();
    while caputre.isOpened():
        success, frameBgr = caputre.read()
        if not success:
            break

        contours = processor.Update(frameBgr)

        # Set up the image that will end up being displayed to screen
        images      = [processor.frameBgr, processor.frameGray, processor.blurGray, processor.deltaGray, processor.thresholdGray, processor.dialatedGray]
        imageName   = ["Source", "Grayscale", "Blurred", "Delta", "Threshold", "Dialated"]
        imageCount  = len(images)

        display.update(images[displayIndex], displayIndex == 0)


        # compute trigger window
        triggerRect = Rect(
            display.width  * 1 // 3, # x
            display.height * 2 // 4, # y
            display.width  * 1 // 3, # w
            display.height * 1 // 4  # h
        )


        # Determine if any of the contours overlap the trigger window
        triggered = False
        for contour in contours:
            if cv2.contourArea(contour) < kMinContourArea:
                continue
            contourRect = Rect(*cv2.boundingRect(contour))

            display.rectangle(contourRect, kGreen)

            if Rect.overlap(triggerRect, contourRect):
                triggered = True

        display.rectangle(triggerRect, kRed if triggered else kBlue)

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
        stateValue = "Init" if state == kStateInit else "In" if state == kStateIn else "Out"
        text = "State: {} | Power Usage: {} Watts".format(stateValue, lastKnownPower)
        display.text(text, (15, 15), kRed)

        # Display the image
        display.present()

        # Process key press
        key = cv2.waitKey(20)
        if key == kKeyEscape:
            break
        elif key == kKeyUp:
            displayIndex = (displayIndex + imageCount - 1) % imageCount
        elif key == kKeyDown:
            displayIndex = (displayIndex + 1) % imageCount
        
    caputre.release()

runPowerWatch()