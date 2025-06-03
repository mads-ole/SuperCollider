#ziel: in einem videobild soll die helligkeit von zuvor bestimmten punkten getracked werden
#auf basis dieser helligkeit soll dann eine OSC nachricht an supercollider gesendet werden, je nachdem 
#welcher punkt, wie hell bzw. dunkel ist (hell: 0, dunkel: 1)

import cv2
import numpy as np

from osc4py3.as_eventloop import *
from osc4py3 import oscbuildparse
from osc4py3 import oscmethod as osm


#video von webcam abgreifen (0, 1 or 2)
#wenn zwei webcams angeschlossen muss vid2 "unauskommentiert" werden
vid1 = cv2.VideoCapture(0)
#vid2 = cv2.VideoCapture(1)

#erzeuge leeres dictionary mit wer für jeden getrackten bildausschnitt
#später wird darin für jeden ausschnitt die zeit der abdunklung gezählt
def createDictionary(trackedFrames):
    counter = {}

    for i in range(trackedFrames):
        counter[i] = 0

    return counter

#webcam bild wird eingelesen und so auf frame basis nutzbar gemacht
#auperdem werden die Koordinaten festgelegt, die bestimmen, wo sich die kleinen, getrackten bildausschnitte befinden
def getInformation(vid):
    ret, frame = vid.read()
    h, w = frame.shape[:2]

    w1 = int(((w-h)/2)+h * 1/10)
    w2 = int(((w-h)/2)+h * 3/10)
    w3 = int(((w-h)/2)+h * 5/10)
    w4 = int(((w-h)/2)+h * 7/10)
    w5 = int(((w-h)/2)+h * 9/10)
    #((w-h)/2)+ = Feld in die Mitte verschieben

    h1 = int(h * 1/10)
    h2 = int(h * 3/10)
    h3 = int(h * 5/10)
    h4 = int(h * 7/10)
    h5 = int(h * 9/10)
    
    heights = (h1, h2, h3, h4, h5)

    # coordinates = ((w1,h1), (w2, h2))
    x = 0
    coordinates = []
    for i in heights:
        height = heights[x]
        coordinates.append((w1, height))
        coordinates.append((w2, height))
        coordinates.append((w3, height))
        coordinates.append((w4, height))
        coordinates.append((w5, height))

        x += 1

    return frame, coordinates

#ein kreis wird jeweils da in das video bild eingefügt, wo sich die zu untersuchenden bildausschnitte auch befinden
def drawCircle(frame, coordinates):
    for index in coordinates:
        
        circle = cv2.circle(frame, index, 25, (0,0,255), 2)
    return circle

#mehrere kleine bildausschnitte werden (entsprechend den koordinaten) aus dem webcam bild ausgeschnitten
#in jedem for loop einer und diese werden dann zusammen in eine liste getan
#--> diese liste, mit allen ausschnitten drin, wird weitergegeben und für jeden frame des videos
#wird die ganze liste einmal durch die ganzen folgenden funktionen geschleust
#… außerdem wird ein bild ausgegeben, welches  veranschaulicht, wo sich die zu berabeitenden bildausschnitte
#befinden …
def cropImage(image, coordinates):

    cropList = [] 
    for index in coordinates:  
        x = index[0]
        y = index[1]
        x,y,w,h = (x-25), (y-25), 50, 50

        crop = image[y:y+h, x:x+w]
        gain = 1.5
        crop = (gain * crop.astype(np.float64)).clip(0,255).astype(np.uint8)

        cropList.append(crop)
        
        #füge ausschnitt zu bild hinzu
        result = image.copy()
        result[y:y+h, x:x+h] = crop
        image = result     

    return cropList, image

#konvertierung zu hsv und daraufhin ermittlung der helligkeit der einzelnen bildausschnitte
def rgb2hsv(bgr_image_list):
    i = 0
    valueList = []
    for index in bgr_image_list:
        bgr_image = bgr_image_list[i]
        
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        value = hsv[...,2].mean()
        
        valueList.append(round(value, 3))
        i += 1
    return valueList 

#helligkeit eines jeden bildausschnittes wird in 'hell' oder 'dunkel' eingeordnet
#je nachdem ob sich der wert über oder unter dem threshold befindet
#(zusammenführung mit nächster funktion wäre effizienter - aber praktisch zu debuggen)
def threshold(brightnessValue):
    #count gibt an um welchen gecroppten frame es sich handelt
    count = 0
    classification = []
    for item in brightnessValue:
        
        threshold = 110
        if item >= threshold:
            classification.append((count, ': hell !'))
            count += 1   
        else: 
            classification.append((count, ': dunkel !'))
            count += 1

    return classification


#wenn ein feld dunkel ist, zählt der entsprechende index des counterDict hoch
#wenn es hell ist, wird der entsprechende index auf null zurückgesetzt
def timeOfDarkness(counterDict, brightnessClassification):

    for item in brightnessClassification:
        #index ist bei dictionarys der key
        i = item[0]
        count = counterDict[i]

        if item[1] == ': dunkel !':
            count += 1
            counterDict[i] = count 

        elif item[1] == ': hell !':
            #count -= 1
            counterDict[i] = 0

        else:
            pass

    return counterDict

#wenn value von einem der frames über bestimmter anzahl (10) ist, wird enstprechender listenindex auf 1 gesetzt
#1 heißt: dises feld wird als permanent abgeduckelt betrachtet
def triggerWhenDark(counterDict):
    oscList = []
    for i in range(25):
        oscList.append(0)

    #key = feld in der matrix
    for key in counterDict:
        count = counterDict[key]

        if count > 10: #buffer zwischen abdunkeln und registrieren dessen
            
            oscList[key] = 1
        elif count == 0:
            oscList[key] = 0
        else:
            pass

    return oscList

#sendet OSC nachricht
def oscSender(oscList):

    # Build a message and send it.
    msg = oscbuildparse.OSCMessage("/test1", ",iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii", oscList)
    osc_send(msg, "aclientname")

#handlerfunction für OSC
text = "h"
def handlerfunction(*args):

    print(args[0])
    
    global text
    text = args[0]

#zusammenführen beider webcam bilder für die ausgabe
#dabei werden die bilder auf die dimension der matrix beschränkt
def imageOutput(result1, result2):
        
    #matrixen ausschneiden und bilder zusammenführen
    h, w = result1.shape[:2]
    x = int(((w-h)/2)+h * 1/10) - 125
    w1 = int(((w-h)/2)+h * 9/10) - int(((w-h)/2)+h * 1/10) + 250
    y = 0
    h1 = h 

    result_cropped = result1[y:y+h1, x:x+w1]
    result2_cropped = result2[y:y+h1, x:x+w1]
    both = cv2.hconcat([result_cropped, result2_cropped])

    #größe anpassen
    both_resized = cv2.resize(both, (1920, 1080), interpolation=cv2.INTER_AREA)

    return both_resized



#funktionen werden für angestrebten ablauf zusammengefügt
#mit 'q' kann das ganze beendet werden
def main(vid1, vid2):
    grid = 5 * 5 
    counter1 = createDictionary(grid)
    counter2 = createDictionary(grid)
    

    count = 0
    #start osc system
    osc_startup()

    # Make client channels to send packets. (OSC out)
    osc_udp_client("127.0.0.1", 57120, "aclientname")
    
    while True:
        # ===== webcam nr. 1 ===== #   
        frame1, coordinates1 = getInformation(vid1)   
        circle1 = drawCircle(frame1, coordinates1)
        cropList1, result1 = cropImage(frame1, coordinates1)
        brightnessValue1 = rgb2hsv(cropList1)
       
        brightnessClassification1 = threshold(brightnessValue1) 
        counterDict1 = timeOfDarkness(counter1, brightnessClassification1)  
        oscList1 = triggerWhenDark(counterDict1)


        # ===== webcam nr.2 ===== #        
        frame2, coordinates2 = getInformation(vid2)   
        circle2 = drawCircle(frame2, coordinates2)
        cropList2, result2 = cropImage(frame2, coordinates2)
        brightnessValue2 = rgb2hsv(cropList2)

        brightnessClassification2 = threshold(brightnessValue2)
        counterDict2 = timeOfDarkness(counter2, brightnessClassification2) 
        oscList2 = triggerWhenDark(counterDict2)


        # ===== OSC out ===== #
        oscFinal = oscList1 + oscList2
    
        osc_process()
        #frequenz an osc sendungen verringern (%30 ≈ 1sek.)
        if (count%5) == 4:
            oscSender(oscFinal)
        count += 1


        #video output
        both_resized = imageOutput(result1, result2)
        
        cv2.imshow('Frame2', both_resized)


        #cv2.imshow('video1', result) #display changed and 'tracked' cam
        #cv2.imshow('video2', result2) #display changed and 'tracked' cam2 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            osc_terminate()
            break

    vid1.release()
    vid2.release()
    #oscSender([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    #           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    cv2.destroyAllWindows()


main(vid1, vid1)
