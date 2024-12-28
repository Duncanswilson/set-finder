############## Python-OpenCV Playing Card Detector ###############
#
# Author: Evan Juras
# Date: 9/5/17
# Description: Python script to detect and identify playing cards
# from a PiCamera video feed.
#

# Import necessary packages
import cv2
import numpy as np
import time
import os
import Cards
import VideoStream

def all_same_or_all_different(a, b, c):
    return (a == b == c) or (a != b and b != c and a != c)

def is_set(card1, card2, card3):
    return (all_same_or_all_different(card1.color, card2.color, card3.color) and
            all_same_or_all_different(card1.shape, card2.shape, card3.shape) and
            all_same_or_all_different(card1.shading, card2.shading, card3.shading) and
            all_same_or_all_different(card1.quantity, card2.quantity, card3.quantity))


### ---- INITIALIZATION ---- ###
# Define constants and initialize variables

## Define colors for different sets
SET_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Navy
]

## Camera settings
IM_WIDTH = 1280 #dont use these 
IM_HEIGHT = 720 #dont use these 
FRAME_RATE = 10

## Initialize calculated frame rate because it's calculated AFTER the first time it's displayed
frame_rate_calc = 1
freq = cv2.getTickFrequency()

## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera object and video feed from the camera. The video stream is set up
# as a seperate thread that constantly grabs frames from the camera feed. 
# See VideoStream.py for VideoStream class definition
## IF USING USB CAMERA INSTEAD OF PICAMERA,
## CHANGE THE THIRD ARGUMENT FROM 1 TO 2 IN THE FOLLOWING LINE:
videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE,2,1).start()
time.sleep(1) # Give the camera time to warm up

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
# train_ranks = Cards.load_ranks( path + '/Card_Imgs/')
# train_suits = Cards.load_suits( path + '/Card_Imgs/')


### ---- MAIN LOOP ---- ###
# The main loop repeatedly grabs frames from the video stream
# and processes them to find and identify playing cards.

cam_quit = 0 # Loop control variable

# Begin capturing frames
while cam_quit == 0:

    # Grab frame from video stream
    image = videostream.read()
    
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Pre-process camera image (gray, blur, and threshold it)
    pre_proc = Cards.preprocess_image(image)
	
    # Find and sort the contours of all cards in the image (query cards)
    cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

    # If there are no contours, do nothing
    if len(cnts_sort) != 0:
        # Initialize a new "cards" list to assign the card objects.
        # k indexes the newly made array of cards.
        cards = []
        k = 0
        # For each contour detected:
        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):
                cards.append(Cards.preprocess_card(cnts_sort[i],image))
                Cards.match_card(cards[k])
                # Draw center point and match result on the image.
                image = Cards.draw_results(image, cards[k])
                k = k + 1
        found_sets = []
        # Create a dictionary to track which sets each card belongs to
        card_sets = {i: [] for i in range(len(cards))}
        
        for i in range(len(cards)):
            for j in range(i+1, len(cards)):
                for k in range(j+1, len(cards)):
                    if is_set(cards[i], cards[j], cards[k]):
                        set_index = len(found_sets)
                        found_sets.append([cards[i].contour, cards[j].contour, cards[k].contour])
                        # Track which sets each card belongs to
                        card_sets[i].append(set_index)
                        card_sets[j].append(set_index)
                        card_sets[k].append(set_index)

        # Draw card contours and set numbers
        if (len(found_sets) != 0):
            # Draw each set with its unique color
            for i, set_cards in enumerate(found_sets):
                color = SET_COLORS[i % len(SET_COLORS)]
                cv2.drawContours(image, set_cards, -1, color, 2)
            
            # Add set numbers for each card
            for card_idx, set_numbers in card_sets.items():
                if set_numbers:  # Only process cards that are part of sets
                    card = cards[card_idx]
                    # Calculate text position near the card's center
                    text = f"Sets: {','.join(str(n+1) for n in set_numbers)}"
                    # Get the center point of the card
                    M = cv2.moments(card.contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        # Draw white background for better visibility
                        text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
                        cv2.rectangle(image, 
                                    (cx - text_size[0]//2 - 5, cy - text_size[1]//2 - 5),
                                    (cx + text_size[0]//2 + 5, cy + text_size[1]//2 + 5),
                                    (255, 255, 255), -1)
                        # Draw text
                        cv2.putText(image, text, (cx - text_size[0]//2, cy + text_size[1]//2),
                                  font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw framerate in the corner of the image. Framerate is calculated at the end of the main loop,
        # so the first time this runs, framerate will be shown as 0.
        cv2.putText(image,"FPS: "+str(int(frame_rate_calc)),(10,26),font,0.7,(255,0,255),2,cv2.LINE_AA)

        # Finally, display the image with the identified cards!
        cv2.imshow("Card Detector",image)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1
        
        # Poll the keyboard. If 'q' is pressed, exit the main loop.
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cam_quit = 1
        

# Close all windows and close the PiCamera video stream.
cv2.destroyAllWindows()
videostream.stop()

