############## Playing Card Detector Functions ###############
#
# Author: Evan Juras
# Date: 9/5/17
# Description: Functions and classes for CardDetector.py that perform 
# various steps of the card detection algorithm


# Import necessary packages
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

### Constants ###

# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000

font = cv2.FONT_HERSHEY_SIMPLEX

### Structures to hold card and train card information ###
class Set_card:
    """Structure to store information about query cards in the camera image."""
    def __init__(self):
        self.contour = [] # Contour of card
        self.width, self.height = 0, 0 # Width and height of card
        self.corner_pts = [] # Corner points of card
        self.center = [] # Center point of card
        self.warp = [] # 200x300, flattened, grayed, blurred image
        self.color = "Unknown"
        self.roi = None
        self.quantity = "Unknown"
        self.shading = "Unknown"
        self.shape = "Unknown"

def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    # _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return thresh

def find_cards(thresh_image):
    """Finds all card-sized contours in a thresholded camera image.
    Returns the number of cards, and a list of card contours sorted
    from largest to smallest."""

    # Find contours and sort their indices by contour size
    cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print(len(cnts))
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)

    # Create debug images - convert threshold image to BGR so we can draw colored contours
    debug_image = cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2BGR)
    
    # Draw all contours in blue
    cv2.drawContours(debug_image, cnts, -1, (255,0,0), 2)
    cv2.imshow("All Contours", debug_image)
    
    # Create image for size-filtered contours
    size_filtered = debug_image.copy()
    
    # Rest of the existing code...
    if len(cnts) == 0:
        return [], []
    
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)

    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    # Determine which of the contours are cards by applying the
    # following criteria: 1) Smaller area than the maximum card size,
    # 2), bigger area than the minimum card size, 3) have no parents,
    # and 4) have four corners

    # Define card aspect ratio constants
    CARD_ASPECT_RATIO = 1.56  # height/width for a Set card
    ASPECT_RATIO_TOLERANCE = 0.3  # Allow ±30% variation

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        
        # Draw contours that pass size filter in green
        if (size < CARD_MAX_AREA) and (size > CARD_MIN_AREA):
            # Get bounding rectangle to check aspect ratio
            x, y, w, h = cv2.boundingRect(cnts_sort[i])
            aspect_ratio = h / w
            
            # Check if aspect ratio matches either vertical or horizontal orientation
            vertical_ok = abs(aspect_ratio - CARD_ASPECT_RATIO) < (CARD_ASPECT_RATIO * ASPECT_RATIO_TOLERANCE)
            horizontal_ok = abs(aspect_ratio - (1/CARD_ASPECT_RATIO)) < ((1/CARD_ASPECT_RATIO) * ASPECT_RATIO_TOLERANCE)
            aspect_ratio_ok = vertical_ok or horizontal_ok
            
            if aspect_ratio_ok:
                cv2.drawContours(size_filtered, [cnts_sort[i]], -1, (0,255,0), 2)
                # Add aspect ratio text with orientation
                orientation = "V" if vertical_ok else "H"
                cv2.putText(size_filtered, f"AR: {aspect_ratio:.2f} {orientation}", 
                          (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, (0,255,0), 2)
                
                peri = cv2.arcLength(cnts_sort[i],True)
                approx = cv2.approxPolyDP(cnts_sort[i],0.02*peri,True)
                
                # If contour has 4 corners, draw it in red
                if len(approx) == 4:
                    cv2.drawContours(size_filtered, [cnts_sort[i]], -1, (0,0,255), 2)
                    
                    # Convert points to more usable format
                    pts = approx.reshape(4, 2)
                    
                    # Calculate and draw angles
                    angles = []  # Create list to store angles
                    for j in range(4):
                        pt1 = pts[j]
                        pt2 = pts[(j+1)%4]
                        pt3 = pts[(j+2)%4]
                        
                        # Calculate vectors and angle
                        v1 = pt1 - pt2
                        v2 = pt3 - pt2
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        angle = np.abs(np.arccos(cos_angle) * 180 / np.pi)
                        angles.append(angle)  # Add angle to list
                        
                        # Draw angle text at each corner
                        cv2.putText(size_filtered, f"{angle:.1f}", 
                                  tuple(pt2.astype(int)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.6, (255,255,255), 2)
                    
                    # Check if angles are approximately 90 degrees
                    angles_ok = all(abs(angle - 90) < 20 for angle in angles)
                    
                    if angles_ok:
                        cnt_is_card[i] = 1
            else:
                # Draw rejected contours in yellow with their aspect ratio
                cv2.drawContours(size_filtered, [cnts_sort[i]], -1, (0,255,255), 2)
                cv2.putText(size_filtered, f"Bad AR: {aspect_ratio:.2f}", 
                          (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, (0,255,255), 2)

    cv2.imshow("Filtered Contours", size_filtered)
    return cnts_sort, cnt_is_card

def preprocess_card(contour, image):
    """Uses contour to find information about the query card. Isolates rank
    and suit images from the card."""

    # Initialize new Query_card object
    sCard = Set_card()

    sCard.contour = contour

    # Find perimeter of card and use it to approximate corner points
    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*peri,True)
    pts = np.float32(approx)
    sCard.corner_pts = pts

    # Find width and height of card's bounding rectangle
    x,y,w,h = cv2.boundingRect(contour)
    sCard.width, sCard.height = w, h

    # Find center point of card by taking x and y average of the four corners.
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    sCard.center = [cent_x, cent_y]

    # Warp card into 200x300 flattened image using perspective transform
    sCard.warp = flattener(image, pts, w, h)

    #create the region of interest for the card (might want to use the warp instead)
    sCard.roi =  image[y:y+h, x:x+w].copy()

    return sCard

def match_card(sCard):
    sCard.color = detect_color(sCard.warp)
    sCard.shape, sCard.quantity, sCard.shading = detect_features(sCard.warp)

    
def draw_results(image, sCard):
    """Draw the card name, center point, and contour on the camera image."""

    x = sCard.center[0]
    y = sCard.center[1]
    cv2.circle(image,(x,y),5,(255,0,0),-1)

    # rank_name = qCard.best_rank_match
    # suit_name = qCard.best_suit_match
    color = sCard.color
    quantity = sCard.quantity
    shading = sCard.shading
    shape = sCard.shape

    # Draw card name twice, so letters have black outline
    cv2.putText(image,quantity,(x-60,y-50),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,quantity,(x-60,y-50),font,1,(50,200,200),2,cv2.LINE_AA)    

    cv2.putText(image,color,(x-60,y-25),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,color,(x-60,y-25),font,1,(50,200,200),2,cv2.LINE_AA)

    cv2.putText(image,shading,(x-60,y+25),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,shading,(x-60,y+25),font,1,(50,200,200),2,cv2.LINE_AA)
    
    cv2.putText(image,shape,(x-60,y+50),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,shape,(x-60,y+50),font,1,(50,200,200),2,cv2.LINE_AA)    

    return image

def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.
    
    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
            
    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warp


def detect_color(card_roi):
    # Convert to HSV and get average color values
    hsv = cv2.cvtColor(card_roi, cv2.COLOR_BGR2HSV)
    
    # Create a mask to focus on the symbols/shapes
    gray = cv2.cvtColor(card_roi, cv2.COLOR_BGR2GRAY)
    _, symbol_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    symbol_mask = cv2.morphologyEx(symbol_mask, cv2.MORPH_CLOSE, kernel)
    
    # Define HSV ranges for each color - adjusted for better purple detection
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])  # Shifted from 160 to 170 to reduce purple confusion
    red_upper2 = np.array([180, 255, 255])
    
    green_lower = np.array([35, 50, 50])
    green_upper = np.array([85, 255, 255])
    
    # Widened purple range and increased saturation threshold
    purple_lower = np.array([125, 30, 50])  # Lowered from 130 to 125
    purple_upper = np.array([165, 255, 255])  # Increased from 155 to 165
    
    # Create masks for each color
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
    
    # Apply symbol mask to color masks
    red_mask = cv2.bitwise_and(red_mask, symbol_mask)
    green_mask = cv2.bitwise_and(green_mask, symbol_mask)
    purple_mask = cv2.bitwise_and(purple_mask, symbol_mask)
    
    # Calculate weighted color scores
    symbol_pixels = np.sum(symbol_mask > 0)
    if symbol_pixels == 0:
        return 'unknown'
    
    # Get mean HSV values for the symbol areas
    symbol_hsv = cv2.bitwise_and(hsv, hsv, mask=symbol_mask)
    mean_hsv = cv2.mean(symbol_hsv, mask=symbol_mask)[:3]
    
    # Calculate color scores with weights
    red_score = np.sum(red_mask > 0) / symbol_pixels
    green_score = np.sum(green_mask > 0) / symbol_pixels
    purple_score = np.sum(purple_mask > 0) / symbol_pixels
    
    # Add hue-based weighting for purple vs red disambiguation
    hue = mean_hsv[0]
    if 125 <= hue <= 165:  # If hue is in purple range
        purple_score *= 1.2  # Boost purple score
    
    print("\n=== Color Detection ===")
    print(f"Mean HSV: {mean_hsv}")
    print(f"Red score: {red_score:.3f}")
    print(f"Green score: {green_score:.3f}")
    print(f"Purple score: {purple_score:.3f}")
    
    # Determine color based on highest score with minimum threshold
    min_score_threshold = 0.15
    max_score = max(red_score, green_score, purple_score)
    
    # if max_score < min_score_threshold:
    #     print("Warning: Color scores below threshold")
    #     detected_color = 'unknown'
    if red_score > green_score and red_score > purple_score:
        detected_color = 'red'
    elif green_score > red_score and green_score > purple_score:
        detected_color = 'green'
    else:
        detected_color = 'purple'
        
    print(f"Detected color: {detected_color}")
    print("==================\n")
    
    return detected_color


def detect_features(card_warp):
    print("\n=== Detecting Features for Card ===")
    
    # Convert to grayscale if not already
    if len(card_warp.shape) == 3:
        gray = cv2.cvtColor(card_warp, cv2.COLOR_BGR2GRAY)
    else:
        gray = card_warp.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding instead of simple threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Clean up the threshold image
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert contours to relative coordinates
    height, width = card_warp.shape[:2]
    card_area = height * width
    
    # Store valid contours and their properties
    valid_contours = []
    contour_properties = []
    relative_contours = []
    
    # Adjust area thresholds for warped image
    MIN_RELATIVE_AREA = 0.03  # Decreased from 0.05
    MAX_RELATIVE_AREA = 0.35  # Increased from 0.3
    
    # First pass: gather properties of all potential symbol contours
    areas = []
    for c in contours:
        area = cv2.contourArea(c)
        relative_area = area / card_area
        if MIN_RELATIVE_AREA < relative_area < MAX_RELATIVE_AREA:
            areas.append(area)
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            rect = cv2.minAreaRect(c)
            aspect_ratio = min(rect[1]) / max(rect[1]) if max(rect[1]) > 0 else 0
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            valid_contours.append(c)
            contour_properties.append({
                'area': area,
                'relative_area': relative_area,
                'circularity': circularity,
                'aspect_ratio': aspect_ratio,
                'vertices': len(approx),
                'solidity': solidity,
                'perimeter': perimeter
            })
    
    # If we found potential symbols, calculate mean area for filtering
    if areas:
        mean_area = np.mean(areas)
        area_std = np.std(areas)
    
    # Second pass: filter and classify shapes using statistical properties
    symbol_count = 0
    shapes = []
    final_contours = []
    
    for i, (c, props) in enumerate(zip(valid_contours, contour_properties)):
        # Additional filtering based on area consistency
        if areas and abs(props['area'] - mean_area) > area_std * 1.5:
            print(f"Rejecting contour {i}: Area too different from mean")
            continue
            
        # Filter based on aspect ratio - symbols should be roughly proportional
        if props['aspect_ratio'] < 0.3:
            print(f"Rejecting contour {i}: Aspect ratio too extreme")
            continue
        
        # Improved shape classification
        if props['circularity'] > 0.65 and props['solidity'] > 0.9:
            shapes.append('oval')
            print("  - Classified as oval")
        elif props['vertices'] == 4 and props['solidity'] > 0.9:  # Diamonds need 4 vertices and high solidity
            shapes.append('diamond')
            print("  - Classified as diamond")
        elif props['solidity'] > 0.8:  # Back to using solidity for squiggles
            shapes.append('squiggle')
            print("  - Classified as squiggle")
        else:
            print("  - Rejected: Does not match any shape criteria")
            continue
        
        symbol_count += 1
        final_contours.append(c)
        
        # Add to relative contours
        relative_contour = (c.reshape(-1, 2).astype(float) / [width, height]).tolist()
        relative_contours.append(relative_contour)
    
    detected_shape = shapes[0] if shapes else 'unknown'
    detected_quantity = str(symbol_count)

    # Use only the final filtered contours for shading detection
    symbol_mask = np.zeros_like(thresh)
    if final_contours:  # Check if list is not empty before accessing first element
        cv2.drawContours(symbol_mask, [final_contours[0]], -1, 255, -1)
    # Convert card_warp to grayscale for symbol content analysis
    if len(card_warp.shape) == 3:
        gray_warp = cv2.cvtColor(card_warp, cv2.COLOR_BGR2GRAY)
    else:
        gray_warp = card_warp.copy()
    
    # First create an outline mask to identify the symbol regions
    _, outline_mask = cv2.threshold(gray_warp, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5), np.uint8)
    outline_mask = cv2.morphologyEx(outline_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours to get symbol regions
    contours, _ = cv2.findContours(outline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    symbol_regions = np.zeros_like(outline_mask)
    for c in contours:
        area = cv2.contourArea(c)
        if area > (card_warp.shape[0] * card_warp.shape[1] * 0.01):
            cv2.drawContours(symbol_regions, [c], -1, 255, -1)
    
    # Try adaptive thresholding to better capture thin stripes
    stripe_mask = cv2.adaptiveThreshold(
        gray_warp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size - adjust this if needed
        2    # C constant - adjust this if needed
    )    
    # Combine the masks - only keep stripe details within symbol regions
    symbol_mask = cv2.bitwise_and(stripe_mask, stripe_mask, mask=symbol_regions)

    # Calculate fill and striped ratios
    symbol_pixels = np.sum(symbol_regions > 0)  # Total area of symbols
    _, filled_mask = cv2.threshold(gray_warp, 128, 255, cv2.THRESH_BINARY_INV)
    filled_mask = cv2.bitwise_and(filled_mask, symbol_regions)
    white_pixels = np.sum(filled_mask == 255)
    solid_fill_ratio = white_pixels / symbol_pixels if symbol_pixels > 0 else 0
    white_pixels = np.sum(symbol_mask == 255)
    striped_fill_ratio = white_pixels / symbol_pixels if symbol_pixels > 0 else 0
    
    # Adjust thresholds for different shading types
    THRESHOLD = 0.70
    print(f"Solid fill ratio: {solid_fill_ratio:.3f}")
    print(f"Striped fill ratio: {striped_fill_ratio:.3f}")

    # Classification logic
    if solid_fill_ratio > THRESHOLD:
        shading = 'solid'
    else:
        # For fill ratios between thresholds, look for stripe patterns
        if striped_fill_ratio < THRESHOLD:
            shading = 'striped'
        else:
            # If we're not confident about stripes, use fill ratio
            shading = 'open'

    print(f"Detected shading: {shading}")

    return detected_shape, detected_quantity, shading
