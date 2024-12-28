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

### Structures to hold query card and train card information ###

class Query_card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.contour = [] # Contour of card
        self.width, self.height = 0, 0 # Width and height of card
        self.corner_pts = [] # Corner points of card
        self.center = [] # Center point of card
        self.warp = [] # 200x300, flattened, grayed, blurred image
        self.rank_img = [] # Thresholded, sized image of card's rank
        self.suit_img = [] # Thresholded, sized image of card's suit
        self.best_rank_match = "Unknown" # Best matched rank
        self.best_suit_match = "Unknown" # Best matched suit
        self.rank_diff = 0 # Difference between rank image and best matched train rank image
        self.suit_diff = 0 # Difference between suit image and best matched train suit image

class Set_card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.contour = [] # Contour of card
        self.width, self.height = 0, 0 # Width and height of card
        self.corner_pts = [] # Corner points of card
        self.center = [] # Center point of card
        self.warp = [] # 200x300, flattened, grayed, blurred image
        # self.rank_img = [] # Thresholded, sized image of card's rank
        # self.suit_img = [] # Thresholded, sized image of card's suit
        self.best_rank_match = "Unknown" # Best matched rank
        self.best_suit_match = "Unknown" # Best matched suit
        self.rank_diff = 0 # Difference between rank image and best matched train rank image
        self.suit_diff = 0 # Difference between suit image and best matched train suit image
        self.color = "Unknown"
        self.roi = None
        self.quantity = "Unknown"
        self.shading = "Unknown"
        self.shape = "Unknown"

    def debug_shading(self, shape_img):
        """Debug visualization for shading detection"""
        hsv = cv2.cvtColor(shape_img, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        
        # Create masks
        gray = cv2.cvtColor(shape_img, cv2.COLOR_BGR2GRAY)
        _, dark_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        _, light_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        symbol_mask = cv2.bitwise_or(dark_mask, light_mask)
        
        # Get masked hue values
        masked_hue = cv2.bitwise_and(h_channel, h_channel, mask=symbol_mask)
        
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(141)
        plt.imshow(cv2.cvtColor(shape_img, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        
        # Mask
        plt.subplot(142)
        plt.imshow(symbol_mask, cmap='gray')
        plt.title('Symbol Mask')
        
        # Masked hue
        plt.subplot(143)
        plt.imshow(masked_hue, cmap='hsv')
        plt.title('Masked Hue')
        
        # Histogram
        plt.subplot(144)
        hue_values = masked_hue[symbol_mask > 0]
        if len(hue_values) > 0:
            plt.hist(hue_values, bins=30)
            plt.title(f'Hue Histogram\nVariance: {np.var(hue_values):.2f}')
        
        plt.tight_layout()
        plt.show()

def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # The best threshold level depends on the ambient lighting conditions.
    # For bright lighting, a high threshold must be used to isolate the cards
    # from the background. For dim lighting, a low threshold must be used.
    # To make the card detector independent of lighting conditions, the
    # following adaptive threshold method is used.
    #
    # A background pixel in the center top of the image is sampled to determine
    # its intensity. The adaptive threshold is set at 50 (THRESH_ADDER) higher
    # than that. This allows the threshold to adapt to the lighting conditions.
    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg_level + BKG_THRESH

    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    # retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)

    return thresh

def find_cards(thresh_image):
    """Finds all card-sized contours in a thresholded camera image.
    Returns the number of cards, and a list of card contours sorted
    from largest to smallest."""

    # Find contours and sort their indices by contour size
    cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)

    # If there are no contours, do nothing
    if len(cnts) == 0:
        return [], []
    
    # Otherwise, initialize empty sorted contour and hierarchy lists
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)

    # Fill empty lists with sorted contour and sorted hierarchy. Now,
    # the indices of the contour list still correspond with those of
    # the hierarchy list. The hierarchy array can be used to check if
    # the contours have parents or not.
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    # Determine which of the contours are cards by applying the
    # following criteria: 1) Smaller area than the maximum card size,
    # 2), bigger area than the minimum card size, 3) have no parents,
    # and 4) have four corners

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)
        
        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card

def preprocess_card(contour, image):
    """Uses contour to find information about the query card. Isolates rank
    and suit images from the card."""

    # Initialize new Query_card object
    qCard = Set_card()

    qCard.contour = contour

    # Find perimeter of card and use it to approximate corner points
    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*peri,True)
    pts = np.float32(approx)
    qCard.corner_pts = pts

    # Find width and height of card's bounding rectangle
    x,y,w,h = cv2.boundingRect(contour)
    qCard.width, qCard.height = w, h

    # Find center point of card by taking x and y average of the four corners.
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    # Warp card into 200x300 flattened image using perspective transform
    qCard.warp = flattener(image, pts, w, h)

    #create the region of interest for the card (might want to use the warp instead)
    qCard.roi =  image[y:y+h, x:x+w].copy()

    return qCard

def match_card(qCard):
    qCard.color = detect_color(qCard.warp)
    qCard.shape, qCard.quantity, qCard.shading = detect_features(qCard.warp)

    
def draw_results(image, qCard):
    """Draw the card name, center point, and contour on the camera image."""

    x = qCard.center[0]
    y = qCard.center[1]
    cv2.circle(image,(x,y),5,(255,0,0),-1)

    # rank_name = qCard.best_rank_match
    # suit_name = qCard.best_suit_match
    color = qCard.color
    quantity = qCard.quantity
    shading = qCard.shading
    shape = qCard.shape

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
    Returns the flattened, re-sized, grayed image.
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
    hsv = cv2.cvtColor(card_roi, cv2.COLOR_BGR2HSV)
    # hsv = card_roi

    # Define HSV ranges for each color
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    
    green_lower = np.array([35, 50, 50])
    green_upper = np.array([85, 255, 255])
    
    purple_lower = np.array([130, 50, 50])
    purple_upper = np.array([155, 255, 255])
    
    # Create masks for each color
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = red_mask1 + red_mask2
    
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
    
    # Calculate percentage of pixels in each color range
    total_pixels = card_roi.shape[0] * card_roi.shape[1]
    red_pixels = np.sum(red_mask > 0)
    green_pixels = np.sum(green_mask > 0)
    purple_pixels = np.sum(purple_mask > 0)
    
    # Calculate percentages
    red_percent = red_pixels / total_pixels
    green_percent = green_pixels / total_pixels
    purple_percent = purple_pixels / total_pixels
    
    print("\n=== Color Detection ===")
    print(f"Average HSV: {np.mean(hsv, axis=(0,1))}")
    print(f"Red pixels: {red_percent:.3f}")
    print(f"Green pixels: {green_percent:.3f}")
    print(f"Purple pixels: {purple_percent:.3f}")
    
    # Determine color based on highest percentage
    if max(red_percent, green_percent, purple_percent) < 0.05:
        print("Warning: Very few colored pixels detected")
        
    if red_percent > green_percent and red_percent > purple_percent:
        detected_color = 'red'
    elif green_percent > red_percent and green_percent > purple_percent:
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
            
        print(f"\nContour {i}:")
        print(f"  Area = {props['area']}, Relative area = {props['relative_area']:.4f}")
        print(f"  Circularity: {props['circularity']:.3f}")
        print(f"  Aspect ratio: {props['aspect_ratio']:.3f}")
        print(f"  Approx vertices: {props['vertices']}")
        print(f"  Solidity: {props['solidity']:.3f}")
        
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
    # for c in final_contours:
    if final_contours:  # Check if list is not empty before accessing first element
        cv2.drawContours(symbol_mask, [final_contours[0]], -1, 255, -1)
    
    # # Debug shading detection
    # total_pixels = card_roi.shape[0] * card_roi.shape[1]
    # white_pixels = np.sum(thresh == 255)
    # fill_fraction = white_pixels / total_pixels
    
    # Get the thresholded image only within symbols
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
    
    # Debug visualization
    # cv2.imshow('Gray Warp', gray_warp)
    # cv2.imshow('Symbol Regions', symbol_regions)
    # cv2.imshow('Stripe Mask', stripe_mask)
    # cv2.imshow('Final Symbol Mask', symbol_mask)
    # cv2.waitKey(1)
    
    # Calculate fill ratio using the symbol mask
    symbol_pixels = np.sum(symbol_regions > 0)  # Total area of symbols
    _, filled_mask = cv2.threshold(gray_warp, 128, 255, cv2.THRESH_BINARY_INV)
    filled_mask = cv2.bitwise_and(filled_mask, symbol_regions)
    white_pixels = np.sum(filled_mask == 255)
    solid_fill_ratio = white_pixels / symbol_pixels if symbol_pixels > 0 else 0
    white_pixels = np.sum(symbol_mask == 255)
    striped_fill_ratio = white_pixels / symbol_pixels if symbol_pixels > 0 else 0
    
    print(f"\nShading Analysis Metrics:")
    print(f"Solid fill ratio: {solid_fill_ratio:.3f}")
    print(f"Striped fill ratio: {striped_fill_ratio:.3f}")
    
    # Adjust thresholds for different shading types
    THRESHOLD = 0.65

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