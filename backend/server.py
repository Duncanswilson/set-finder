# backend.py
import cv2
import numpy as np
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
from typing import List, Dict, Any

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Utility functions remain the same
def detect_color(card_roi):
    hsv = cv2.cvtColor(card_roi, cv2.COLOR_BGR2HSV)
    
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

def detect_features(card_roi):
    print("\n=== Detecting Features for Card ===")
    
    gray = cv2.cvtColor(card_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try adaptive thresholding instead of simple binary threshold
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,  # Block size
        2    # C constant
    )
    
    # Add morphological operations to clean up noise
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Try multiple contour retrieval methods if first fails    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        # If no contours found, try different threshold
        _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert contours to relative coordinates
    relative_contours = []
    height, width = card_roi.shape[:2]
    card_area = height * width
    
    # Store valid contours and their properties
    valid_contours = []
    contour_properties = []
    
    # First pass: gather properties of all potential symbol contours
    areas = []
    for c in contours:
        area = cv2.contourArea(c)
        relative_area = area / card_area
        if 0.05 < relative_area < 0.3:  # Basic size filter
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
        elif ((props['vertices'] == 4 or 
               (props['vertices'] <= 6 and props['aspect_ratio'] > 0.6)) and 
               props['solidity'] > 0.95):
            shapes.append('diamond')
            print("  - Classified as diamond")
        elif props['solidity'] > 0.8:  # Increased minimum solidity for squiggles
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
    detected_quantity = symbol_count

    # Use only the final filtered contours for shading detection
    symbol_mask = np.zeros_like(thresh)
    for c in final_contours:
        cv2.drawContours(symbol_mask, [c], -1, 255, -1)
    
    # Debug shading detection
    total_pixels = card_roi.shape[0] * card_roi.shape[1]
    white_pixels = np.sum(thresh == 255)
    fill_fraction = white_pixels / total_pixels
    
    # Calculate fill fraction for detected symbols only
    symbol_mask = np.zeros_like(thresh)
    for c in contours:
        area = cv2.contourArea(c)
        relative_area = area / card_area
        if 0.05 < relative_area < 0.3:
            cv2.drawContours(symbol_mask, [c], -1, 255, -1)
    
    # Get the thresholded image only within symbols
    symbol_content = thresh & symbol_mask
    
    # Calculate basic fill statistics
    symbol_pixels = np.sum(symbol_mask > 0)
    filled_pixels = np.sum(symbol_content > 0)
    fill_ratio = filled_pixels / symbol_pixels if symbol_pixels > 0 else 0

    # Enhanced pattern analysis
    if symbol_pixels > 0:
        # 1. Analyze horizontal pattern
        horizontal_profile = np.sum(symbol_content, axis=1)
        horizontal_profile = horizontal_profile[horizontal_profile > 0]
        
        # 2. Analyze vertical pattern
        vertical_profile = np.sum(symbol_content, axis=0)
        vertical_profile = vertical_profile[vertical_profile > 0]
        
        # Calculate pattern metrics
        if len(horizontal_profile) > 5 and len(vertical_profile) > 5:
            # Compute variations in the profiles
            h_variations = np.abs(np.diff(horizontal_profile))
            v_variations = np.abs(np.diff(vertical_profile))
            
            # Normalize variations
            h_variations = h_variations / np.max(h_variations) if np.max(h_variations) > 0 else h_variations
            v_variations = v_variations / np.max(v_variations) if np.max(v_variations) > 0 else v_variations
            
            # Calculate pattern regularity scores
            h_regularity = np.std(h_variations)
            v_regularity = np.std(v_variations)
            
            # Calculate frequency of changes (stripes should have more changes)
            h_changes = np.sum(np.abs(np.diff(np.signbit(np.diff(horizontal_profile)))))
            v_changes = np.sum(np.abs(np.diff(np.signbit(np.diff(vertical_profile)))))
            
            pattern_score = (h_changes + v_changes) / (len(horizontal_profile) + len(vertical_profile))
            regularity_score = 2.0 - (h_regularity + v_regularity)  # Lower variation means more regular pattern
        else:
            pattern_score = 0
            regularity_score = 0
    else:
        pattern_score = 0
        regularity_score = 0

    # Calculate texture features
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(symbol_content, kernel, iterations=1)
    eroded = cv2.erode(symbol_content, kernel, iterations=1)
    texture_diff = cv2.absdiff(dilated, eroded)
    texture_score = np.sum(texture_diff) / symbol_pixels if symbol_pixels > 0 else 0

    # New classification logic with adjusted thresholds
    print(f"\nShading Analysis Metrics:")
    print(f"Fill ratio: {fill_ratio:.3f}")
    print(f"Pattern score: {pattern_score:.3f}")
    print(f"Regularity score: {regularity_score:.3f}")
    print(f"Texture score: {texture_score:.3f}")

    # Classification using multiple features
    if fill_ratio > 0.75:
        shading = 'solid'
    elif (pattern_score > 0.15 and regularity_score > 0.5) or \
         (0.2 < fill_ratio < 0.7 and texture_score > 75):
        shading = 'striped'
    else:
        shading = 'open'

    print(f"Detected shading: {shading}")
    
    return (detected_shape, detected_quantity, shading, relative_contours)

def all_same_or_all_different(a, b, c):
    return (a == b == c) or (a != b and b != c and a != c)

def is_set(card1, card2, card3):
    return (all_same_or_all_different(card1['color'], card2['color'], card3['color']) and
            all_same_or_all_different(card1['shape'], card2['shape'], card3['shape']) and
            all_same_or_all_different(card1['shading'], card2['shading'], card3['shading']) and
            all_same_or_all_different(card1['quantity'], card2['quantity'], card3['quantity']))

def detect_cards_and_sets(image_path: str) -> Dict[str, Any]:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = []
    card_id = 0
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(approx)
            card_roi = img[y:y+h, x:x+w].copy()

            print(f"\nProcessing card {card_id} with bounding box: {x}, {y}, {w}, {h}")
            detected_color = detect_color(card_roi)
            detected_shape, detected_quantity, detected_shading, contours = detect_features(card_roi)

            cards.append({
                'id': card_id,
                'bbox': [int(x), int(y), int(w), int(h)],
                'color': detected_color,
                'shape': detected_shape,
                'shading': detected_shading,
                'quantity': detected_quantity,
                'contours': contours
            })
            card_id += 1

    found_sets = []
    for i in range(len(cards)):
        for j in range(i+1, len(cards)):
            for k in range(j+1, len(cards)):
                if is_set(cards[i], cards[j], cards[k]):
                    found_sets.append([cards[i]['id'], cards[j]['id'], cards[k]['id']])

    return {
        'cards': cards,
        'sets': found_sets
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the image
        results = detect_cards_and_sets(file_path)
        
        # Print card details
        print("\nDetected Cards:")
        print("-" * 50)
        for card in results['cards']:
            print(f"Card {card['id']}:")
            print(f"  Quantity: {card['quantity']}")
            print(f"  Color: {card['color']}")
            print(f"  Shape: {card['shape']}")
            print(f"  Shading: {card['shading']}")
            print("-" * 50)

        # Clean up
        os.remove(file_path)

        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",  # use "server:app" instead of app directly for reload to work
        host="0.0.0.0",
        port=5001,
        reload=True,  # Enable hot reloading
        reload_dirs=["backend"]  # Watch the backend directory for changes
    )