import numpy as np
import cv2
from Cards import Set_card

class CardTracker:
    def __init__(self, memory_frames=10):
        self.memory_frames = memory_frames
        self.tracked_cards = {}  # Dictionary to store card histories
        self.required_misses = 8  # Number of missed detections before removing a card
        self.tracked_sets = []    # List to store currently tracked sets
        self.set_memory = 20      # Increased from 15 to 20 frames before removing a set
        self.set_counters = []    # Counter for each tracked set
        self.required_set_detections = 5  # Increased from 3 to 5
        self.set_detection_counts = []  # Track how many times we've seen each set
        
    def update(self, current_cards, frame):
        # Get current card centers
        current_centers = {tuple(card.center): card for card in current_cards}
        
        # Update existing tracked cards
        for center, history in list(self.tracked_cards.items()):
            # Find the closest current card to this tracked position
            closest_card = None
            min_dist = 50  # Maximum pixel distance to consider as same card
            
            for curr_center, card in current_centers.items():
                dist = np.sqrt((center[0] - curr_center[0])**2 + 
                             (center[1] - curr_center[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_card = card
            
            if closest_card:
                # Update history with new detection
                history.append(closest_card)
                if len(history) > self.memory_frames:
                    history.pop(0)
                # Update tracked position to new position
                new_center = tuple(closest_card.center)
                if new_center != center:
                    self.tracked_cards[new_center] = history
                    del self.tracked_cards[center]
                # Remove from current_centers to mark as handled
                del current_centers[tuple(closest_card.center)]
            else:
                # Card not found in current frame
                history.append(None)
                if len(history) > self.memory_frames:
                    history.pop(0)
                # Only remove if we've missed it many times
                if sum(1 for x in history[-self.required_misses:] if x is None) >= self.required_misses:
                    del self.tracked_cards[center]
        
        # Add any new cards immediately
        for center, card in current_centers.items():
            self.tracked_cards[center] = [card]
    
    def _smooth_property(self, values, valid_cards):
        """Return most common non-Unknown value from recent history."""
        if not values:
            return "Unknown"
        
        # Count occurrences of each value
        value_counts = {}
        for value in values:
            value_counts[value] = value_counts.get(value, 0) + 1
        
        # Return the most common value
        return max(value_counts.items(), key=lambda x: x[1])[0]

    def get_stable_cards(self):
        stable_cards = []
        for history in self.tracked_cards.values():
            valid_cards = [c for c in history if c is not None]
            if not valid_cards:
                continue
            
            # Create smoothed card from most recent detection
            smoothed_card = Set_card()
            latest_card = valid_cards[-1]
            
            # Use averaged center position for stability
            centers = np.array([card.center for card in valid_cards])
            smoothed_center = np.mean(centers, axis=0).astype(int)
            smoothed_card.center = smoothed_center.tolist()
            
            # Use the most stable contour (one closest to mean center)
            mean_center = np.mean([card.center for card in valid_cards], axis=0)
            distances = [np.linalg.norm(np.array(card.center) - mean_center) for card in valid_cards]
            most_stable_idx = np.argmin(distances)
            smoothed_card.contour = valid_cards[most_stable_idx].contour
            smoothed_card.corner_pts = valid_cards[most_stable_idx].corner_pts
            
            # Rest of physical properties from the stable card
            smoothed_card.width = valid_cards[most_stable_idx].width
            smoothed_card.height = valid_cards[most_stable_idx].height
            smoothed_card.warp = valid_cards[most_stable_idx].warp
            smoothed_card.roi = valid_cards[most_stable_idx].roi
            
            # Smooth card properties using most common values
            smoothed_card.color = self._smooth_property(
                [c.color for c in valid_cards if c.color != "Unknown"],
                valid_cards
            )
            smoothed_card.shape = self._smooth_property(
                [c.shape for c in valid_cards if c.shape != "Unknown"],
                valid_cards
            )
            smoothed_card.shading = self._smooth_property(
                [c.shading for c in valid_cards if c.shading != "Unknown"],
                valid_cards
            )
            smoothed_card.quantity = self._smooth_property(
                [c.quantity for c in valid_cards if c.quantity != "Unknown"],
                valid_cards
            )
            
            stable_cards.append(smoothed_card)
        
        return stable_cards

    def update_sets(self, current_sets):
        """Update tracked sets with new detections."""
        if not self.tracked_sets:
            # Initialize tracking for each new set
            self.tracked_sets = current_sets.copy()
            self.set_counters = [self.set_memory for _ in current_sets]
            self.set_detection_counts = [1 for _ in current_sets]
            return []  # Don't show any sets initially until they're stable

        if not current_sets:
            # If no current sets, decrement all counters more aggressively
            for i in range(len(self.tracked_sets) - 1, -1, -1):
                self.set_counters[i] -= 3  # More aggressive decay when no sets detected
                self.set_detection_counts[i] = max(1, self.set_detection_counts[i] - 2)  # Decrease detection count faster
                if self.set_counters[i] <= 0:
                    self.tracked_sets.pop(i)
                    self.set_counters.pop(i)
                    self.set_detection_counts.pop(i)
            return [s for i, s in enumerate(self.tracked_sets) 
                   if self.set_detection_counts[i] >= self.required_set_detections]

        # Match current sets with tracked sets
        matched_tracked = [False] * len(self.tracked_sets)
        matched_current = [False] * len(current_sets)

        # For each current set, try to match it with a tracked set
        for i, curr_set in enumerate(current_sets):
            best_match = -1
            best_match_score = 0

            for j, tracked_set in enumerate(self.tracked_sets):
                if matched_tracked[j]:
                    continue

                # Count how many cards overlap between the sets
                match_score = self._calculate_set_overlap(curr_set, tracked_set)
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match = j

            # If we found a good match, update the tracked set
            if best_match_score >= 2 and best_match >= 0:
                matched_current[i] = True
                matched_tracked[best_match] = True
                self.tracked_sets[best_match] = curr_set
                self.set_counters[best_match] = self.set_memory
                self.set_detection_counts[best_match] += 1  # Increment detection count

        # Add new sets
        for i, matched in enumerate(matched_current):
            if not matched:
                self.tracked_sets.append(current_sets[i])
                self.set_counters.append(self.set_memory)
                self.set_detection_counts.append(1)  # Initialize detection count for new set

        # Remove expired sets
        for i in range(len(self.tracked_sets) - 1, -1, -1):
            if i < len(matched_tracked) and not matched_tracked[i]:
                self.set_counters[i] -= 2  # Decrease counter faster when not matched
                self.set_detection_counts[i] = max(1, self.set_detection_counts[i] - 1)  # Decrease detection count
                if self.set_counters[i] <= 0:
                    self.tracked_sets.pop(i)
                    self.set_counters.pop(i)
                    self.set_detection_counts.pop(i)

        # Only return sets that have been detected enough times
        return [s for i, s in enumerate(self.tracked_sets) 
                if self.set_detection_counts[i] >= self.required_set_detections]

    def _calculate_set_overlap(self, set1, set2):
        """Calculate how many cards overlap between two sets based on card centers."""
        overlap_count = 0
        for contour1 in set1:
            M1 = cv2.moments(contour1)
            if M1["m00"] == 0:
                continue
            c1 = (int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"]))
            
            for contour2 in set2:
                M2 = cv2.moments(contour2)
                if M2["m00"] == 0:
                    continue
                c2 = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))
                
                # If centers are close enough, consider it the same card
                if np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) < 50:
                    overlap_count += 1
                    break
                    
        return overlap_count