import cv2
import numpy as np
import pygame

# Initialize Pygame for sound
pygame.init()

# Load piano sounds
sounds = [
    pygame.mixer.Sound(f"note_{i}.wav") for i in range(1, 8)
]  # Replace with actual piano note files (note_1.wav, note_2.wav, etc.)

# Piano dimensions
piano_keys = 14  # Number of keys (7 white and 7 black)
screen_width = 1280  # Full-screen width
screen_height = 720  # Full-screen height
white_key_width = screen_width // 7  # White key width
black_key_width = white_key_width // 2  # Black key width
black_key_height = int(screen_height * 0.4)  # Black key height
white_key_height = int(screen_height * 0.6)  # White key height

# Set up the OpenCV window
cv2.namedWindow("Virtual Piano", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Virtual Piano", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def draw_piano(image):
    """Draw the piano keys on the screen."""
    # Draw white keys
    for i in range(7):
        x_start = i * white_key_width
        cv2.rectangle(
            image, 
            (x_start, screen_height - white_key_height), 
            (x_start + white_key_width, screen_height), 
            (255, 255, 255), 
            -1
        )
        cv2.rectangle(
            image, 
            (x_start, screen_height - white_key_height), 
            (x_start + white_key_width, screen_height), 
            (0, 0, 0), 
            2
        )
    
    # Draw black keys
    black_key_positions = [0.75, 1.75, 3.25, 4.25, 5.75]  # Relative positions of black keys
    for i in range(len(black_key_positions)):
        x_start = int((black_key_positions[i] + i) * white_key_width)
        cv2.rectangle(
            image, 
            (x_start, screen_height - white_key_height), 
            (x_start + black_key_width, screen_height - black_key_height), 
            (0, 0, 0), 
            -1
        )

def detect_fingertip(frame):
    """Detect hand and return the coordinates of the index fingertip."""
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological transformations to clean up the mask
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour (assumed to be the hand)
        largest_contour = max(contours, key=cv2.contourArea)

        # Draw the contour for visualization
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

        # Find the convex hull
        hull = cv2.convexHull(largest_contour, returnPoints=False)

        # Find convexity defects
        defects = cv2.convexityDefects(largest_contour, hull)

        if defects is not None:
            for i in range(defects.shape[0]):
                start_idx, end_idx, far_idx, _ = defects[i, 0]
                start_point = tuple(largest_contour[start_idx][0])
                if start_point[1] < screen_height - white_key_height:  # Fingertip above keys
                    return start_point  # Return fingertip coordinates
    return None

def play_note_if_touched(fingertip, image):
    """Check if the fingertip touches a key and play the corresponding note."""
    if fingertip:
        x, y = fingertip
        if screen_height - white_key_height <= y <= screen_height:
            key_index = x // white_key_width
            if 0 <= key_index < 7:
                sounds[key_index].play()  # Play sound
                cv2.rectangle(
                    image,
                    (key_index * white_key_width, screen_height - white_key_height),
                    ((key_index + 1) * white_key_width, screen_height),
                    (0, 255, 0),
                    -1,
                )

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Resize frame to full screen
    frame = cv2.resize(frame, (screen_width, screen_height))

    # Draw piano on the frame
    draw_piano(frame)

    # Detect fingertip
    fingertip = detect_fingertip(frame)

    # Play note if fingertip is touching a key
    play_note_if_touched(fingertip, frame)

    # Display the result
    cv2.imshow("Virtual Piano", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
