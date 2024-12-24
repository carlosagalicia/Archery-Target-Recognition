import cv2
import numpy as np
import tkinter as tk

# Get screen dimensions
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# Configure grid to organize windows
rows = 2
cols = 2
window_width = screen_width // cols
window_height = screen_height // rows

# Lower and upper color recognition limits
red_lower = np.array([158, 87, 0], np.uint8)
red_upper = np.array([180, 255, 255], np.uint8)

green_lower = np.array([35, 80, 2], np.uint8)
green_upper = np.array([75, 255, 255], np.uint8)

orange_lower = np.array([1, 87, 111], np.uint8)
orange_upper = np.array([15, 255, 255], np.uint8)

target_lower = np.array([0, 87, 180], np.uint8)
target_upper = np.array([15, 255, 255], np.uint8)

blue_lower = np.array([94, 130, 2], np.uint8)
blue_upper = np.array([113, 255, 255], np.uint8)

yellow_lower = np.array([20, 80, 20], np.uint8)
yellow_upper = np.array([35, 255, 255], np.uint8)

capture = cv2.VideoCapture(0)
width = int(capture.get(3))
height = int(capture.get(4))

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    image = frame

    # Dimensions of the target zone
    center_width = int(width / 2)
    center_height = int(height / 2)
    xp1 = center_width - 30
    xp2 = center_width + 30
    yp1 = center_height - 30
    yp2 = center_height + 30

    # Convert to HSV and apply blur
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

    # Create masks for colors
    masks = {
        "Yellow": cv2.inRange(blurred, yellow_lower, yellow_upper),
        "Blue": cv2.inRange(blurred, blue_lower, blue_upper),
        # "Green": cv2.inRange(blurred, green_lower, green_upper),
        # "Orange": cv2.inRange(hsv, orange_lower, orange_upper),
        # "Red": cv2.inRange(blurred, red_lower, red_upper),
        "Target_mask_1": cv2.inRange(blurred, target_lower, target_upper)
    }

    # Highlight edges in the orange mask
    edges = cv2.Canny(masks["Target_mask_1"], 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # For each detected contour...
    for card in contours:
        # Calculate starting coordinates, width, and height of the rectangle
        x1, y1, w, h = cv2.boundingRect(card)

        # Calculate the contour area
        epsilon = 0.09 * cv2.arcLength(card, True)
        approx = cv2.approxPolyDP(card, epsilon, True)
        area = cv2.contourArea(card)
        print("Area: ", area)

        if area >= 1000:
            # Draw the rectangle
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 127, 0), 2)

            # Calculate the centroid
            M = cv2.moments(card)
            if M["m00"] == 0: M["m00"] = 1
            x = int(M["m10"] / M["m00"])  # Average x positions (Centroid x coordinate)
            y = int(M["m01"] / M["m00"])  # Average y positions (Centroid y coordinate)

            # If the centroid is in the target zone, display the message "Fire"
            if xp1 < x < xp2 and yp1 < y < yp2:
                cv2.putText(image, "Fire", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)

            # Draw a point at the center
            cv2.circle(image, (x, y), radius=1, color=(255, 0, 0), thickness=-1)

    # Draw contours and target zone
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    cv2.rectangle(frame, (xp1, yp1), (xp2, yp2), (0, 0, 255), 2)

    # Display each resized windowq
    all_windows = list(masks.items()) + [("webCam", frame)]  # + [("Edges", edges)]
    for i, (name, img) in enumerate(all_windows):
        resized_img = cv2.resize(img, (int(window_width * .85), int(window_height * .85)))
        cv2.imshow(name, resized_img)
        cv2.moveWindow(name, (i % cols) * int(window_width * .95), (i // cols) * int(window_height * .95))

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
