import cv2
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Use a small model; replace with 'yolov8s.pt' or larger for better accuracy

# Open the video file or camera feed
video_path = "vid.mp4"  # Replace with your video file or 0 for webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file or camera feed.")
    exit()

# Set up Tkinter window
root = tk.Tk()
root.title("YOLO and Fuzzy Consistency Tracking")

# Create a label to display frames
label = tk.Label(root)
label.pack()

# Create a text box to display fuzzy information
text_box = tk.Text(root, height=10, width=80)
text_box.pack()

# Initialize previous centroids and sizes
prev_centroids = []
prev_sizes = []

# Define fuzzy logic variables
size = ctrl.Antecedent(np.arange(0, 101, 1), 'size')
speed = ctrl.Antecedent(np.arange(0, 101, 1), 'speed')
class_validity = ctrl.Consequent(np.arange(0, 101, 1), 'class_validity')

# Fuzzy membership functions
size['small'] = fuzz.trimf(size.universe, [0, 0, 50])
size['medium'] = fuzz.trimf(size.universe, [0, 50, 100])
size['large'] = fuzz.trimf(size.universe, [50, 100, 100])

speed['slow'] = fuzz.trimf(speed.universe, [0, 0, 50])
speed['moderate'] = fuzz.trimf(speed.universe, [0, 50, 100])
speed['fast'] = fuzz.trimf(speed.universe, [50, 100, 100])

# Fuzzy output
class_validity['low'] = fuzz.trimf(class_validity.universe, [0, 0, 50])
class_validity['medium'] = fuzz.trimf(class_validity.universe, [0, 50, 100])
class_validity['high'] = fuzz.trimf(class_validity.universe, [50, 100, 100])

# Fuzzy rules for vehicle classification
rule1 = ctrl.Rule(size['small'] & speed['slow'], class_validity['low'])
rule2 = ctrl.Rule(size['medium'] & speed['moderate'], class_validity['medium'])
rule3 = ctrl.Rule(size['large'] & speed['fast'], class_validity['high'])
rule4 = ctrl.Rule(size['small'] & speed['moderate'], class_validity['medium'])
rule5 = ctrl.Rule(size['large'] & speed['moderate'], class_validity['high'])

# Additional fuzzy rules for vehicle-specific classifications
rule6 = ctrl.Rule(size['large'] & speed['moderate'], class_validity['high'])
rule7 = ctrl.Rule(size['medium'] & speed['fast'], class_validity['medium'])
rule8 = ctrl.Rule(size['large'] & speed['slow'], class_validity['medium'])

# For Van
rule9 = ctrl.Rule(size['medium'] & speed['moderate'], class_validity['medium'])
rule10 = ctrl.Rule(size['small'] & speed['moderate'], class_validity['low'])
rule11 = ctrl.Rule(size['medium'] & speed['slow'], class_validity['low'])

# For Truck
rule12 = ctrl.Rule(size['large'] & speed['slow'], class_validity['high'])
rule13 = ctrl.Rule(size['large'] & speed['moderate'], class_validity['high'])
rule14 = ctrl.Rule(size['medium'] & speed['fast'], class_validity['medium'])

# For Car
rule15 = ctrl.Rule(size['small'] & speed['moderate'], class_validity['medium'])
rule16 = ctrl.Rule(size['small'] & speed['fast'], class_validity['high'])
rule17 = ctrl.Rule(size['small'] & speed['slow'], class_validity['low'])

# For Train
rule18 = ctrl.Rule(size['large'] & speed['fast'], class_validity['high'])
rule19 = ctrl.Rule(size['large'] & speed['moderate'], class_validity['high'])
rule20 = ctrl.Rule(size['large'] & speed['slow'], class_validity['medium'])

# Control system and simulation
class_validation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8,
                                            rule9, rule10, rule11, rule12, rule13, rule14, rule15,
                                            rule16, rule17, rule18, rule19, rule20])
class_validation_sim = ctrl.ControlSystemSimulation(class_validation_ctrl)


# Calculate speed from centroids
def calculate_speed(curr_centroid, prev_centroid):
    if prev_centroid:
        dx = curr_centroid[0] - prev_centroid[0]
        dy = curr_centroid[1] - prev_centroid[1]
        distance = np.sqrt(dx ** 2 + dy ** 2)
        speed = distance  # You can scale this value for actual speed
        return speed
    return 0


# Fuzzy validation for object classification
def fuzzy_validate_class(predicted_class, size, speed):
    text_box.delete(1.0, tk.END)

    # Normalize inputs and clamp values to range
    size = max(0, min(size, 100))
    speed = max(0, min(speed, 100))

    print(f"Normalized Size: {size}, Speed: {speed}")

    try:
        # Input normalized size and speed into the fuzzy system
        class_validation_sim.input['size'] = size
        class_validation_sim.input['speed'] = speed
        class_validation_sim.compute()

        # Get the fuzzy output
        validity_score = class_validation_sim.output['class_validity']
        print(f"Fuzzy Validity Score: {validity_score}")

        # Log to GUI
        text_box.insert(tk.END, f"Normalized Size: {size}\n")
        text_box.insert(tk.END, f"Normalized Speed: {speed}\n")
        text_box.insert(tk.END, f"Class Validity Score: {validity_score:.2f}\n")

        # Determine the class based on the fuzzy validity score
        if validity_score > 70:
            text_box.insert(tk.END, f"Prediction: Valid classification for {predicted_class}.\n")
        elif validity_score > 40:
            text_box.insert(tk.END,
                            f"Prediction: Likely valid classification for {predicted_class}, but check for accuracy.\n")
        else:
            text_box.insert(tk.END, f"Prediction: Invalid classification for {predicted_class}.\n")

        return validity_score
    except Exception as e:
        print(f"Error during fuzzy validation: {e}")
        return 0


def update_frame():
    global prev_centroids, prev_sizes  # Declare as global to modify these lists

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    # Perform object detection
    results = model(frame, conf=0.3)  # Confidence threshold of 0.3

    curr_centroids = []
    curr_sizes = []

    # Visualize the results on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            predicted_class = result.names[box.cls[0].item()]  # Class name predicted by YOLO
            confidence = box.conf.item()

            # Filter for vehicle classes (car, truck, bus, van, train)
            if predicted_class in ['car', 'truck', 'bus', 'van', 'train']:
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Calculate the centroid and size of the object
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                size = (x2 - x1) * (y2 - y1)

                # Calculate speed
                speed = 0
                if prev_centroids:
                    speed = calculate_speed(centroid, prev_centroids[0])

                # Fuzzy consistency check with previous frame's data
                validity_score = fuzzy_validate_class(predicted_class, size, speed)

                # Get the expected class based on fuzzy validation score
                if validity_score < 50:  # Low validity score, indicating incorrect classification
                    expected_class = "Invalid"
                else:
                    expected_class = predicted_class  # Assume the class is correct if fuzzy validity is high

                # Show the fuzzy predicted class in the bounding box
                cv2.putText(
                    frame,
                    f"{expected_class} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                # Check if the predicted class matches the expected class
                if predicted_class != expected_class:
                    # Show invalid classification with the wrong class
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bounding box
                    cv2.putText(
                        frame,
                        f"Invalid {predicted_class}",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
                else:
                    # If classification is valid, use a green bounding box
                    cv2.putText(
                        frame,
                        f"Valid: {validity_score:.2f}",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2,
                    )

                curr_centroids.append(centroid)
                curr_sizes.append(size)

    # Update previous centroids and sizes
    prev_centroids = curr_centroids
    prev_sizes = curr_sizes

    # Convert frame to Tkinter-compatible format
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    # Update the frame every 30ms
    label.after(30, update_frame)


# Start the frame update loop
update_frame()
root.mainloop()
