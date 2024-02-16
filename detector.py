import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import pygame

def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # wait for the music to finish playing
        pygame.time.Clock().tick(10)  # delay for 10 milliseconds

# import RPi.GPIO as GPIO

# Initialize GPIO for buzzer
# GPIO.setmode(GPIO.BOARD)
buzzer_pin = 18
# GPIO.setup(buzzer_pin, GPIO.OUT)

# Load the model
model = tf.saved_model.load('.')

# Load the label map
label_map_path = 'object_detection/data/mscoco_complete_label_map.pbtxt'
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Prepare the input
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, axis=0), dtype=tf.float32)

    # Run the model
    input_tensor = tf.cast(input_tensor, dtype=tf.uint8)
    results = model(input_tensor)

    # Extract the detection boxes, classes, and scores
    detection_boxes = results['detection_boxes'][0].numpy()
    detection_classes = results['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = results['detection_scores'][0].numpy()

    # Check if cat is detected
    if any((detection_scores > 0.5) & (detection_classes == 17)):
        print("Cat detected!")
        play_audio('buzzer.mp3')

        # GPIO.output(buzzer_pin, GPIO.HIGH)  # Activate buzzer
    else:
        # GPIO.output(buzzer_pin, GPIO.LOW)  # Deactivate buzzer
        pass

    # Visualize the results
    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        detection_boxes,
        detection_classes,
        detection_scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)

    # Display the frame (optional)
    cv2.imshow('Cat Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
# GPIO.cleanup()