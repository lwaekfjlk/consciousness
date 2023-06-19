# Import the necessary libraries
import cv2
from mtcnn import MTCNN

from fer import FER

img = cv2.imread("1_7047_frame.png")
detector = FER()
emotions = detector.top_emotion(img)[0]

import pdb; pdb.set_trace()

# Create an instance of the MTCNN detector class
detector = MTCNN()

# Load the image
image = cv2.imread("1_7047_frame.png")

# Use the MTCNN detector to find faces in the image
result = detector.detect_faces(image)

# The result will be a list of dictionaries.
# Each dictionary contains the bounding box coordinates and facial landmarks for one face.
for i in range(len(result)):
    x1, y1, width, height = result[i]['box']
    x2, y2 = x1 + width, y1 + height
    
    # Draw a rectangle over the detected face (optional)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the image or display it
cv2.imwrite("emotion_detection_result.jpg", image)


import pdb; pdb.set_trace()


from PIL import Image
import face_recognition
from fer import FER
from facenet_pytorch import MTCNN


image = face_recognition.load_image_file("2_339_frame.png")
face_locations = face_recognition.face_locations(image)

emo_detector = FER(mtcnn=True)

for idx, face_location in enumerate(face_locations):
    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    captured_emotions = emo_detector.detect_emotions(pil_image)
    pil_image.save('emotion_detection_result.jpg')
    import pdb; pdb.set_trace()
