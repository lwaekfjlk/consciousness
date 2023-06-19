import cv2
import torch
import numpy as np
from torchvision import transforms

# Define your model architecture
class EmotionClassifier(torch.nn.Module):
    # Define your model here
    pass

# Load the cascade
import face_recognition
image = face_recognition.load_image_file("2_339_frame.png")
face_locations = face_recognition.face_locations(image)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the emotion model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
emotion_model = EmotionClassifier()
emotion_model.load_state_dict(torch.load('emotion_model.pth'))
emotion_model = emotion_model.to(device)
emotion_model.eval()
'''

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Read the frame
frame = cv2.imread('2_339_frame.png')

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

for idx, face_location in enumerate(face_locations):

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()

# Save the resulting image
cv2.imwrite('emotion_detection_result.jpg' )