from insightface_face_detection import IFRClient
import cv2

# Create an instance of the IFRClient class
client = IFRClient()

def process_batch_faces(frames):
    face_batch = []

    for frame in frames:
        #print("hello")
        faces = process_faces(frame)
        face_batch.append(faces)

    return face_batch

def process_faces(frame):
    rgb_frame = frame[:, :, ::-1]

    faces = client.face_locations(rgb_frame)

    return faces
