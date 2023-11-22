import cv2
import numpy as np
from insightface_face_detection import IFRClient

# Create an instance of the IFRClient class
client = IFRClient()

# Load YOLO model and configuration
def load_yolo_model():
    net = cv2.dnn.readNet("yolov3-spp.weights", "yolov3-spp.cfg")
    return net

# Load COCO labels
def load_coco_labels():
    with open("coco.names", "r") as f:
        classes = f.read().strip().split('\n')
    return classes


# Initialize the camera
def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to read camera feed")
    return cap

# Preprocess the frame for YOLO
def preprocess_frame(frame, net):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)
    return outputs

# Process YOLO detections
def process_yolo_detections(frame, outputs, classes):
    confidence_threshold = 0.5
    score_threshold = 0.4
    nms_threshold = 0.4
    cell_phones = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold and classes[class_id] == 'cell phone':
                center_x, center_y = int(detection[0] * frame.shape[1]), int(detection[1] * frame.shape[0])
                w, h = int(detection[2] * frame.shape[1]), int(detection[3] * frame.shape[0])
                x, y = center_x - w // 2, center_y - h // 2
                cell_phones.append((x, y, x + w, y + h))
                cv2.putText(frame, "cell phone", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return cell_phones

# Process faces and detect spoof
def process_faces(frame, faces, cell_phones, offsetPercentageW, offsetPercentageH):
    for face in faces:
        x_min, y_min, x_max, y_max = face
        width = x_max - x_min
        height = y_max - y_min

        face_within_cellphone = any(
            x_min >= x and y_min >= y and x_max <= x_plus_w and y_max <= y_plus_h for x, y, x_plus_w, y_plus_h in cell_phones
        )

        predicted_label = 'spoof' if face_within_cellphone else 'real'

        offsetW = (offsetPercentageW / 100) * width
        x_min = int(x_min - offsetW)
        x_max = int(x_max + offsetW)
        offsetH = (offsetPercentageH / 100) * height
        y_min = int(y_min - offsetH)
        y_max = int(y_max + offsetH)

        x_min = max(0, x_min)
        y_min = max(0, y_min)

        face_region = frame[y_min:y_max, x_min:x_max]
        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        cv2.putText(frame, predicted_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

# Main function to process the video stream
def main():
    net = load_yolo_model()
    classes = load_coco_labels()
    cap = initialize_camera()
    offsetPercentageW = 20
    offsetPercentageH = 20

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        rgb_frame = frame[:, :, ::-1]

        faces = client.face_locations(rgb_frame)
        outputs = preprocess_frame(frame, net)
        cell_phones = process_yolo_detections(frame, outputs, classes)
        process_faces(frame, faces, cell_phones, offsetPercentageW, offsetPercentageH)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
