import cv2
import numpy as np
import concurrent.futures
from insightface_face_detection import IFRClient
import time

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
    cap = cv2.VideoCapture("http://192.168.3.208:4747/video")
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

# Process YOLO detections for a single frame
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

    return cell_phones

# Process a batch of frames for cell phone detection
def process_batch_frames(frames, net, classes):
    cell_phone_batch = []

    for frame in frames:
        outputs = preprocess_frame(frame, net)
        cell_phones = process_yolo_detections(frame, outputs, classes)
        cell_phone_batch.append(cell_phones)

    return cell_phone_batch

# Main function to process the video stream
def main():
    net = load_yolo_model()
    classes = load_coco_labels()
    cap = initialize_camera()
    offsetPercentageW = 20
    offsetPercentageH = 20

    batch_size = 64
    frame_buffer = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            frame = cap.read()[1]
            if frame is None:
                break

            frame = cv2.resize(frame, (640, 360))
            frame_buffer.append(frame)

            if len(frame_buffer) == batch_size:
                start_time=time.time()
                cell_phone_batch = process_batch_frames(frame_buffer, net, classes)
                end_time = time.time()
                frame_processing_time = end_time - start_time
                print(f"Frame processing time: {frame_processing_time} seconds")
                frame_buffer = []
                frame_output = []
                for cell_phones in cell_phone_batch:
                    frame_output.append([
                        len(cell_phones),
                        cell_phones
                    ])

                print(frame_output)
                # Print the list of detected cell phones and their count for each frame in the batch
                #for i, cell_phones in enumerate(cell_phone_batch):
                #    print(f"Frame {i + 1} - Detected {len(cell_phones)} cell phones:")
                #    for j, cell_phone in enumerate(cell_phones):
                #        print(f"Cell Phone {j + 1}: {cell_phone}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
