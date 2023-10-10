import cv2
import detection
import numpy as np

def main():
    model_path = "yolov8s.onnx"
    yolo = detection.YOLOv8ONNX(model_path)

    # Open the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read the frame.")
                break

            # Perform detection
            detections, preprocessed_image_shape  = yolo.detect(frame)
            
            # Visualize detections only if there are any
            if len(detections) > 0:
                vis_image = yolo.visualize_detections(np.array(frame), detections, detection.nms.COCO_NAMES, detection.nms._COLORS)
            else:
                vis_image = frame

            # Display the frame
            cv2.imshow("Live Object Detection", vis_image)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release the camera and close the window
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
