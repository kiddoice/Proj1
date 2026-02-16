import cv2
import numpy as np

class YOLODetector:
    def __init__(self, config_path, weights_path, classes_path):
        self.config_path = config_path
        self.weights_path = weights_path
        self.classes_path = classes_path
        self.net = self._load_model()
        self.classes = self._load_classes()

    def _load_model(self):
        return cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)

    def _load_classes(self):
        with open(self.classes_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def detect(self, image):
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        result_boxes = [boxes[i] for i in range(len(boxes)) if i in indexes]
        result_classes = [class_ids[i] for i in range(len(class_ids)) if i in indexes]

        return result_boxes, result_classes
