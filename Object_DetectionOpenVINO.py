from openvino.inference_engine import IECore
import cv2
import numpy as np

# Initialize OpenVINO Runtime
ie = IECore()

# Read the network and corresponding weights from file
net = ie.read_network(model='C:/Users/machk/gitkraken/SmartTrafficLights/data/vehicle-detection-0201.xml', weights='C:/Users/machk/gitkraken/SmartTrafficLights/data/vehicle-detection-0201.bin')

# Load the network on the inference engine
exec_net = ie.load_network(network=net, device_name='CPU')

# Load the video
cap = cv2.VideoCapture('C:/Users/machk/gitkraken/SmartTrafficLights/data/carstest.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (resize, reshape, etc., as per model requirements)
    # This example assumes your model requires 300x300 RGB images
    input_frame = cv2.resize(frame, (300, 300))
    input_frame = input_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    input_frame = np.expand_dims(input_frame, axis=0)

    # Perform inference
    output = exec_net.infer(inputs={'input_layer_name': input_frame})

    # Process the output - This depends on your model's output
    # Typically, you will have to extract bounding boxes and possibly scores
    # This is a placeholder for processing; replace it with your model's specifics
    for box in output['output_layer_name'][0][0]:
        # Assuming box = [image_id, label, conf, x_min, y_min, x_max, y_max]
        if box[2] > 0.5:  # Confidence threshold
            xmin = int(box[3] * frame.shape[1])
            ymin = int(box[4] * frame.shape[0])
            xmax = int(box[5] * frame.shape[1])
            ymax = int(box[6] * frame.shape[0])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    # Display the output
    cv2.imshow('Output', frame)

    # Break loop with ESC key
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
