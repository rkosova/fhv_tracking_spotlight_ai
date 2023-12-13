from pytorchyolo import detect, models
import cv2
import requests


model = models.load_model(
    "./config/yolov3-tiny.cfg",
    "./weights/yolov3-tiny.weights"
).to('cuda')


cap = cv2.VideoCapture(0)
WIDTH = 500
W_CELLS = 9
W_INC = WIDTH/W_CELLS
W_RIGHT = [i for i in range(int(W_INC), WIDTH+1, int(W_INC))]

counter = 0 
previous = False
try:
    while True:
        ret, frame = cap.read()
        im = frame
        frame = cv2.resize(frame, (500, 500))

        boxes = detect.detect_image(model, frame)  

        if len(boxes) > 0:
            left_x = boxes[0][0]
            left_y = boxes[0][1]

            right_x = boxes[0][2]
            right_y = boxes[0][3]

            center_x = int((left_x + right_x)/2)
            center_y = int((left_y + right_y)/2)
                        
            for wr_i in range(len(W_RIGHT)):
                if center_x < W_RIGHT[wr_i]:
                    grid_x = wr_i
                    break

            grid_x = 5 - grid_x
            # requests.post("http://172.21.64.54:8080/location", data=f"x_location={50 - grid_x}")
            print("a")
            if not previous:
                # requests.post("http://172.21.64.60:5000/upload", files={"d": im})
                  _, img_encoded = cv2.imencode('.jpg', im)
                  img_bytes = img_encoded.tobytes()
                  e = requests.post("http://127.0.0.1:5000/process_data", files={"image": img_bytes})
                  print()
                  previous = True
        else:
            previous = False


except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
