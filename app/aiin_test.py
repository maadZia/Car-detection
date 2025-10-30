from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from detection.plate_detection import PlateDetection
from detection.ocr_detection import read_licence_plate
from detection.car_detection import CarDetection
from detection.color_detection import ColorDetection
import datetime
import cv2
import os

# --- KONFIGURACJA ---
camera_number = 1
rotate = False
CURRENT_VIDEO = "amcia3.mp4"

app = Flask(__name__)

plate_detector = PlateDetection('../models/my_model/my_model.pt')
car_detector = CarDetection('../models/car_model/car_model.pt')
color_detector = ColorDetection()

# aktualna detekcja ---
current_plate_result = {"plate": None, "color": None, "timestamp": None}

def generate_frames(video_path=None):
    if not video_path:
        video_path = os.path.join('videos', CURRENT_VIDEO)

    plate_detector.last_bbox = None
    extension = os.path.splitext(video_path)[1].lower()

    # zdjęcia
    if extension in ['.jpg', '.jpeg', '.png']:
        frame = cv2.imread(video_path)
        if frame is None:
            return

        # detekcja tablicy
        plate_img = plate_detector._process_frame(frame)
        plate_number = None
        if plate_img is not None:
            reads = []
            for _ in range(3):
                plate_text, conf = read_licence_plate(plate_img)
                if plate_text and len(plate_text) > 4:
                    reads.append(plate_text)
            if reads:
                plate_number = max(set(reads), key=reads.count)

        # detekcja samochodu i koloru
        car_color = None
        car_results = car_detector.model(frame, verbose=False)
        for det in car_results[0].boxes:
            if det.conf > car_detector.min_thresh:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                car_crop = frame[y1:y2, x1:x2]
                car_color = color_detector.classify_color(car_crop)

        # ramka dla tablicy
        if plate_detector.last_bbox is not None:
            x1, y1, x2, y2 = plate_detector.last_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # aktualny wynik detekcji
        current_plate_result.update({
            "plate": plate_number,
            "color": car_color,
            "timestamp": datetime.datetime.now().isoformat()
        })

        # jedną klatka jako mjpeg
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return

    # film
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        plate_number = current_plate_result.get("plate", "Processing...")
        car_color = None

        if frame_count % 3 == 0:
            plate_img = plate_detector._process_frame(frame)
            if plate_img is not None:
                reads = []
                for _ in range(3):
                    plate_text, conf = read_licence_plate(plate_img)
                    if plate_text and len(plate_text) > 4:
                        reads.append(plate_text)
                if reads:
                    plate_number = max(set(reads), key=reads.count)

        car_results = car_detector.model(frame, verbose=False)
        for det in car_results[0].boxes:
            if det.conf > car_detector.min_thresh:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                car_crop = frame[y1:y2, x1:x2]
                car_color = color_detector.classify_color(car_crop)

        if plate_detector.last_bbox is not None:
            x1, y1, x2, y2 = plate_detector.last_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        current_plate_result.update({
            "plate": plate_number,
            "color": car_color,
            "timestamp": datetime.datetime.now().isoformat()
        })

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def reset_detection_state():
    global current_plate_result
    current_plate_result = {"plate": None, "color": None, "timestamp": None}
    plate_detector.last_bbox = None


# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html', current_video=CURRENT_VIDEO)

@app.route('/video_with_detection')
def video_with_detection():
    video_file = request.args.get('video', CURRENT_VIDEO)
    video_path = os.path.join('videos', video_file)
    if not os.path.exists(video_path):
        return "Video not found", 404

    if request.args.get('reset') == 'true':
        reset_detection_state()

    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest_detection')
def latest_detection():
    return jsonify(current_plate_result)

@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory('videos', filename)

if __name__ == '__main__':
    app.run(debug=True)
