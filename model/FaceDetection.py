from cv2 import cv2


class FaceDetection():
    # Loading the cascades
    _face_cascading = cv2.CascadeClassifier("cascade-feature/haarcascade_frontalface_default.xml")
    _eye_cascading = cv2.CascadeClassifier("cascade-feature/haarcascade_eye.xml")

    def __init__(self, window_name="Face Detection", detect_eye=False):
        self._window_name = window_name
        self._detect_eye = detect_eye

    def _detect(self, gray_image, colored_img):
        faces = self._face_cascading.detectMultiScale(image=gray_image,
                                                      scaleFactor=1.3,
                                                      minNeighbors=5)  # detecting faces in gray scale image
        for (x, y, width, height) in faces:
            cv2.rectangle(colored_img, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
            if not self._detect_eye:
                continue
            rof_gray_img = gray_image[y:y + height, x:x + width]  # region of face in gray scale image
            rof_colored_img = colored_img[y:y + width, x:x + width]  # region of face in colored image
            eyes = self._eye_cascading.detectMultiScale(image=rof_gray_img,
                                                        scaleFactor=1.1,
                                                        minNeighbors=3)  # detecting eyes

            for (eye_x, eye_y, eye_width, eye_height) in eyes:
                cv2.rectangle(rof_colored_img, (eye_x, eye_y), (eye_x + eye_width, eye_y + eye_height),
                              color=(0, 200, 0),
                              thickness=2)
        return colored_img

    def start_face_detection(self):
        # doing some face detection with the webcam
        video_capture = cv2.VideoCapture(0)  # open the webcam

        while True:
            _, frame = video_capture.read()  # read last frame of webcam
            # cv2.imshow("sad", frame)
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canvas = self._detect(gray_img, frame)
            cv2.imshow(self._window_name, canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()
