from model.FaceDetection import FaceDetection


if __name__ == '__main__':
    face_detection = FaceDetection(detect_eye=False )
    face_detection.start_face_detection()
