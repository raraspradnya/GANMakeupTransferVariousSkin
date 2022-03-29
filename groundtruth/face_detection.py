import cv2
import dlib
import numpy as np
PREDICTOR_PATH = '../model/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)

## Face detection
def face_detection(img,upsample_times=1):
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, upsample_times)

    return faces

## Face and points detection
def face_points_detection(img, bbox:dlib.rectangle):
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, bbox)
    

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    coords = np.asarray(list([p.x, p.y] for p in shape.parts()), dtype=np.int)


    # return the array of (x, y)-coordinates
    return coords

def select_face(im, r=10, choose=True):
    faces = face_detection(im)

    if len(faces) == 0:
        return None, None, None

    if len(faces) == 1 or not choose:
        idx = np.argmax([(face.right() - face.left()) * (face.bottom() - face.top()) for face in faces])
        bbox = faces[idx]
    else:
        bbox = []

        def click_on_face(event, x, y, flags, params):
            if event != cv2.EVENT_LBUTTONDOWN:
                return

            for face in faces:
                if face.left() < x < face.right() and face.top() < y < face.bottom():
                    bbox.append(face)
                    break

        im_copy = im.copy()
        for face in faces:
            # draw the face bounding box
            cv2.rectangle(im_copy, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 1)
        cv2.imshow('Click the Face:', im_copy)
        cv2.setMouseCallback('Click the Face:', click_on_face)
        while len(bbox) == 0:
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        bbox = bbox[0]

    points = np.asarray(face_points_detection(im, bbox))

    im_w, im_h = im.shape[:2]
    left, top = np.min(points, 0)
    right, bottom = np.max(points, 0)

    x, y = max(0, left - r), max(0, top - r)
    w, h = min(right + r, im_h) - x, min(bottom + r, im_w) - y

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mask_eye = np.zeros_like(gray)
    left_eye_points2 = np.array(points[36:42], np.int32)
    right_eye_points2 = np.array(points[42:48], np.int32)
    mouth_points2 = np.array(points[60:68], np.int32)
    convexhull_left_eye = cv2.convexHull(left_eye_points2)
    convexhull_right_eye = cv2.convexHull(right_eye_points2)
    convexhull_mouth = cv2.convexHull(mouth_points2)
    mask_eye = cv2.fillConvexPoly(mask_eye, convexhull_left_eye, 255)
    mask_eye = cv2.fillConvexPoly(mask_eye, convexhull_right_eye, 255)
    mask_eye = cv2.fillConvexPoly(mask_eye, convexhull_mouth, 255)
    im[mask_eye == 255] = 0

    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y + h, x:x + w]


def select_all_faces(im, r=10):
    faces = face_detection(im)

    if len(faces) == 0:
        return None

    faceBoxes = {k : {"points" : None,
                      "shape" : None,
                      "face" : None} for k in range(len(faces))}
    for i, bbox in enumerate(faces):
        points = np.asarray(face_points_detection(im, bbox))

        im_w, im_h = im.shape[:2]
        left, top = np.min(points, 0)
        right, bottom = np.max(points, 0)

        x, y = max(0, left - r), max(0, top - r)
        w, h = min(right + r, im_h) - x, min(bottom + r, im_w) - y
        faceBoxes[i]["points"] = points - np.asarray([[x, y]])
        faceBoxes[i]["shape"] = (x, y, w, h)
        faceBoxes[i]["face"] = im[y:y + h, x:x + w]

    return faceBoxes
