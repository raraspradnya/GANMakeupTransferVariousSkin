import cv2
import math
import matplotlib.pyplot as plt
import dlib
import numpy as np
import os
from PIL import Image
import mediapipe as mp
from face_parsing import vis_parsing_maps

# https://github.com/serengil/tensorflow-101/blob/master/python/face-alignment.py
def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)

def detectEyes(img):
    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor("../model/shape_predictor_68_face_landmarks.dat")

    faces = detector(img)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    landmarks = predictor(img_gray, faces[0])
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
    left_eye_points = landmarks_points[36:42]
    right_eye_points = landmarks_points[42:48]
    return left_eye_points, right_eye_points

def detectFace(img, scale_factor):
    img_rgb_copy = img.copy()
    detector = dlib.get_frontal_face_detector()

    faces = detector(img)
    if len(faces) == 1:
        x, y, w, h = convert_and_trim_bb(img,faces[0])

        # crop_img = img[y:y+h, x:x+w]
        cX = x + w // 2
        cY = y + h // 2
        M = (abs(w) + abs(h)) / 2
        
       
        ## Get the resized rectangle points
        newLeft = max(0, int(cX - scale_factor * M))
        newTop = max(0, int(cY - scale_factor * M))
        newRight = min(img.shape[1], int(cX + scale_factor * M))
        newBottom = min(img.shape[0], int(cY + scale_factor * M))

        new_img = Image.fromarray(img)
        new_img = np.array(new_img)


        ## Draw the circle and bounding boxes
        cv2.circle(img_rgb_copy, (cX, cY), radius=0, color=(0, 0, 255), thickness=2)
        cv2.rectangle(img_rgb_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(img_rgb_copy, (newLeft, newTop), (newRight, newBottom), (255, 0, 0), 2)

        plt.imshow(img_rgb_copy[:, :, ::-1])
        # plt.show()
        
        img = img[int(newTop):int(newBottom), int(newLeft):int(newRight)]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return img, img_gray, 1
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, img_gray, 0

def getCenter(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid = (int(sum(x) / len(points)), int(sum(y) / len(points)))
    return centroid

def alignFace(img_path):
    img = cv2.imread(img_path)
    plt.imshow(img[:, :, ::-1])
    # plt.show()

    img_raw = img.copy()

    plt.imshow(img[:, :, ::-1])
    # plt.show()
    right_eye, left_eye = detectEyes(img)
    
    left_eye_x = left_eye[0][0]
    left_eye_w = abs(left_eye[3][0] - left_eye[0][0])
    left_eye_y = left_eye[1][1]
    left_eye_h = abs(left_eye[4][1] - left_eye[1][1])

    right_eye_x = right_eye[0][0]
    right_eye_w = abs(right_eye[3][0] - right_eye[0][0])
    right_eye_y = right_eye[1][1]
    right_eye_h = abs(right_eye[4][1] - right_eye[1][1])


    cv2.rectangle(img,(left_eye_x, left_eye_y),(left_eye_x+left_eye_w, left_eye_y+left_eye_h), (255, 0,0), 2)
    cv2.rectangle(img,(right_eye_x, right_eye_y),(right_eye_x+right_eye_w, right_eye_y+right_eye_h), (255, 0,0), 2)
    plt.imshow(img[:, :, ::-1])
    # plt.show()

    # compute the center of mass for each eye
    leftEyeCenter = getCenter(left_eye)
    rightEyeCenter = getCenter(right_eye)
   
    left_eye_x = leftEyeCenter[0]; left_eye_y = leftEyeCenter[1]
    right_eye_x = rightEyeCenter[0]; right_eye_y = rightEyeCenter[1]
        
    center_of_eyes = (int((left_eye_x+right_eye_x)/2), int((left_eye_y+right_eye_y)/2))
    
    cv2.circle(img, leftEyeCenter, 2, (255, 0, 0) , 2)
    cv2.circle(img, rightEyeCenter, 2, (255, 0, 0) , 2)
    cv2.circle(img, center_of_eyes, 2, (255, 0, 0) , 2)
    
    #----------------------
    #find rotation direction
    
    if left_eye_y < right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
        print("rotate to clock direction")
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock
        print("rotate to inverse clock direction")
        
    #----------------------
    
    cv2.circle(img, point_3rd, 2, (255, 0, 0) , 2)
    cv2.line(img,rightEyeCenter, leftEyeCenter,(67,67,67),1)
    cv2.line(img,leftEyeCenter, point_3rd,(67,67,67),1)
    cv2.line(img,rightEyeCenter, point_3rd,(67,67,67),1)
    
    a = euclidean_distance(leftEyeCenter, point_3rd)
    b = euclidean_distance(rightEyeCenter, point_3rd)
    c = euclidean_distance(rightEyeCenter, leftEyeCenter)
    
    cos_a = (b*b + c*c - a*a)/(2*b*c)
    #print("cos(a) = ", cos_a)
    angle = np.arccos(cos_a)
    #print("angle: ", angle," in radian")
    
    angle = (angle * 180) / math.pi
    print("angle: ", angle," in degree")
    
    if direction == -1:
        angle = 90 - angle
    
    print("angle: ", angle," in degree")
    
    #--------------------
    #rotate image
    
    new_img = Image.fromarray(img_raw)
    new_img = np.array(new_img.rotate(direction * angle,  resample=Image.BICUBIC, expand=True))

    return new_img, 1

def expand2square(pil_img, background_color):
    pil_img = Image.fromarray(pil_img, 'RGB')
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def estimate_pose(img_path):
    img = cv2.imread(img_path)
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        img_h, img_w, img_c = img.shape
        face_3d = []
        face_2d = []

        if not results.multi_face_landmarks:
            print("No face detected!")
        else:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        # if idx == 1:
                        #     nose_2d = (lm.x * img_w, lm.y * img_h)
                        #     nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                    
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360

                print(y,x)

                # See where the user's head tilting
                if y < -10:
                    return "Looking Left"
                elif y > 10:
                    return "Looking Right"
                elif x < -10:
                    return "Looking Down"
                elif x > 10:
                    return "Looking Up"
                else:
                    return "Forward"

#------------------------

# test_set = ["D:/# Raras/src/data/coba\\1003.jpg"]
test_set = []
for (path, dirnames, filenames) in os.walk('D:/# Raras/src/makeup_dataset/all/images/non-makeup'):
    test_set.extend(os.path.join(path, name) for name in filenames)

for instance in test_set:
    if (os.path.exists(instance.strip())):
        img_name= instance[51:]
        print(img_name)
        pose = estimate_pose(instance)
        print(pose)

        if (pose == "Forward"):
            # try:
            #     alignedFace, check = alignFace(instance)
            #     if (check == 1):
            #         plt.imshow(alignedFace[:, :, ::-1])
            #         # plt.show()
            #         try:
            #             img, gray_img, checka = detectFace(alignedFace, 0.75)
            #         except:
            #             print(img_name + " is error")

            #         img = expand2square(img[:, :, ::-1], (0,0,0))
                    
            # plt.show()
            img = cv2.imread(instance)
            img = Image.fromarray(img[:,:,::-1])
            img_path = "D:/# Raras/src/makeup_dataset/final2/no_makeup/" + str(img_name)
            img.save(img_path)

            seg_path = 'D:/# Raras/src/makeup_dataset/all/segs/non-makeup/' + str(img_name)
            seg = cv2.imread(seg_path,0)
            seg_img = Image.fromarray(seg, 'L')
            vis_parsing_maps(img, seg_img, stride=1, save_im=True, save_path='D:/# Raras/GitHub/TA/groundtruth/res/test_res/seg1.jpg')
            seg_path_save = "D:/# Raras/src/makeup_dataset/final2/no_makeup_segs/" + str(img_name)
            seg_img.save(seg_path_save)
            
                # else:
                #     print("No face Detectedd")
            # except:
            #     img = Image.open(instance)
            #     img_path = "D:/# Raras/src/makeup_dataset/all/images/error/no_makeup/" + str(img_name)
            #     img.save(img_path)
            #     # print("gagal 1")
        else:
            # img = Image.open(instance)
            # img_path = "D:/# Raras/src/makeup_dataset/all/images/error/no_makeup/" + str(img_name)
            # img.save(img_path)
            # print("gagal 2")
            pass
    else:
        print("file not found")

