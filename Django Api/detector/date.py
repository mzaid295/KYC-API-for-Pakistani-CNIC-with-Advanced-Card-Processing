import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mash = mp.solutions.face_mesh
face_mesh = mp_face_mash.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)

cap = cv2.VideoCapture(0)
# video_path = "1024544360-preview.mp4"
# cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, image = cap.read()
    start = time.time()
    #
    #
    image = cv2.cvtColor(cv2.flip(image, 1),cv2.COLOR_BGR2GRAY)

    #To improve Performance
    image.flags.writeable = False
    results = face_mesh.process(image)

    #To improve performance
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        node_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    #Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 2D Coordinates
                    face_3d.append([x, y, lm.z])

            #Convert into the Numpy Array
            face_2d = np.array(face_2d, dtype = np.float64)

            #Convert into the Numpy Array
            face_3d = np.array(face_3d, dtype = np.float64)

            #The Camera Matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([focal_length, 0, img_h / 2],
                                  [0, focal_length, img_w / 2],
                                  [0, 0, 1])
            #The distortion matrix
            dist_matrix = np.zeros((4,1), dtype = np.float64)

            #Solve
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            #Get rotational Matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            #Get Angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            #Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            #See where the user's head tilting
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x <- 10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            #Display The Nose Direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 =(int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            cv2.line(image, p1, p2 (255, 0, 0), 3)

            #Add the text on the image
            cv2.putText(image, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255))
            cv2.putText(image, "y: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255))
            cv2.putText(image, "z: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255))

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            print("FPS: ", fps)

            cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255,0),2)
            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = face_landmarks,
                connections = mp_face_mash.Face_CONNECTIONS,
                landmark_drawing_spec = drawing_spec,
                connection_drawing_spec = drawing_spec,)
            cv2.imshow('Head Pose Detection', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

            cap.release()