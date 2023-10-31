import cv2
import mediapipe as mp
import numpy as np
import time
import math

mp_face_mesh = mp.solutions.face_mesh


RIGHT_EYE_CONTOURS=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
RIGHT_IRIS_POLYGON = [468, 469, 470, 471, 472]
RIGHT_IRIS_CENTER = 468
HEAD_POINTS = [33, 263, 1, 61, 291, 199] # 2 eye outer corners + nose + 2 mouth outer corners + chain 
NOSE_INDX = 1

def euclidean_distance(point1, point2):
    x1, y1 =point1[:2]
    x2, y2 =point2[:2]
    return (math.sqrt((x2-x1)**2 + (y2-y1)**2))

def iris_h_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist/total_distance
    iris_position =""
    if ratio <= 0.42:
        iris_position="right"
    elif ratio > 0.42 and ratio <= 0.57:
        iris_position="center"
    else:
        iris_position = "left"
    return iris_position, ratio

def iris_tracking (face_2d):
    
    
    right_iris_center = (face_2d[468][:2])

    # Right eye horizontal points
    h_point1 =(face_2d[33][:2]) # right eye right corner 33
    h_point2 =(face_2d[173][:2]) # right eye left corner 133


    iris_h_pose, h_ratio=iris_h_position(right_iris_center, h_point1, h_point2)

    return iris_h_pose,h_ratio

def gaze_direction (cap):
    face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
    old_time=time.time()
    img_timer=time.time()
    initial_nose_2d=(0,0)
    initial_angels=(0,0)
    zero_time = time.time() 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_20 = cv2.VideoWriter('video\output20.avi', fourcc, 20.0, (1280, 720))

    imgs = []
    while True:
        start = time.time()
        success, frame = cap.read()
        if time.time() - img_timer >= 1:
            imgs.append(frame)
            img_timer=time.time()
            #cv2.imwrite(str('img\frame' + img_num), frame)
            #img_num + 1
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break
        #
        
        # To improve performance, optionally mark the frame as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        frame = cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_BGR2RGB)
        #frame = cv2.cvtColor( frame,cv2.COLOR_BGR2RGB)

        results = face_mesh.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_h, frame_w, _ = frame.shape
        points_3d = []
        points_2d = []
        face_2d=[]
    

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0] # capture 1st face
            for indx, landmark in enumerate(face_landmarks.landmark):
                x, y = (landmark.x*frame_w), (landmark.y*frame_h)
                face_2d.append([x,y])
                if indx in HEAD_POINTS:
                # if indx== 33 or indx==263 or indx==1 or indx==61 or indx==291 or indx==199:
                    # Nose landmark index = 1
                    if indx == NOSE_INDX :
                        nose_2d = (landmark.x*frame_w,landmark.y*frame_h)
                        nose_3d = (landmark.x*frame_w,landmark.y*frame_h,landmark.z*50+3100)
                    #x, y = int(landmark.x*frame_w), int(landmark.y*frame_h)
                    
                    #Get the 2D coordinates
                    points_2d.append([x,y])
                    #Get the 2D coordinates
                    points_3d.append([x,y,landmark.z])

                #print (face_2d)

            #Convert it to a Numpy array
            points_2d= np.array(points_2d, dtype=np.float64)
            #Convert it to a Numpy array
            points_3d= np.array(points_3d, dtype=np.float64)
            

            
            # The camera matrix 3x3
            # focal_lenth = screen_w
            focal_length = 1280

            cam_matrix = np.array([ [focal_length,0,frame_w/2],
                                   [0,focal_length,frame_h/2],
                                   [0,0,1] ])

            # The distortion parameters assumes no distortion
            distor_matx= np.zeros((4,1), dtype=np.float64)
        
            #Solve PnP
            ret , rot_vec, trans_vec =cv2.solvePnP(points_3d, points_2d, cam_matrix, distor_matx,flags=cv2.SOLVEPNP_EPNP)
            
            #Get rotaion martix
            rot_matx , _ = cv2.Rodrigues(rot_vec)

            #Get the angels
            angles, matxR, matxQ, Qx, Qy, Qz =cv2.RQDecomp3x3(rot_matx)
  
            #Get the rotation in degrees
            x=angles[0]*360
            y=angles[1]*360
            z=angles[2]*360
            
            #Infere the user's head direction
            if y < -5:
                out_txt="Left"
            elif y > 5 :
                out_txt="Right"
            elif x < 0.3 :
                out_txt="Down"
            elif x > 6.3 :
                out_txt="Up"     
            else:
                out_txt="Direct"          
            
            current_time= time.time()

            #calibration
            
            if (int(current_time-zero_time)) < 5 : 
                cv2.circle(frame,(int(nose_2d[0]), int(nose_2d[1])), 10,(0,255,0),3)

               
            # Display the nose direction
            nose_3d_projection, _ =cv2.projectPoints(nose_3d, rot_vec, trans_vec,cam_matrix,
                                                            distor_matx)

            # flatten the matrix to 1 D
            nose_3d_projection=np.matrix.flatten(nose_3d_projection)
            

            iris_h_pose,h_ratio=iris_tracking(face_2d)
            # p2= (min(int(nose_3d_projection[0] +((y)+(h_ratio/(y+.5))) * 80),frame_w), min(int(nose_3d_projection[1] 
            #                                                                      - ((x-3)) * 60),frame_h)) # Extend the nose vector when tilting
            if iris_h_pose =="left":
                p2= (min(int(nose_3d_projection[0] +((y)+(h_ratio/(y+.1))) * 90),frame_w), min(int(nose_3d_projection[1] 
                                                                                - ((x-3.3)) * 60),frame_h)) # Extend the nose vector when tilting
            elif iris_h_pose =="right":
                p2= (min(int(nose_3d_projection[0] +((y)-(h_ratio/(y+.1))) * 90),frame_w), min(int(nose_3d_projection[1] 
                                                                                - ((x-3.3)) * 65),frame_h)) # Extend the nose vector when tilting
            else: p2 = (min(int(nose_3d_projection[0] +(y) * 90),frame_w), min(int(nose_3d_projection[1] 
                                                                                - (x-3.3) * 65),frame_h)) # Extend the nose vector when tilting
    
            #cv2.line(frame, p1, p2, (255,0,0), 3)
            cv2.circle(frame,p2, 3,(255,0,0), 4)
            
            #Add the text on the image
            
            cv2.putText(frame, out_txt,(20,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3 )
          
            cv2.putText(frame, "x: Pitch " + str(np.round(x,2)), (500,50), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 2 )
           
            cv2.putText(frame, "y: Yaw " + str(np.round(y,2)), (500,100), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 2 )
           
            #cv2.putText(frame, "z: Roll " + str(np.round(z,2)), (500,150), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 2 )
            
        end = time.time()
        
        totalTime = end - start
       
        fps = 1 / totalTime

        cv2.putText(frame, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),2)
        
        cv2.imshow('Head Pose Estimation', frame)

        out_20.write(frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()

    return imgs

# cap = cv2.VideoCapture(1)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# gaze_direction(cap)    