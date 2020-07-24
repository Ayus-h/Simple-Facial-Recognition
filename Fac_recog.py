import face_recognition as fr
import os
import cv2

KNOWN_FACES_DIR = "Known pics"
UNKNOWN_FACES_DIR ="Unknown pics"
TOLERANCE = 0.45
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"

print("Loading known faces :")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = fr.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encoding = fr.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

print("processing unknown faces :")

for filename in os.listdir(UNKNOWN_FACES_DIR):
    print(filename)
    image = fr.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    locations =  fr.face_locations(image,model = MODEL)
    encodings = fr.face_encodings(image,locations)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    
    for face_encoding, face_location in zip(encodings,locations):
        results = fr.compare_faces(known_faces,face_encoding,TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found:{match}")
            color =[0,255,0]

        else:
            match = "Unknown"
            color =[200,200,200]



        top_left = (face_location[3],face_location[0]-75)
        bottom_right = (face_location[1],face_location[2])

        

        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            
        top_left = (face_location[3],face_location[2])
        bottom_right = (face_location[1],face_location[2]+22)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, match,(face_location[3]+10,face_location[2]+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),FONT_THICKNESS)

    cv2.namedWindow(filename,cv2.WINDOW_NORMAL)
    cv2.imshow(filename,image)
    cv2.waitKey(1000)


