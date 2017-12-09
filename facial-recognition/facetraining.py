# facetraining file to train the file
import cv2, sys, numpy, os
size = 4
frontal_face_haar = 'haarcascade_frontalface_default.xml'
faces_database_path = 'humanfaces'
try:
    face_name_setup = sys.argv[1]
except:
    print("Please provide your name!")
    sys.exit(0)
#set the path to setup the face name in the face database
path = os.path.join(faces_database_path, face_name_setup)

if not os.path.isdir(path):
    os.mkdir(path)
(im_image_width, im_image_height) = (100, 100)
haar_cascade = cv2.CascadeClassifier(frontal_face_haar)
camera = cv2.VideoCapture(0)

# Create the face name folder for images
insert_image_name=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
     if n[0]!='.' ]+[0])[-1] + 1

# Show the message to guide the user
print("\nThe program will automatically take 30 pictures and save them into humanfaces folder. \
Move your head around to let the camera take picture of you face with all angles.\n")

# The program loops until it has 30 images of the face.
count = 0
pause = 0
total_images = 30
while count < total_images:

    # While loop check if the 
    check_camera = False
    while(not check_camera):
        # Put the image from the camera into 'video_frame'
        (check_camera, video_frame) = camera.read()
        if(not check_camera):
            print("Failed to open camera. Trying again...")

    # Get all the image info
    image_height, image_width, features = video_frame.shape

    # Flip video_frame direction (from left to right)
    video_frame = cv2.flip(video_frame, 1, 0)

    # Convert color picture to grayscalescale mode
    grayscale = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)

    # Scale down image size for speed
    mini_img = cv2.resize(grayscale, (int(grayscale.shape[1] / size), int(grayscale.shape[0] / size)))

    # Detect faces with multiple scale using haar cascade method
    faces = haar_cascade.detectMultiScale(mini_img)

    # But we just focus on the largest one
    faces = sorted(faces, key=lambda x: x[3])
    if faces:
        face_i = faces[0]
        (x, y, w, h) = [v * size for v in face_i]

        face = grayscale[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_image_width, im_image_height))

        # Draw rectangle and write name on the the image or video stream
        cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(video_frame, face_name_setup, (x - 5, y - 5), cv2.FONT_ITALIC,
            1,(0, 0, 255))

        # Remove false positives
        if(w * 6 < image_width or h * 6 < image_height):
            print("The face is too small")
        else:

            # Detect the angle changes of the face and capture it
            if(pause == 0):

                print("The image "+str(count+1)+" is saved. ("+str(count+1)+"/"+str(total_images)+")")

                # Save all images as .png file
                cv2.imwrite('%s/%s.png' % (path, insert_image_name), face_resize)

                insert_image_name += 1
                count += 1

                pause = 1
	#using esc to escape
    if(pause > 0):
        pause = (pause + 1) % 5
    cv2.imshow('Face Training', video_frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
