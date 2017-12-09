# facial_recognition file
import cv2, sys, numpy, os
size = 2
frontal_face_haar = 'haarcascade_frontalface_default.xml'
faces_database_path = 'humanfaces'

# Create an array list of images with names and labels
(images, lables, names, id) = ([], [], {}, 0)

# Go the the humanfaces folder to check the training images
for (image_paths, apaths, files) in os.walk(faces_database_path):

    # Go through all folder names to compare
    for image_path in apaths:
        names[id] = image_path
        subjectpath = os.path.join(faces_database_path, image_path)

        # Check every single image and compare
        for filename in os.listdir(subjectpath):

            # inorg all images formates
            f_name, f_extension = os.path.splitext(filename)
            if(f_extension.lower() not in
                    ['.png','.jpg','.jpeg','.gif','.pgm']):
                print("Skipping "+filename+", wrong file type")
                continue
            path = subjectpath + '/' + filename
            lable = id

            # Add it to training data
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(im_width, im_height) = (100, 100)

# Create a Numpy array from the two lists above
(images, lables) = [numpy.array(lis) for lis in [images, lables]]

# OpenCV trains a model from the images
model = cv2.face.FisherFaceRecognizer_create()
model.train(images, lables)

# Use fisherRecognizer on camera stream
haar_like_feature = cv2.CascadeClassifier(frontal_face_haar)
open_webcam = cv2.VideoCapture(0)
while True:

    # Loop until the camera is working
    check_webcam = False
    while(not check_webcam):
        # Put the image from the open_webcam into 'frame'
        (check_webcam, frame) = open_webcam.read()
        if(not check_webcam):
            print("Failed to open open_webcam. Trying again...")

    # Flip video_frame direction (from left to right)
    frame=cv2.flip(frame,1,0)

    # Convert to the color image to grayscale image
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the image to increase the speed of detection
    mini_image = cv2.resize(grayscale, (int(grayscale.shape[1] / size), int(grayscale.shape[0] / size)))

    # Detect faces using for loop
    faces = haar_like_feature.detectMultiScale(mini_image)
    for i in range(len(faces)):
        face_i = faces[i]

        # Coordinates of face after scaling back by `size`
        (x, y, w, h) = [v * size for v in face_i]
        face = grayscale[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))

        # Try to recognize the face using predict function in model
        prediction = model.predict(face_resize)
		
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # Write the predicted name on video stream
        cv2.putText(frame,
           '%s' % (names[prediction[0]]),
           (x-5, y-5), cv2.FONT_ITALIC,0.5,(0, 0, 255))

    # Show the image and wait for the "ESC" to exit
    cv2.imshow('Facial Recognition', frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
