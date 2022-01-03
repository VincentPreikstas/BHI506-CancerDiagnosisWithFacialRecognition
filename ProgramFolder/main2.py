import cv2
import os
import numpy

people = ["", "No Cancer", "Cancer"]
training_photos_path = "training_photos2"
test_photos_path = "test_photos2"

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    img_copy = colored_img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img_copy

def detect_face_multi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        print("Failed")
        test = "test"
        cv2.imshow("This Image Failed", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(training_photos_path):
    dirs = os.listdir(training_photos_path)
    faces = []
    labels = []
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')

    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue;
        label = int(dir_name.replace("s", ""))
        subject_dir_path = training_photos_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue;
            image_path = subject_dir_path + "/" +  image_name
            image = cv2.imread(image_path)
            show_image = detect_faces(face_cascade, image, 1.2)

            cv2.imshow("Training on image...", show_image)
            cv2.waitKey(0) #was 200
            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
    #cv2.destroyAllWindows()
    #cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels



def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    label, confidence = face_recognizer.predict(face)
    label_text = people[label]

    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1] - 5)
    return img

def predict_multi(test_img):
    img = test_img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rect_list = detect_face_multi(img)
    for face in rect_list:
        (x, y, w, h) = face
        label, confidence = face_recognizer.predict(gray[y:y+w, x:x+h])
        label_text = people[label]

        draw_rectangle(img, face)
        draw_text(img, label_text, face[0], face[1] - 5)
    return img

def predict_function(test_img):
    predicted_img = predict_multi(test_img)
    cv2.imshow("Prediction complete", predicted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_test_images(directory_path):
    images = os.listdir(directory_path)
    image_list = []
    for images_name in images:
        image_path = directory_path + "/" + images_name
        image = cv2.imread(image_path)
        image_list.append(image)
    return image_list

def process_test_images(directory_path):

    image_list = load_test_images(directory_path)
    #print(len(image_list))
    for image in image_list:
        if image is not None:
            cv2.imshow("Showing Test Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            predict_function(image)

faces, labels = prepare_training_data(training_photos_path)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, numpy.array(labels))

process_test_images(test_photos_path)