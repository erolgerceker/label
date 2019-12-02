import os
import dlib
import cv2

label_folder = 'combined4'

options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True
options.C = 5
options.num_threads = 4 # how many CPU cores
options.be_verbose = True

training_xml_path = os.path.join(label_folder, "training7.xml")

dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)

print("")  # Print blank line to create gap from previous output
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(training_xml_path, "detector.svm")))

detector = dlib.simple_object_detector("detector.svm")

cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 2
while True:
    ret_val, img = cam.read()
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_image)
    print("Etiket sayısı: {}".format(len(dets)))
    for det in dets:
        cv2.rectangle(img,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
    cv2.imshow('ETIKET', img)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()