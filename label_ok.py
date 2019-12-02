import os
import sys
import glob
from skimage import io
import dlib

label_folder = 'combined2'

options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True
options.C = 5
options.num_threads = 4 # how many CPU cores
options.be_verbose = True

training_xml_path = os.path.join(label_folder, "training5.xml")

dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)

print("")  # Print blank line to create gap from previous output
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(training_xml_path, "detector.svm")))

detector = dlib.simple_object_detector("detector.svm")

win_det = dlib.image_window()
win_det.set_image(detector)

print("Showing detections on the images in the label folder...")
win = dlib.image_window()
for f in glob.glob(os.path.join(label_folder, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)
    dets = detector(img)
    print("Etiket sayısı: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()