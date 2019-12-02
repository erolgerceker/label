import cv2
import glob

path="C:/Users/erol.gerceker/PycharmProjects/sayac/sayac-images"

for imFile in glob.glob(path+"/*.jpg"):
    print(imFile)
    img = cv2.imread(imFile)
    img = cv2.resize(img,None,fx=0.2,fy=0.2)
    cv2.imwrite(imFile,img)
    cv2.waitKey(0)
'''
    image = cv2.imread('index.jpg')
    smaller_image = cv2.resize(image, (300, 300), inerpolation='linear')
    plt.imshow(smaller_image)
'''