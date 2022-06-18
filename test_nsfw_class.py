
import os
from nsfw_class import detect_nsfw
import cv2

ns = detect_nsfw()
image = cv2.imread('/home/sehat/dataset/single_test/test2 (14).jpg')
a,b = ns.sfw(input=image)
print(a)
print(b)