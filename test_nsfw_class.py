
import os
from nsfw_3class import detect_violence
import cv2

ns = detect_violence()
image = cv2.imread('/home/sehat/dataset/single_test/test2 (14).jpg')
a, b = ns.sfw(input=image)
print(a)
print(b)