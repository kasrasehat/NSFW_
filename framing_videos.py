import cv2
import os
import tqdm
path1 = '/home/sehat/Downloads/all_violence_data'
#path2 = 'Data/VIS_Onshore/HorizonGT'
store_path = '/home/sehat/Downloads/frames'
if not os.path.exists(store_path):
    os.makedirs(store_path)
files = os.listdir(path1)
#horizon_data = os.listdir(path2)
#file = [value[:-14] for value in horizon_data]

frameNr = 0
for i in tqdm.tqdm(range(len(files))):

    #if video[:-4] in file:
    video = files[i]
    capture = cv2.VideoCapture(path1 + '/' + video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    amountOfFrames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    success = True

    while (success):

        success, frame = capture.read()

        if success and frameNr % 10 == 0:
            cv2.imwrite('{}/frame_{}.jpg'.format(store_path, frameNr), frame)


        frameNr = frameNr + 1

capture.release()

