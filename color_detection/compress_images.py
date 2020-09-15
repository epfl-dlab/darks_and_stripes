import cv2
from pathlib import Path
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')
    parser.add_argument('--folder', type=str, required=True)

    args = parser.parse_args()
    img_list = Path(args.folder).glob('*.jpg')
    
    for img_file in img_list:
        img = str(img_file)
        try:
            oriImg = cv2.imread(img)  # B,G,R order
            while (oriImg.shape[0] > 1000 and oriImg.shape[1] > 1000):
                oriImg = cv2.resize(oriImg, (0,0), fx=0.5, fy=0.5) 
                #cv2.imwrite(img.replace('.jpg', '_comp.jpg'), oriImg)
                cv2.imwrite(img, oriImg)
        except AttributeError:
            print("Problem with", img)
            
    