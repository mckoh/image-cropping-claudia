from matplotlib.image import imread
from matplotlib.pyplot import imshow
from os.path import join, isfile, isdir
from os import mkdir
from os import listdir
from numpy import array
import cv2

PATH = join("input_data")
LOWER_BOUND = (10, 100, 20)
UPPER_BOUND = (25, 255, 255)
MIN_CONTOUR_WIDTH = 500

def crop_images_to_circle(root_folder, output_root_folder):

    if not isdir(output_root_folder):
        mkdir(output_root_folder)

    for folder in listdir(root_folder):

        output_folder = join(output_root_folder, folder)
        if not isdir(output_folder):
            mkdir(output_folder)

        folder_path = join(root_folder, folder)

        for element in listdir(folder_path):
            if element.endswith(".jpg"):
                image_path = join(folder_path, element)

                # read the image in BGR mode (default with cv2)
                image = cv2.imread(image_path)

                # transform the image into RGB for proper display with matplotlib
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                # find orange are in the image
                mask = cv2.inRange(hsv, LOWER_BOUND, UPPER_BOUND)

                # find contours in the masked image
                contours, hierarchy = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                # Loop all contours and take large ones
                for j, contour in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > MIN_CONTOUR_WIDTH:
                        cropped_image = image[y:y+h, x:x+w]
                        cv2.imwrite(join(output_folder, f"{element[:-4]}__{j+1:003d}.jpg"), cropped_image)
    return True