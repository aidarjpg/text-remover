import os, io
from google.cloud import vision
import cv2
import numpy as np
from matplotlib import pyplot as plt

SPACE = 10
KERNEL_SIZE = 3
INPAINTING_RADIUS = 3

def show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)

def add_noise(img):
    x, y, ch = img.shape
    gauss_noise=np.zeros((x, y, ch),dtype=np.uint8)
    cv2.randn(gauss_noise, 128, 10)
    gauss_noise=(gauss_noise*0.8).astype(np.uint8)
    return cv2.add(img, gauss_noise)

def median_filter(img, filter_size):
    temp = []
    indexer = filter_size // 2
    img_final = []
    img_final = np.zeros((len(img),len(img[0])))
    for i in range(len(img)): # iterate through width
        for j in range(len(img[0])): # iterate through height 
            for z in range(filter_size): # check for out of bounds 
                if i + z - indexer < 0 or i + z - indexer > len(img) - 1: # i + z - indexer puts the current pixel into the center
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(img[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(img[i + z - indexer][j + k - indexer])
            temp.sort() # list the pixels
            img_final[i][j] = temp[len(temp) // 2] # assign the median
            temp = []
    return img_final

def inpaint(img_hsv, vertices_y, vertices_x, space, inpainting_radius):
    start_y = abs(min(vertices_y) - space)
    end_y = max(vertices_y) + space
    start_x = abs(min(vertices_x) - space)
    end_x = max(vertices_x) + space

    part_hsv = img_hsv[
        start_y:end_y,
        start_x:end_x
    ]

    # Get Hue channel
    a_channel = part_hsv[:,:,1]

    # Automate threshold using Otsu method
    th = cv2.threshold(a_channel, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Replace the part 
    img_hsv[start_y:end_y, start_x:end_x] = cv2.inpaint(part_hsv, th, inpainting_radius, cv2.INPAINT_TELEA)
    return img_hsv

def smoothing(img_hsv, vertices_y, vertices_x, space, kernel_size):
    spaced_start_y = abs(min(vertices_y) - space)
    spaced_end_y = max(vertices_y) + space
    spaced_start_x = abs(min(vertices_x) - space)
    spaced_end_x = max(vertices_x) + space

    to_blur_part = img_hsv[spaced_start_y:spaced_end_y, spaced_start_x:spaced_end_x]
    img_hsv[spaced_start_y:spaced_end_y, spaced_start_x:spaced_end_x] = cv2.GaussianBlur(to_blur_part, (kernel_size,kernel_size), 0)
    return img_hsv

def get_texts(path):
    # Get the client API instance
    client = vision.ImageAnnotatorClient()

    # Read the image from the file system
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    # Create the Image class instance from the provided image
    image = vision.Image(content=content)

    # Send text detection request to the API and get a response
    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    
    return response.text_annotations

def remove_text(img, texts, denoise, eq):
    if denoise:
        img = cv2.medianBlur(img, 3) 
        show(img)

    if eq:
        # convert from RGB color-space to YCrCb
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        # equalize the histogram of the Y channel
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

        # convert back to RGB color-space from YCrCb
        img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        show(img)

    # Convert to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Cycle through text annotations and print
    # the detected text and its boundaries
    for text in texts:
        if text == texts[0]:
            continue

        # Print detected text
        print(text.description)

        vertices_y = []
        vertices_x = []
        for vertex in text.bounding_poly.vertices:
            vertices_y.append(vertex.y)
            vertices_x.append(vertex.x)

        inpaint(img_hsv, vertices_y, vertices_x, SPACE, INPAINTING_RADIUS)

        smoothing(img_hsv, vertices_y, vertices_x, SPACE, KERNEL_SIZE)

    # Convert back to RGB and return
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

def process(path, denoise, eq):
    img = cv2.imread(path)
    show(img)
    texts = get_texts(path)
    img = remove_text(img, texts, denoise, eq)
    show(img)
    cv2.imwrite("removed_" + path, img)

# Set the environmental variable to the authentication token
# provided for individual Google Cloud account
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'token.json'

process('noise.png', True, True)
process('abbey.png', True, True)
process('car_wash.png', True, True)