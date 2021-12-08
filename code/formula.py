import os, cv2, random, copy, imutils

import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
#from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.segmentation import clear_border, watershed
from skimage import measure, color, io


# Image Directory Generator
def getImgPath(folder_list, Sequences_path, n):
    # give the image list in n sequence
    imageList = os.listdir(Sequences_path + '/' + folder_list[n])
    # Sort the list
    imageList = [(int(i[1:-4]), i) for i in imageList]
    imageList = sorted(imageList, key=lambda x: x[0])
    # generate complete image list
    imageList = [Sequences_path + '/' + folder_list[n] + '/' + i[1] for i in imageList]
    return imageList


def getImgPath2(folder_list, More_data_path, n):
    # give the image list in n sequence
    imageList = os.listdir(More_data_path + '/' + folder_list[n] + '/' + 'SEG')
    # Sort the list
    imageList = [(int(i[7:-4]), i) for i in imageList]
    imageList = sorted(imageList, key=lambda x: x[0])
    # generate complete image list
    imageList = [More_data_path + '/' + folder_list[n] + '/' + 'SEG' + '/' + i[1] for i in imageList]
    return imageList


def getImgPath3(folder_list, More_data_path, n):
    # give the image list in n sequence
    imageList = os.listdir(More_data_path + '/' + folder_list[n])
    # Sort the list
    imageList = [(int(i[1:-4]), i) for i in imageList]
    imageList = sorted(imageList, key=lambda x: x[0])
    # generate complete image list
    imageList = [More_data_path + '/' + folder_list[n] + '/' + i[1] for i in imageList]
    return imageList


# uint16 to uint8 Convertor
def uint16_to_uint8(img_path):
    img = cv2.imread(img_path, -1)

    img_uint8 = img / 65536 * 256
    return img_uint8.astype('uint8')


# Contrast Stretch
def contrast_stretch(img_path):
    # uint16 to uint8
    img_gray = cv2.imread(img_path, -1)
    
    a, b, c, d = 0, 65535, np.min(img_gray), np.max(img_gray)

    img_gray = (img_gray - c) * ((b - a) / (d - c)) + a
    img_gray = img_gray.astype('uint16')

    img_gray = img_gray / 65536 * 256

    img_gray = img_gray.astype('uint8')

    return img_gray


# histogram (including GaussianBlur)
def whole_hist(image):
    return np.argmax(np.bincount(image.ravel().astype('uint8'))) + 3


# Binary thresholding
def binary_thresh(img_o, thresh):
    # return _, mask_otsu
    blur = cv2.GaussianBlur(img_o, (7, 7), cv2.BORDER_DEFAULT)
    _, mask_otsu = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
    return mask_otsu


def ero_dilation(mask_otsu):
    kernel = np.ones((3, 3), np.uint8)
    # Erosion combine with Dilation
    image_ero = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel, iterations=4)
    # Clear the cells that are on the border
    image_ero = clear_border(image_ero)
    return image_ero


def erosion(img, n):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(img, kernel, iterations=n)


# Watershed
# Oversegmentation, not adopted
def w_labels(img):
    img = img.astype('uint8')
    D = ndi.distance_transform_edt(img)
    localMax = peak_local_max(D, indices=False, min_distance=10, labels=img)
    markers = ndi.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=img)

    return labels


# Marker-based watershed
# Manually create markers instead using local maxima
def ws_labels(img):
    # img = img.astype('uint8')
    kernel = np.ones((4, 4), np.uint8)
    sure_bg = cv2.dilate(img, kernel, iterations=7)
    # Using erosion may erase small cells
    percent = 0.4
    dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, percent * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    # Unknown should labled as zero not background
    markers = markers + 1
    markers[unknown == 255] = 0
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    labels = cv2.watershed(img, markers)
    labels = labels.astype('uint8')
    return labels


# FindContours
def find_contours(labels, image):
    # return img_label, contours, hierarchy
    contours = []
    num0 = 0
    num1 = -1
    num2 = 0
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw it on the mask
        mask = np.zeros(image.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        img0, cnts, hieracy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours((img0, cnts, hieracy))

        contours.append(cnts)

    return contours[1:-1]  # get rid of double centre


# Check if cell area > 130
def area_checker(contours):
    area, contours1 = [], []
    for i in contours:
        if cv2.contourArea(i[0]) > 120:
            contours1.append(i[0])
    return contours1


# Area Calculator
def area_calculator(i):
    return cv2.contourArea(i)


# Equi diameter Calculator
def diameter_calulator(area):
    return np.sqrt(4 * area / np.pi)


# Intersection Calculator c = [contour1, contour2]
def get_coordinate(cnt): # contours1[0]
    a = np.squeeze(cnt)
    a = [list(i) for i in list(a)]

    coordinate = []
    a_new = sorted(list(a), key = lambda x:x[0])
    index = sorted(list(set(i[0] for i in a_new)))

    for i in index:
        temp_1 = sorted([j[1] for j in a_new if j[0] == i])

        if len(temp_1) == 1:
            continue

        temp_2 = list(range(temp_1[0], temp_1[-1] + 1))
        result = [(i, j) for j in temp_2 if j not in temp_1]
        coordinate += result

    return coordinate


def intensity(image, coordinate): # original image & coordinate list
    inten = 0
    count = 0
    for i in coordinate:
        count += 1
        inten += image[i[1]][i[0]]

    return inten / count


def overlapping(image, c):
    temp_1 = get_coordinate(c[0])
    temp_2 = get_coordinate(c[1])

    return len([i for i in temp_1 if i in temp_2])


# Color generator for starter (random)
def color_generator_1(contours1):
    color_L = []
    while True:
        R = random.randint(0, 255)
        G = random.randint(0, 255)
        B = random.randint(0, 255)
        color = (R, G, B)
        if color not in color_L:
            color_L.append(color)
        if len(color_L) == len(contours1):
            break
    return color_L


# Color generator for all new cells, add 1 unique color when used
def color_generator_2(color_L):
    while True:
        R = random.randint(0, 255)
        G = random.randint(0, 255)
        B = random.randint(0, 255)
        color = (R, G, B)
        if color not in color_L:
            color_L.append(color)
            break
    return color


# Cell painter
def draw_cells(img_o, color_L, contours1):
    img_o = cv2.cvtColor(img_o, cv2.COLOR_GRAY2RGB)

    for index in range(len(contours1)):
        draw_1 = cv2.drawContours(img_o, contours1, index, color_L[index], 2)

    return draw_1


# Centre of Cell [(cX1, cY1)(cX2, cY2)...(cXn, cYn)]
def cell_centre(contours1):
    centreList = []
    for i in range(len(contours1)):
        M = cv2.moments(contours1[i])
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        centreList.append((cX, cY))
    return centreList
    

def labelling(kyes_list, centre_L, draw_1):
    for i in range(len(kyes_list)):
        draw_2 = cv2.putText(draw_1, kyes_list[i], (centre_L[i][0] - 22, centre_L[i][1]), 1, 1.4, (255, 255, 255), 2)
    return draw_2


# Draw track
def draw_lines(draw_2, track_L, color_L):
    for i in range(len(track_L)):
        if len(track_L[i]) > 1:
            for x in range(len(track_L[i]) - 1):
                draw_2 = cv2.line(draw_2, track_L[i][x], track_L[i][x + 1], color_L[i], 1)
    return draw_2

# Put average size and displacement
def put_info(draw_2, area, disp, count, mitosis_count):
    area = "{:.2f}".format(area)
    disp = "{:.2f}".format(disp)
    border = cv2.copyMakeBorder(draw_2,160,0,0,0,borderType=cv2.BORDER_CONSTANT,value=[255, 255, 255])
    info_1 = 'Cell count: '+str(count)
    info_2 = 'Average area: '+str(area)
    info_3 = 'Average displacement: '+str(disp)
    mitosis = 'Number of dividing cells: '+ str(mitosis_count)
    result = cv2.putText(border,info_1,(0,30), 1, 2, (0, 0, 0), 2,cv2.LINE_AA)
    result = cv2.putText(border,info_2,(0,70), 1, 2, (0, 0, 0), 2,cv2.LINE_AA)
    result = cv2.putText(border,info_3,(0,110), 1, 2, (0, 0, 0), 2,cv2.LINE_AA)
    result = cv2.putText(border,mitosis,(0,150), 1, 2, (0, 0, 0), 2,cv2.LINE_AA)
    return result


def starter_1(img_path):
    image = contrast_stretch(img_path)  # 1
    thresh = whole_hist(image)  # 2
    mask_otsu = binary_thresh(image, thresh)  # 3
    image_ero = ero_dilation(mask_otsu)  # 4

    return image, image_ero


def starter_2(image, image_ero):
    labels = w_labels(image_ero)  # 5
    contours = find_contours(labels, image_ero)  # 6
    contours1 = area_checker(contours)  # 7
    areas = [area_calculator(i) for i in contours1]
    centreList = cell_centre(contours1)  # Record cell centre

    return centreList, contours1, areas
