import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys
from glob import iglob
from scipy.ndimage import label

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
PRE_IMAGE_WIDTH = 45
PRE_IMAGE_HEIGHT = 45
IMAGE_PADDING = int((IMAGE_WIDTH - PRE_IMAGE_WIDTH)/2)
#IMAGE_OUTPUT_DIR = "../test/test_images"
IMAGE_OUTPUT_DIR = "../test_images"

IMAGE = "./NOSTest/KakaoTalk_20190625_104903457.jpg"
#IMAGE = "./sample_lee_equl.jpg"
#IMAGE = sys.argv[1]


def show(thing):
    plt.imshow(thing)
    plt.show()


#이미지 읽기
def read_image(image):
    image_gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE) #이미지 등록
    image_gray2 = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    ret, img_th = cv2.threshold(image_gray, 127, 225, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #img_th2 = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img_file = (int(img_th[0][0]) + int(img_th[0][-1]) + int(img_th[-1][0]) + int(img_th[-1][-1])) / 4
    #print(img_file)

    show(img_th)

    if img_file >= 100:
        ret, img_th = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    show(img_th)
    return img_th, image_gray, image_gray2

"""
#opening
kernel2 = np.ones((3, 3), np.uint8)
img_erode = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel2, iterations=2)

plt.imshow(img_erode)
plt.show()
"""
#추가

#이미지 기울기 보정
def skew_correction(image_th):
    coords = np.column_stack(np.where(image_th > 0))
    angle = cv2.minAreaRect(coords)[- 1]

    if angle < - 45:
        angle = - (90 + angle)
    else:
        angle = - angle

    (h, w) = image_th.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D (center, angle, 1.0)
    rotated_gray = cv2.warpAffine(image_th, M, (w, h), flags = cv2.INTER_CUBIC, borderMode = cv2. BORDER_REPLICATE)


    show(rotated_gray)
    #cv2.putText(rotated_gray, "Angle: {:.2f} degrees".format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return rotated_gray


def text_area_detection(rotate_image, image_gray, image_gray2):
    #팽창
    kernel3 = np.ones((1, 30), np.uint8)
    img_erode3 = cv2.dilate(rotate_image, kernel3, iterations = 10)

    show(img_erode3)

    im3, contours2, hierarchy = cv2.findContours(img_erode3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #print(contours2)
    contour2 = contours2[0]
    max_rect = cv2.boundingRect(contour2)
    total_rects = [cv2.boundingRect(each) for each in contours2]
    for rect in total_rects:
        print(rect)
        size1 = rect[2] * rect[3]
        size2 = max_rect[2] * max_rect[3]
        if size1 > size2:
            max_rect = rect

    total_rect = max_rect
    print(total_rect)

    #plt.imshow(image_gray)
    #plt.show()

    img_total = rotate_image.copy()
    img_total2 = image_gray2.copy()
    img_total3 = image_gray2.copy()
    total_text = img_total[total_rect[1] : total_rect[1] + total_rect[3], total_rect[0] : total_rect[0] + total_rect[2]]
    gray_total = img_total2[total_rect[1] : total_rect[1] + total_rect[3], total_rect[0] : total_rect[0] + total_rect[2]]
    gray_total2 = img_total3[total_rect[1] : total_rect[1] + total_rect[3], total_rect[0] : total_rect[0] + total_rect[2]]
    print(total_text)

    show(total_text)
    return total_text, gray_total



def separate_combine_characters(boxes):
    add_width = 0
    add_height = 0
    min_box = boxes[0][2]
    max_box = boxes[0][2]
    # print(boxes[0])
    boxes_sort = sorted(boxes)
    # print(boxes_sort)

    for box in boxes:
        add_width += box[2]
        add_height += box[3]

        if box[2] < min_box:
            min_box = box[2]
        elif box[2] > max_box:
            max_box = box[2]
        # print(add_width, add_height)
        print(min_box, max_box)

    avg_width_min = (add_width - min_box) / (len(boxes) - 1)
    avg_width_max = (add_width - max_box) / (len(boxes) - 1)
    avg_width = add_width / len(boxes)
    avg_height = add_height / len(boxes)

    merge_box = []
    for i in range(0, len(boxes_sort)):
        x, y, width, height = boxes_sort[i]
        x2, y2, width2, height2 = boxes_sort[i - 1]

        # 모음만 잡히면 얇다 - 너비
        if width < (avg_width_min * 0.6):
            print("no width: ")
            print(boxes_sort[i])

            if i != len(boxes_sort) - 1:
                x3, y3, width3, height3 = boxes_sort[i + 1]
                if width3 < (avg_width_min * 0.6):
                    merge_box.append(boxes_sort[i])
                    print(merge_box)
                else:
                    merge_box[-1] = (x2, min(y, y2), x - x2 + width, max(height, height2))
                    print(merge_box)
            else:
                merge_box[-1] = (x2, min(y, y2), x - x2 + width, max(height, height2))
                print(merge_box)

        elif (width > (avg_width_max * 1.4)) and (width > (height * 1.4)):
            print("big width: ")
            merge_box.append((x, y, int(width / 2), height))
            merge_box.append((x + int(width / 2), y, int(width / 2), height))
            print(merge_box)

        else:
            print("avg width: ")
            print(boxes_sort[i])
            merge_box.append(boxes_sort[i])
            print(merge_box)


        return merge_box


def character_detection(text_area, text_area_gray, show_image):
    #팽창
    kernel = np.ones((30, 1), np.uint8) #15, 3,
    img_erode = cv2.dilate(text_area, kernel, iterations=5) #iter = 1

    """
    #침식
    kernel = np.ones((3, 3), np.uint16) #15, 3,
    img_erode2 = cv2.erode(img_th, kernel, iterations=1) #iter = 1
    """
    show(img_erode)

    #im3, contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im3, contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    rects = [cv2.boundingRect(each) for each in contours]
    tmp = [w*h for (x, y, w, h) in rects]
    tmp.sort()
    #print(tmp)
    add_area = 0
    for add_tmp in tmp:
        add_area += add_tmp

    avg_area = add_area/len(tmp)/5
    #print(avg_area)

    boxes = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        #cv2.rectangle(image_file, (x, y), (x + width, y + height), (0, 0, 250), 2)

        area = width * height

        #if area > 1000 and area < 10000:
        if area > avg_area:
            cv2.rectangle(text_area_gray, (x, y), (x + width, y + height), (0, 0, 255), 2)
            #rects = [cv2.boundingRect(each) for each in contours]
            boxes.append((x, y, width, height))

    show(text_area_gray)


    #print(boxes)

    merge_box = separate_combine_characters(boxes)

    """
        #모음만 잡히면 얇다 - 길이
        if height < (avg_height * 0.7):
            print('no height: ')
            print(boxes_sort[i])
            merge-box
   """

    print(merge_box)
    for i in range(0, len(merge_box)):
        cv2.rectangle(show_image, (merge_box[i][0], merge_box[i][1]), (merge_box[i][0] + merge_box[i][2], merge_box[i][1] + merge_box[i][3]), (0, 0, 250), 2)

    show(show_image)

    return merge_box



def separated_img(text_area, boxes):
    img_output = []
    #img_point = img_th.copy() #복사(주소)
    #img_point = img_erode2.copy() #복사(주소)
    img_point = text_area.copy()

    for rect in boxes:
        img_output.append(
            img_point[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]])


    """
    for image in img_output:
        plt.imshow(image)
        plt.show()
    """

    i=0
    pre_img_output = []
    for n in img_output:
        #이미지의 크기 55*55로 변경
        pre_img_output.append(cv2.resize(img_output[i], (PRE_IMAGE_WIDTH, PRE_IMAGE_HEIGHT)))
        #cv2.imwrite("./test_images/test_image_{}.jpeg".format(i), pre_img_output[i])
        img_output[i] = np.lib.pad(pre_img_output[i], (IMAGE_PADDING, IMAGE_PADDING), 'constant')
        if len(img_output[i]) is not IMAGE_WIDTH:
            print(len(img_output[i]))
            img_output[i] = cv2.resize(img_output[i], (IMAGE_WIDTH, IMAGE_HEIGHT))

        #cv2.imwrite(IMAGE_OUTPUT_DIR + "/test_image_{}.jpeg".format(i+104), img_output[i])
        i += 1


    for image in img_output:
        show(image)



if __name__ == "__main__":
    img_th, gray_image, gray_image2 = read_image(IMAGE)
    rotate_image = skew_correction(img_th)
    text_area, text_area_gray = text_area_detection(rotate_image, gray_image, gray_image2)

    merge_box = character_detection(text_area, text_area_gray, gray_image2)
    separated_img(text_area, merge_box)
