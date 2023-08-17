import cv2
import numpy as np
def gethog(img):
    s = (128,128)
    new_img = cv2.resize(img, s, interpolation=cv2.INTER_AREA)
    win_size =  new_img.shape
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
    cell_size, num_bins)
    hog_descriptor = hog.compute(new_img)
    hog_descriptor_list = hog_descriptor.flatten().tolist()
    # print ('HOG Descriptor:', hog_descriptor)
    return hog_descriptor_list