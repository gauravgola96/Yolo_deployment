import cv2

def preprocess_image_data(image):
    data = cv2.resize(image, (416, 416))
    data = data / 255.
    data = data[:, :, ::-1]
    return data