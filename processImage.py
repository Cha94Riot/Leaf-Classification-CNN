import cv2

def processImage(file_path):
    image = open_image(file_path)
    cropped_image = crop_image(image)
    scaled_image = scale_image(cropped_image)

    return scaled_image


def open_image(file_path):
    image = cv2.imread(file_path)
    return image

def crop_image(original_image):
    grey_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    grey_image = cv2.GaussianBlur(grey_image, (7, 7), 0)

    thresh_image = cv2.adaptiveThreshold(grey_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    padded_image = pad_image(original_image)
    padded_thresh = pad_image(thresh_image)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(padded_thresh, cv2.MORPH_CLOSE, kernel)

    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_x = 0
    max_y = 0
    max_w = 0
    max_h = 0

    for c in cnts:

        x, y, w, h = cv2.boundingRect(c)

        area = w * h
        if area > max_area:
            max_x = x
            max_y = y
            max_w = w
            max_h = h
            max_area = area

    if (max_w < max_h):
        max_x = int(max_x - ((max_h - max_w) / 2))
        max_w = int(max_h)

    elif (max_w > max_h):
        max_y = int(max_y - ((max_w - max_h) / 2))
        max_h = int(max_w)

    cropped_image = padded_image[max_y:max_y + max_h, max_x:max_x + max_w]

    return cropped_image


def scale_image(image_in):
    return cv2.resize(image_in, (128, 128))

def pad_image(image_in):
    image_size = image_in.shape
    new_size = (max(image_size)) * 1.5

    delta_w = new_size - image_size[1]
    delta_h = new_size - image_size[0]
    top, bottom = int(delta_h // 2), int(delta_h - (delta_h // 2))
    left, right = int(delta_w // 2), int(delta_w - (delta_w // 2))

    color = [0, 0, 0]
    padded_image = cv2.copyMakeBorder(image_in, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return padded_image