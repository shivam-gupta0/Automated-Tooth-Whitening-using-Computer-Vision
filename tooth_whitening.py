import numpy as np
import cv2

images = ["98131650599781345.jpeg", "97281656150603693.jpeg", "92941642517882289.jpeg", "90771650991876562.jpeg",
          "90491644539985109.jpeg", "90431646980152686.jpeg", "81471647469196045.jpeg", "72481654091972855.jpeg",
          "66981650755750147.jpeg", "52001650756788238.jpeg", "81511652514777874.jpeg", "80841652332768762.jpeg"]

# Load the mouth cascade classifier
mouth_cascade = cv2.CascadeClassifier("Mouth.xml")

def get_largest_contour(contours):
    """
    Find the largest contour from a list of contours
    """
    largest_contour_area = 0
    largest_contour = None

    for contour in contours:
        contour_area = cv2.contourArea(contour)

        if contour_area > largest_contour_area:
            largest_contour_area = contour_area
            largest_contour = contour

    return largest_contour

def get_largest_mouth_region(mouths):
    """
    Get the largest mouth region from a list of detected mouths
    """
    largest_region = 0
    largest_mouth = None

    for mouth in mouths:
        mx, my, mw, mh = mouth
        area = (mx + mw) * (my + mh)

        if area > largest_region:
            largest_region = area
            largest_mouth = mouth

    return largest_mouth


def first_layer_segmentation(mouth_section):
    """
    Apply the first layer segmentation to the mouth section
    """
    teeth_image = cv2.cvtColor(mouth_section, cv2.COLOR_RGB2HSV)

    # Color mask for lips
    lower = np.array([110, 40, 0])
    upper = np.array([255, 201, 255])
    mask = cv2.inRange(teeth_image, lower, upper)
    imageResult = cv2.bitwise_and(teeth_image, teeth_image, mask=mask)

    im_hsv = imageResult.copy()
    h, s, v = cv2.split(im_hsv)

    v[v == 0] = 255
    v[v != 0] = 0

    img_lip_mask = cv2.merge((h, s, v))
    img_lip_mask = cv2.cvtColor(img_lip_mask, cv2.COLOR_HSV2RGB)

    img_masked = cv2.cvtColor(img_lip_mask, cv2.COLOR_RGB2GRAY)

    img_c_mask = cv2.bitwise_and(mouth_section, mouth_section, mask=img_masked)

    image_hsv = cv2.cvtColor(img_c_mask, cv2.COLOR_RGB2HSV)
    mask_t = cv2.inRange(image_hsv, np.array([0, 0, 95]), np.array([110, 160, 255]))

    imageR = cv2.bitwise_and(image_hsv, image_hsv, mask=mask_t)
    imageR = cv2.cvtColor(imageR, cv2.COLOR_HSV2RGB)

    return imageR

def second_layer_segmentation(first_layer_image):
    """
    Apply the second layer segmentation to the first layer image
    """
    imr_g = cv2.cvtColor(first_layer_image, cv2.COLOR_RGB2GRAY)

    contours, _ = cv2.findContours(imr_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mouth_contour = get_largest_contour(contours)

    mask_f_t = np.zeros(mouth_section.shape, np.uint8)
    cv2.drawContours(mask_f_t, [mouth_contour], 0, (255, 255, 255), -1)

    mask_f_t = cv2.cvtColor(mask_f_t, cv2.COLOR_RGB2GRAY)

    img_f = cv2.bitwise_and(mouth_section, mouth_section, mask=mask_f_t)

    img_g = cv2.cvtColor(img_f, cv2.COLOR_RGB2GRAY)

    img_g[img_g < 50] = 0
    img_g[(img_g > 80) & (img_g < 230)] += 20

    img_g_rgb = cv2.cvtColor(img_g.copy(), cv2.COLOR_GRAY2RGB)

    return img_g_rgb


def whitened_tooth(second_layer_image, input_img):
    """
    Perform tooth whitening on the second layer image and merge it with the original input image
    """
    img_o = second_layer_image.copy()
    img_o_v = cv2.cvtColor(img_o, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_o_v)

    img_a = second_layer_image.copy()
    img_a[:, :, :] = [216, 216, 207]

    final_a_img = cv2.cvtColor(img_a.copy(), cv2.COLOR_RGB2LAB)
    _, _a, _b = cv2.split(final_a_img)

    _a = l.copy()

    final_result = cv2.merge((_a, _a, _b))

    final_rgb = cv2.cvtColor(final_result.copy(), cv2.COLOR_LAB2RGB)

    blur = cv2.bilateralFilter(final_rgb, 9, 75, 75)
    rr, gg, bb = cv2.split(blur)

    merge_img = input_img.copy()
    r_, g_, b_ = cv2.split(merge_img)

    r_[l != 0] = rr[l != 0]
    g_[l != 0] = gg[l != 0]
    b_[l != 0] = bb[l != 0]

    final_m_img = cv2.merge((r_, g_, b_))

    return final_m_img


for i in images:
    input_im = cv2.imread(i)
    input_image = input_im.copy()
    mouths = mouth_cascade.detectMultiScale(input_image, 1.3, 5)

    mouth = get_largest_mouth_region(mouths)

    if mouth is not None:
        mx, my, mw, mh = mouth
        cv2.rectangle(input_image, (mx, my), (mx + mw, my + mh), (255, 255, 255), -1)

        mask_g_face = cv2.cvtColor(input_image.copy(), cv2.COLOR_RGB2GRAY)
        mask_g_face[mask_g_face < 255] = 0

        contours, _ = cv2.findContours(mask_g_face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            mouth_contour = get_largest_contour(contours)

            mouth_section = input_im.copy()
            mask_t = np.zeros(input_im.shape, np.uint8)
            mask_t = cv2.cvtColor(mask_t, cv2.COLOR_RGB2GRAY)
            cv2.drawContours(mask_t, [mouth_contour], 0, (255, 255, 255), -1)

            mouth_section = cv2.bitwise_and(mouth_section.copy(), mouth_section.copy(), mask=mask_t)

            first_segment_image = first_layer_segmentation(mouth_section)

            second_segment_image = second_layer_segmentation(first_segment_image)

            whitened_tooth_image = whitened_tooth(second_segment_image, input_im.copy())

            # Merge all images into one frame
            Horizontal_frame = np.concatenate(
                (input_im, first_segment_image, second_segment_image, whitened_tooth_image), axis=1
            )

            cv2_imshow(Horizontal_frame)
        else:
            print("NO CONTOUR FOUND ------ TRY DIFFERENT IMAGE")
    else:
        print("Mouth not detected -------- TRY DIFFERENT IMAGE")
