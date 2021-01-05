# ==Imports==
import os, sys
import cv2
import imutils
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# =======================================================================================================================
# Classes to detect pedestrians in different frames. If --colorbased is set to true, information about the color of the
# object that needs to be tracked is taken into account. The respective color can be specified via the variable --color.
# =======================================================================================================================

class Detector:
    def __init__(self, colorbased, color="BLACK"):
        self.colorbased = colorbased
        self.color = color

    def detectblack(self, img):
        """
        Detect all black patches in current frame
        :param frame: input frame
        """
        col_switch = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define thresholds
        lower, upper = self.getthresh()

        # Get mask
        mask_black = cv2.inRange(col_switch, lower, upper)

        # Get contours and patches
        contours_black, _ = cv2.findContours(mask_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        black_patches = self.get_patches(contours_black)

        return black_patches


    def get_patches(self, contour_list):
        """
        Creates a list of patches based on contour list
        :param contour_list: List of contours
        :return: List of patches
        """
        patch_list = []

        for cID, cnt in enumerate(contour_list):
            M = cv2.moments(cnt)
            c_area = M['m00']

            # Neglect small contours:
            if c_area <= 300:
                continue

            # Get rough measurements of contour
            (x, y, w, h) = cv2.boundingRect(cnt)

            # Append patch
            patch_list.append((x,y,w,h))

        return patch_list


    def getthresh(self):
        """
        Get threshold values for color detection
        """
        if self.color.upper() == "BLACK":
            lower = np.array([0, 0, 0], dtype="uint8")
            upper = np.array([180, 255, 30], dtype="uint8")
        else:
            raise ValueError("Color %s invalid!", self.color)

        return lower, upper


    def rescale(self, x, y, w, h, sw, sh):
        """
        Rescales a bounding box
        :param x: x-coordinate of upper left pixel of rectangular
        :param y: y-coordinate of upper left pixel of rectangular
        :param w: width of rectangular
        :param h: height of rectangular
        :param sw: scaling factor for width
        :param sh: scaling factor for height
        :return: (x_new, y_new, ws, hs) = (scaled x-coordinate, scaled y-coordinate, scaled width, scaled height)
        """
        ws = sw * w
        hs = sh * h

        x_new = int(x + 0.5 * (w - ws))
        y_new = int(y + 0.5 * (h - hs))

        return x_new, y_new, int(ws), int(hs)


class HOG(Detector):
    def __init__(self, colorbased, color):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.sw = 0.58
        self.sh = 0.78
        super(HOG, self).__init__(colorbased, color)

    def detection(self, image):
        """
        Detects pedestrians in the image frame and returns the of the respective region of interest
        :param image: input image
        :return: list of (x,y,w,h) values for each rectangular
        """
        rects = []
        weights = []

        # Resize image if necessary
        img = imutils.resize(image, width=min(400, image.shape[1]))

        # Detect all black patches in frame
        if self.colorbased:
            black_patches = self.detectblack(img)

        # Detect all pedestrians
        (regions, confidence) = self.hog.detectMultiScale(img, winStride=(4, 4), padding=(4, 4), scale=1.05)

        # Drawing the regions in the Image
        for i, (x, y, w, h) in enumerate(regions):
            if self.colorbased:
                valid = False
                # check whether black patch in box
                for (x_black, y_black, w_black, h_black) in black_patches:
                    if (x_black >= x) and (y_black >= y) and ((x_black + w_black) <= (x + w)) and ((y_black + h_black) <= (y + h)):
                        valid = True
            else:
                valid = True

            if valid:
                x_new, y_new, ws, hs = self.rescale(x, y, w, h, self.sw, self.sh)
                rects.append((x_new, y_new, ws, hs))
                weights.append(confidence[i][0])
                cv2.rectangle(img, (x_new, y_new), (x_new + ws, y_new + hs), (0, 0, 255), 2)

        # Showing the output Image
        #cv2.imshow("Image", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return rects, weights


# ===============   Detector ===================

class PedestrianDetector:
    def __init__(self, type, colorbased, color):
        if type.upper() == "HOG":
            self.detector = HOG(colorbased, color)
        else:
            raise ValueError("Invalid Detector. Choose either HOG or")

    def detectped(self, image):
        """
        Detect pedestrians in image frame
        :param image: input frame
        """
        rects = self.detector.detection(image)
        return rects


