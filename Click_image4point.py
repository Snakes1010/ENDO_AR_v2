# Generalized function def for click_event that will print the location of a point
# in an image on the left button click and the BGR values on a right button press
# The results print in the Run and are also placed in the according image
def click_event(event, x, y, flags, params):

# checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        # displaying the coordinates
        # on the image window
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(_image_, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv.imshow('image', _image_)

# checking for right mouse clicks
    if event==cv.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        # displaying the coordinates
        # on the image window
        font = cv.FONT_HERSHEY_SIMPLEX
        b = imgL[y, x, 0]
        g = imgL[y, x, 1]
        r = imgL[y, x, 2]
        cv.putText(_image_, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x,y), font, 1,
                    (255, 255, 0), 2)
        cv.imshow('image', _image_)
