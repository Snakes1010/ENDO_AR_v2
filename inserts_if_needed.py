################################################################
# AR with google cardboard
AR_use = True
if AR_use == True:
    cv.namedWindow('AR')

    scale_percent = 150  # percent of original size, 200 makes the image twice as large
    insert_L_size = 0
    insert_M_size = 0
    insert_R_size = 0
    insert_L = np.zeros((480, insert_L_size,3), np.uint8)
    insert_M = np.zeros((480, insert_M_size,3), np.uint8)
    insert_R = np.zeros((480, insert_R_size, 3), np.uint8)
    width = int((1280 + (insert_L_size+ insert_M_size + insert_R_size)) * scale_percent / 100)
    height = int(480 * scale_percent / 100)
    dim = (width, height)
    print(dim)


    def calibrate_insert_L(val):
        global insert_L, insert_L_size, width
        insert_L_size = val
        insert_L = np.zeros((480, insert_L_size, 3), np.uint8)
        width = int(1280 + (insert_L_size + insert_M_size + insert_R_size) * scale_percent / 100)
    def calibrate_insert_M(val):
        global insert_M, insert_M_size, width
        insert_M_size = val
        insert_M = np.zeros((480, insert_M_size, 3), np.uint8)
        width = int(1280 + (insert_L_size + insert_M_size + insert_R_size) * scale_percent / 100)
    def calibrate_insert_R(val):
        global insert_R, insert_R_size, width
        insert_R_size = val
        insert_R = np.zeros((480, insert_R_size, 3), np.uint8)
        width = int(1280 + (insert_L_size + insert_M_size + insert_R_size) * scale_percent / 100)

    cv.namedWindow('Calibration')
    cv.createTrackbar('LEFT', 'Calibration', 0, 500, calibrate_insert_L)
    cv.createTrackbar('MIDDLE', 'Calibration', 0, 500, calibrate_insert_M)
    cv.createTrackbar('RIGHT', 'Calibration', 0, 500, calibrate_insert_R)



    if AR_use == True:
        combined_rectify = np.concatenate((insert_L, frame_left_remap, insert_M, frame_right_remap, insert_R), axis=1)
        resized = cv.resize(combined_rectify, dim, interpolation=cv.INTER_AREA)

        cv.line(resized, (0,(360)), (1920, 360),(255,0,255), 3)
        print(resized.shape)
        cv.imshow('AR', resized)
        cv.moveWindow('AR', 5600, 150)
        cv.imshow('Depth', depth/2000)
        if cv.waitKey(1) & 0xFF == ord('q'):