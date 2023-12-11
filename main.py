from yolov8 import YOLOv8
import cv2
import easyocr
from pyzbar.pyzbar import decode
import csv

def img_read(front_gray_roi,back_image, model):
    # Detect Objects Front
    boxes, scores, class_ids = model(front_gray_roi)
    card_front_roi = model.draw_detections(front_gray_roi)
    # cv2.imshow("Detected Objects", card_front_roi)
    cv2.imwrite("doc/img/detected_objects.jpg", card_front_roi)

    boxes, scores, class_ids = model(back_image)
    card_back_roi = model.draw_detections(back_image)
    # cv2.imshow("Detected Objects", card_back_roi)
    cv2.imwrite("doc/img/detected_objects.jpg", card_back_roi)

    return card_back_roi, card_front_roi


def apply_ocr_front(front_gray_roi):
    # Detect Objects Front
    boxes, scores, class_ids = model(front_gray_roi)
    [coordinates] = boxes
    # Extract the region defined by the coordinates
    x_min, y_min, x_max, y_max = [int(coord) for coord in coordinates]

    # Calculate the midpoint of the bounding box
    midpoint_x = (x_min + x_max) // 2

    # If the midpoint is on the left side, consider only the right half
    if midpoint_x < front_gray_roi.shape[1] // 2:
        x_min = min(front_gray_roi.shape[1] // 2, midpoint_x)
        x_max = min(front_gray_roi.shape[1], midpoint_x + front_gray_roi.shape[1] // 2)

    # roi = front_gray_roi[y_min:y_max, x_min:x_max]

    if len(faces) > 0:
        # print("Face detected in the cropped region.")
        print("::: Just Wait, Sysytem Is Reading Your Card Information.....\n You Card Is Here :::")

        for (x, y, w, h) in faces:
            # Define the coordinates for the left rectangle
            left_rect_length = 410  # Adjust the length as needed
            left_x = max(0, x - left_rect_length)
            left_y = y - 72
            left_width = x - 15
            left_height = y + h + 40

            # Crop the left rectangle
            uper_rect_roi = front_gray_roi[left_y:left_height, left_x:left_width]

            # cv2.cvtColor(uper_rect_roi, cv2.COLOR_BGR2GRAY)
            # sharpened_image = cv2.filter2D(uper_rect_roi, -1, kernel)
            cv2.cvtColor(uper_rect_roi, cv2.COLOR_BGR2GRAY)
            # Display the cropped left rectangle
            cv2.imshow('Upper Rectangle', uper_rect_roi)
            cv2.waitKey(0)


        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'], gpu=False)  # Specify the language(s) you want to read
        gray_roi = cv2.cvtColor(uper_rect_roi, cv2.COLOR_BGR2GRAY)

        # Perform OCR on the grayscale image
        result = reader.readtext(gray_roi)

        # List of common phrases
        common_phrases = ["Pakistan", "Name", "Father Name", "Gender", "Country of Stay", "Identity Number",
                          "Date of Issue",
                          "Date of Expiry", "Signature", "PAKISTAN", "ISLAMIC REpUBLIC OF PAKISTAN", "N", "M",
                          "Date of Birth",
                          "CW", "Ayw", "Holder", "ISLavic REPubLIC Of PAnisI", "Father's name", "pakistan",
                          "Date of Explry", "United Arab Emirates", "eruv Jisl","Uziest","Tusband Name","@vuv_islz","Country 8f Stay"
                          ,"United Arab Emiiates","Fu","e"]

        # Filter out common text
        # filtered_text = [line for line in result if all(phrase not in line[1] for phrase in common_phrases)]

        filtered_results = [(detection[0], detection[1].replace('-', '')) for detection in result if
                            detection[1] not in common_phrases]
        # Create a new variable to store the filtered results
        filtered_data_upper = [detection[1] for detection in filtered_results]
        # print("This is list", filtered_data_upper) #Uncomment this line for data
        print("Name:",filtered_data_upper[0])
        print("Father Name:", filtered_data_upper[1])

        dic_upper = {"Name :" : filtered_data_upper[0],
                     "Father Name:": filtered_data_upper[1]}
        return dic_upper, filtered_data_upper

def crop_front_lower(front_gray_roi):
    for (x, y, w, h) in faces:
        # Define the coordinates for the left rectangle
        left_rect_length = 410  # Adjust the length as needed
        left_x = max(0, x - left_rect_length)
        left_y = y + 110
        left_width = x - 15
        left_height = y + h + 135

    # Crop the left rectangle
    lower_rect_roi = front_gray_roi[left_y:left_height, left_x:left_width]
    # Display the cropped left rectangle
    cv2.imshow('lower Rectangle', lower_rect_roi)
    cv2.waitKey(0)

    # OCR
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(lower_rect_roi)
    common_phrases = ["Pakistan", "Name", "Father Name", "Gender", "Country of Stay", "Identity Number",
                      "Date of Issue",
                      "Date of Expiry", "Signature", "PAKISTAN", "ISLAMIC REpUBLIC OF PAKISTAN", "N", "M",
                      "Date of Birth",
                      "CW", "Ayw", "Holder", "ISLavic REPubLIC Of PAnisI", "Father's name", "pakistan",
                      "Date of Explry", "United Arab Emirates", "eruv Jisl", "Date", "of Expiry","Gender;","Country 8f Stay ",
                      "United Arab Emiiates","Identity Nuinber","Date ofIssue","Date Of Expiry","12 M 2m19","12104 2029","Date of",
                      "Expiry"]

    # Filter out common text
    filtered_text = [line for line in result if all(phrase not in line[1] for phrase in common_phrases)]

    filtered_results = [(detection[0], detection[1].replace('-', '')) for detection in result if
                        detection[1] not in common_phrases]
    # Create a new variable to store the filtered results
    filtered_data_lower = [detection[1] for detection in filtered_results]
    # print("This is list", filtered_data) #Un-comment this line for complete data view

    # print('Id Card Number: ', filtered_data_lower[0],
    #       'Data Of Birth: ', filtered_data_lower[1],
    #       'Data Of Issue: ', filtered_data_lower[2],
    #       'Data Of Expiry: ' , filtered_data_lower[3])
    # print("Name:", filtered_data[0])
    # print("Father Name:", filtered_data[1])

    dic_lower = {'Id Card Number: ': filtered_data_lower[0],
           'Data Of Birth: ': filtered_data_lower[1],
           'Data Of Issue: ': filtered_data_lower[2],
           'Data Of Expiry' : filtered_data_lower[3]
           }
    # return dic_lower
    # print("Dictionary", dic)

    for line in filtered_text:
        if '-' in line[1] and len(line[1]) == 15:
            cnic_no = line[1]
            cnic=cnic_no.replace('-','')
    return cnic, dic_lower, filtered_data_lower


def apply_ocr_back(card_back_roi):
    # Detect Objects
    boxes, scores, class_ids = model(card_back_roi)

    model.draw_detections(card_back_roi)

    # Convert the image to grayscale (required for QR code detection)
    gray_img = cv2.cvtColor(card_back_roi, cv2.COLOR_BGR2GRAY)

    # Use OpenCV to find and decode QR codes
    qr_codes = decode(gray_img)

    if qr_codes:
        for qr_code in qr_codes:
            cnic_back_number = qr_code.data.decode('utf-8')
            # print(cnic_back_number[12:25])
            backText = cnic_back_number[12:25]
            # print('Back Text',backText)
            print("BackSide Id Card Number: ", backText)
        return backText

    else:
        print("No QR code found on the card.")
        print("Please provide an authentic card.")

def cnic_validation(cnic, backText):
    if cnic == backText:
        print("CNIC is valid")
    else:
        print("ERROR !!! FRONT SIDE & BACK SIDE OF THE CARD ARE NOT SAME")

def extract_data(dic_upper, dic_lower,filtered_data_upper, filtered_data_lower):
    merged_dict = {**dic_upper, **dic_lower}
    list = filtered_data_upper + filtered_data_lower
    print("-----------------------------",list)
    csv_file_path = 'IdCardData.csv'

    # Open the file in write mode and create a CSV writer object
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the data to the CSV file
        csv_writer.writerows(list)

    print(f'Data has been written to {csv_file_path}')
    return list

    # print("Merged Dictionary",merged_dict)


if __name__=='__main__':
    f_img = '../images/t2.jpg'
    b_img = '../images/qrback.jpg'
    front_imagee = cv2.imread(f_img)
    front_gray_roi = cv2.cvtColor(front_imagee, cv2.COLOR_BGR2GRAY)


    back_image = cv2.imread(b_img)
    model = "../Model/best.onnx"
    model = YOLOv8(model, conf_thres=0.2, iou_thres=0.3)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(front_gray_roi, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))


    #Image Detection Show
    card_back_roi, card_front_roi = img_read(front_gray_roi ,back_image, model)
    cv2.imshow('Front',card_front_roi)
    cv2.imshow('Back', card_back_roi)
    cv2.waitKey(0)

    #OCR
    dic_upper, filtered_data_upper = apply_ocr_front(front_imagee)
    cnic, dic_lower, filtered_data_lower = crop_front_lower(front_gray_roi)
    # extract_data(dic_upper, dic_lower, filtered_data_upper ,filtered_data_lower)
    back = apply_ocr_back(card_back_roi)
    cnic_validation(cnic, back)
    list = extract_data(dic_upper, dic_lower, filtered_data_upper, filtered_data_lower)
    print(list)
