from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from yolov8 import YOLOv8
import cv2
import numpy as np
import urllib
import easyocr
import csv
from pyzbar.pyzbar import decode
from datetime import datetime

@api_view(['Post'])
def id_detector(request):

    # Grab front and back images from the request
    front_imagee = grab_image(stream=request.FILES["front_imagee"])
    back_image = grab_image(stream=request.FILES["back_image"])

    # Convert front image to Gray Image For Better Read
    front_gray_roi = cv2.cvtColor(front_imagee, cv2.COLOR_BGR2GRAY)

    # Detect faces on the front image
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(front_gray_roi, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    # Image Detection Show
    card_back_roi, card_front_roi = img_read(front_gray_roi, back_image)

    # Applying OCR on the front side
    dic_upper, filtered_data_upper = apply_ocr_front(front_imagee, faces)

    # Crop and process information from the lower part of the front side
    cnic, dic_lower, filtered_data_lower = crop_front_lower(front_gray_roi, faces)

    # Apply OCR on the back side
    back = apply_ocr_back(card_back_roi)

    # Validate CNIC
    valid = cnic_validation(cnic, back)

    if not valid:
        return Response({'error': 'FRONT SIDE & BACK SIDE OF THE CARD ARE NOT SAME'},
                        status=status.HTTP_406_NOT_ACCEPTABLE)
    else:
        # Extract and merge data from both sides
        merged_dict = extract_data(dic_upper, dic_lower, filtered_data_upper, filtered_data_lower)
        return Response(merged_dict, status=status.HTTP_200_OK)


def img_read(front_gray_roi, back_image):
    # Initialize the YOLOv8 Custom Model
    model = "../Model/best.onnx"
    model = YOLOv8(model, conf_thres=0.2, iou_thres=0.3)

    # Detection on the front image
    boxes, scores, class_ids = model(front_gray_roi)
    card_front_roi = model.draw_detections(front_gray_roi)

    # Detection on the back image
    boxes, scores, class_ids = model(back_image)
    card_back_roi = model.draw_detections(back_image)

    return card_back_roi, card_front_roi


def apply_ocr_front(front_gray_roi, faces):
    model = "../Model/best.onnx"
    model = YOLOv8(model, conf_thres=0.2, iou_thres=0.3)

    # Detect Objects From The Front Side
    boxes, scores, class_ids = model(front_gray_roi)
    [coordinates] = boxes

    # Extract the region defined by the coordinates
    x_min, y_min, x_max, y_max = [int(coord) for coord in coordinates]

    # Calculate the midpoint of the bounding box for card authentication
    midpoint_x = (x_min + x_max) // 2

    # If the midpoint is on the left side, consider only the right half
    if midpoint_x < front_gray_roi.shape[1] // 2:
        x_min = min(front_gray_roi.shape[1] // 2, midpoint_x)
        x_max = min(front_gray_roi.shape[1], midpoint_x + front_gray_roi.shape[1] // 2)

    # Detect If the face is available on the card then proceed further
    if len(faces) > 0:
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
            cv2.cvtColor(uper_rect_roi, cv2.COLOR_BGR2GRAY)

        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'], gpu=False)  # Specify the language(s) you want to read
        gray_roi = cv2.cvtColor(uper_rect_roi, cv2.COLOR_BGR2GRAY)

        # Perform OCR on the grayscale image
        result = reader.readtext(gray_roi)

        # List of common phrases
        common_phrases = ["/", "Pakistan", "Name", "Father Name", "Gender", "Country of Stay", "Identity Number",
                          "Date of Issue", "Date of Expiry", "Signature", "PAKISTAN", "ISLAMIC REpUBLIC OF PAKISTAN",
                          "N", "M", "Date of Birth", "SLAM C Repualic Of PaRiSTAn", "Gender", "Country of Stay"
                          "CW", "Ayw", "Holder", "ISLavic REPubLIC Of PAnisI", "Father's name", "pakistan",
                          "Date of Explry", "United Arab Emirates", "eruv Jisl", "Uziest", "Tusband Name", "@vuv_islz",
                          "Country 8f Stay", "United Arab Emiiates", "Fu", "e","Uv ,3","Gender","SLAM C Repualic Of PaRiSTAn", ",","Uv ,3","Husband Name","ecuv Ji","Date of Expiny"]

        # Filter out common text
        filtered_results = [(detection[0], detection[1].replace('-', '')) for detection in result if
                            detection[1] not in common_phrases]

        # Create a new variable to store the filtered results
        filtered_data_upper = [detection[1] for detection in filtered_results]

        dic_upper = {"Name": filtered_data_upper[0], "Gaurdian Name": filtered_data_upper[1]}

        return dic_upper, filtered_data_upper


# Rest of the code follows similar commenting style...


def crop_front_lower(front_gray_roi, faces):
    for (x, y, w, h) in faces:
        # Define the coordinates for the left rectangle
        left_rect_length = 410  # Adjust the length as needed
        left_x = max(0, x - left_rect_length)
        left_y = y + 110
        left_width = x - 15
        left_height = y + h + 135
    # Crop the left rectangle
    lower_rect_roi = front_gray_roi[left_y:left_height, left_x:left_width]
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(lower_rect_roi)
    common_phrases = ["Date of Explry", "United Arab Emirates", "eruv Jisl", "Date", "of Expiry", "Gender;","Country 8f Stay ","United Arab Emiiates", "Identity Nuinber", "Date ofIssue", "Date Of Expiry", "12 M 2m19","12104 2029", "Date of","Expiry","/","Pakistan","Date of Birth","Identity Number", "Date of Issue","578","Date of Expiny","ecuv Ji","Gender","Country of Stay"]
    # Filter out common text
    filtered_text = [line for line in result if all(phrase not in line[1] for phrase in common_phrases)]
    filtered_results = [(detection[0], detection[1].replace('-', '')) for detection in result if detection[1] not in common_phrases]
    # Create a new variable to store the filtered results
    filtered_data_lower = [detection[1] for detection in filtered_results]
    converted_list = convert_date_format_list(filtered_data_lower)
    #   Creating a Dictionary
    dic_lower = {'Id Card Number': converted_list[0], 'Data Of Birth': converted_list[1],'Data Of Issue': converted_list[2],'Data Of Expiry': converted_list[3]}
    #Filter out the Cnic number only
    for line in filtered_text:
        if '-' in line[1] and len(line[1]) == 15:
            cnic_no = line[1]
            cnic = cnic_no.replace('-', '')
    return cnic, dic_lower, converted_list

def apply_ocr_back(card_back_roi):
    model = "../Model/best.onnx"
    model = YOLOv8(model, conf_thres=0.2, iou_thres=0.3)
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
        return True
    else:
        print("ERROR !!! FRONT SIDE & BACK SIDE OF THE CARD ARE NOT SAME")
        return False


def extract_data(dic_upper, dic_lower, filtered_data_upper, filtered_data_lower):
    merged_dict = {**dic_upper, **dic_lower}
    list = filtered_data_upper + filtered_data_lower
    for key, value in merged_dict.items():
        # Replace dots with an empty string
        merged_dict[key] = value.replace('.', '')
    # print("-----------------------------", list)
    csv_file_path = 'IdCardData.csv'

    # Open the file in write mode and create a CSV writer object
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the data to the CSV file
        csv_writer.writerows(merged_dict)
    # print(f'Data has been written to {csv_file_path}')
    return merged_dict

def grab_image(path=None, stream=None, url=None):
# if the path is not None, then load the image from disk
    if path is not None:
        image = cv2.imread(path)
# otherwise, the image does not reside on disk
    else:
    # if the URL is not None, then download the image
        if url is not None:
            resp = urllib.urlopen(url)
            data = resp.read()
        # if the stream is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()
        # convert the image to a NumPy array and then read it into # OpenCV format
            image = np.asarray(bytearray(data), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # return the image return image
    return image

def convert_date_format_list(elements):
    def convert_date_format(element):
        if isinstance(element, str):
            try:
                # Convert the date string to a datetime object
                date_object = datetime.strptime(element, "%d.%m.%Y")
                # Format the datetime object as "DDMMYY"
                formatted_date = date_object.strftime("%d%m%y")
                return formatted_date
            except ValueError:
                # If the conversion fails, return the original string
                return element
        else:
            # If the element is not a string, return it as is
            return element

    # Convert each element in the list
    converted_list = [convert_date_format(element) for element in elements]
    return converted_list
