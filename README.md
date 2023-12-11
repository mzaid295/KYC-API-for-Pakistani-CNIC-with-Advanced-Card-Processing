# KYC-API-for-Pakistani-CNIC-with-Advanced-Card-Processing

# Project Overview:

The goal of this project was to develop a robust Know Your Customer (KYC) API specifically tailored for Pakistani Computerized National Identity Cards (CNICs). The API accommodates three distinct formats: ID cards with Urdu text, smart cards, and Nichop (national ID cards for overseas Pakistanis). The primary objective was to enhance security and streamline user verification by accepting two images – the front and back sides of the CNIC.



# Key Features:
# 1. Card Region Detection:
 Utilized a trained Machine Learning (ML) model on a comprehensive card dataset to accurately identify and extract the card region from provided images.

# 2. Basic Sanity Checks:
Implemented essential validation checks, including the detection of facial features, the presence of the Pakistan logo, or other landmarks specific to the card. This ensured the received card was valid.

# 3. Card Classification:
Developed a classification mechanism to categorize the card into either a smart ID card or the older format with Urdu content. This provided insights into the card type for further processing.

# 4. Data Extraction:
Extracted crucial information such as Name, Guardian Name, Date of Issue, Card Expiry Date, and CNIC Number. For Urdu format cards, focused on extracting CNIC number, card issue date, and expiry date.

# 5. OCR Integration:
Leveraged Easy OCR or Paddle OCR from available open-source options to facilitate seamless extraction of textual data from the card images.

# Implementation:

The project utilized a combination of Python programming, machine learning libraries, and OCR technologies. The ML model was trained on a diverse dataset to accurately identify card regions, while OCR tools were employed for efficient text extraction.

# Challenges Faced:

- Ensuring robustness in ML model for diverse card formats.
- Handling variations in image quality and orientation.
- Fine-tuning OCR tools for accurate text extraction.

# Results:

The KYC API successfully achieved its objectives, providing a reliable and secure means of user identification. The accuracy of card region detection and data extraction met or exceeded project requirements.

# Future Enhancements:

- Continuous model training for evolving card formats.
- Integration with additional OCR libraries for enhanced flexibility.
- Real-time processing and verification capabilities.

# Conclusion:
The KYC API for Pakistani CNICs with advanced card processing has been successfully developed and implemented. It offers a comprehensive solution for secure and efficient user verification, contributing to enhanced security in various applications, including financial services and online platforms. The project lays the foundation for future advancements in identity verification systems.



## Screenshots

![Face Detection FOr Validation Of Card](https://github.com/mzaid295/KYC-API-for-Pakistani-CNIC-with-Advanced-Card-Processing/blob/main/Detected%20Face.JPG)
![Face Detection FOr Validation Of Card](https://github.com/mzaid295/KYC-API-for-Pakistani-CNIC-with-Advanced-Card-Processing/blob/main/Detected%20Text.JPG)
![Face Detection FOr Validation Of Card](https://github.com/mzaid295/KYC-API-for-Pakistani-CNIC-with-Advanced-Card-Processing/blob/main/Dataset/Pycharm%20Result.JPG)
