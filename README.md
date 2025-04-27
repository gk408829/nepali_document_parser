# Nepali Document Parser

## Overview

The Nepali Document Parser is a Python package designed to extract text from documents (PDFs and images) containing Nepali (Devanagari) script. It leverages OCR (Optical Character Recognition) technology, specifically Tesseract, and incorporates image preprocessing and text cleaning techniques to enhance the accuracy of the extracted text. This tool is particularly useful for applications that require processing and digitizing Nepali language documents.

## Features

* **Document Parsing:** Extracts text from both PDF files and common image formats (PNG, JPG, JPEG).
* **Image Preprocessing:**
    * **Denoising:** Reduces noise in images using non-local means denoising.
    * **Adaptive Thresholding:** Converts images to binary format for better OCR performance.
    * **Artifact Removal:** Removes small smudges and noise elements.
    * **Deskewing:** Corrects document rotation to improve text alignment.
* **Text Cleaning:**
    * Removes non-Nepali characters.
    * Cleans up extra whitespace and newlines.
    * Removes special characters.
    * Corrects common OCR errors (e.g., replacing English digits with Nepali).
    * Supports custom character replacements.
* **Streamlit Interface:** Provides a user-friendly web interface for uploading documents, configuring processing settings, and viewing results.

## Installation

### Prerequisites

* **Python:** 3.7 or higher
* **Tesseract OCR:** Install the Tesseract OCR engine.  You'll need to add the Tesseract executable to your system's PATH.
* **pip:** Python package installer

### Installation Steps

1.  **Install Tesseract:**  
    * **macOS (Homebrew):**  
        `brew install tesseract`  
        (You might also need `brew install tesseract-langpack nep`)
    
    * **Ubuntu/Debian:**   
    `sudo apt-get install tesseract-ocr libtesseract-dev`  
    (and `sudo apt-get install tesseract-ocr-nep`)
    * **Windows:** Download the installer from [Tesseract download](https://github.com/UB-Mannheim/tesseract/wiki) and follow the installation instructions.  Make sure to add the Tesseract executable path to your system's `PATH` environment variable.

2.  **Clone the repository (if you have the code as a repository):**
    ```bash
    git clone https://github.com/gk408829/nepali_document_parser
    cd nepali_document_parser
    ```

3.  **Install the Python dependencies:**
    ```bash
    pip install streamlit pdf2image pytesseract opencv-python
    ```

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run nepali_ocr.py  # Replace your_script_name.py
    ```

2.  **Access the application:** Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3.  **Using the application:**
    * Upload a PDF or image file.
    * Configure the image processing and text cleaning settings in the sidebar.
    * View the processed images and extracted text in the main area.
    * Download the extracted text as a plain text file.

## Code Structure

The core functionality is implemented in the following Python files:

* `nepali_ocr.py`: This is the main script that sets up the Streamlit interface and uses the `DocumentParser` class.  It handles file uploads, user interface, and orchestrates the document processing.

### Class: `DocumentParser`

The `DocumentParser` class in `nepali_ocr.py` encapsulates the OCR processing logic.  Here's a breakdown of its methods:

* `__init__(self, tesseract_cmd)`: Initializes the `DocumentParser` with the path to the Tesseract executable.
* `remove_small_artifacts(self, image, min_area)`: Removes small noise elements from a binary image.
    * `image` (numpy.ndarray):  Binary input image.
    * `min_area` (int): Minimum area of a contour to keep.
* `preprocess_image(self, image, denoise_strength, min_smudge_size)`:  Preprocesses an image for OCR.
    * `image` (numpy.ndarray): Input image (grayscale or color).
    * `denoise_strength` (int): Denoising strength.
    * `min_smudge_size` (int): Minimum size of smudges to remove.
* `deskew(self, image)`: Corrects the skew of a document image.
    * `image` (numpy.ndarray): Binary input image.
* `clean_text(self, text, cleaning_options)`: Cleans extracted text.
    * `text` (str):  Raw text extracted by OCR.
    * `cleaning_options` (dict):  Dictionary of cleaning options (remove non-Nepali, whitespace, special chars, fix common errors, custom replacements).
* `parse_pdf(self, pdf_path, processing_params, cleaning_options)`: Extracts text from a PDF file.
    * `pdf_path` (str): Path to the PDF file.
    * `processing_params` (dict): Image processing parameters.
    * `cleaning_options` (dict): Text cleaning options.
    * Returns: (str: extracted text, list: processed images).
* `parse_image(self, image_file, processing_params, cleaning_options)`: Extracts text from an image file.
    * `image_file` (UploadedFile):  Uploaded image file.
    * `processing_params` (dict): Image processing parameters.
    * `cleaning_options` (dict): Text cleaning options.
    * Returns: (str: extracted text,  PIL.Image.Image: processed image).

## Testing

If you have included a test script (e.g., `test_document_parser.py`), add a section here to explain how to run the tests.  For example:

### Running Tests

To ensure the package is working correctly, you can run the included tests:

1.  Navigate to the package's root directory:

    ```bash
    cd nepali_document_parser
    ```

2.  Run the tests using your preferred testing framework. If you're using `pytest`, for example:

    ```bash
    pytest
    ```

    If you have a specific test file:
    ```bash
    pytest test_nepali_ocr.py
    ```



## Dependencies

* [Streamlit](https://streamlit.io/): For the user interface.
* [PyTesseract](https://github.com/madmaze/pytesseract): Python interface for Tesseract OCR.
* [pdf2image](https://github.com/Belval/pdf2image): For converting PDF pages to images.
* [OpenCV](https://opencv.org/): For image processing.
* [NumPy](https://numpy.org/): For numerical operations.
* [PIL (Pillow)](https://pillow.readthedocs.io/): For image manipulation.
* [re](https://docs.python.org/3/library/re.html): For regular expressions.

## Contributing

Contributions are welcome!  If you find a bug, have a feature request, or would like to contribute code, please open an issue or submit a pull request on [GitHub](your_repository_url).

## License

MIT
