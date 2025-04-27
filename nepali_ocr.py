import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
import io
import re

# Set Tesseract command path (adjust for your system)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'


def remove_small_artifacts(image, min_area=15):
    """Remove small blobs (smudges) using contour area filtering
    Args:
        image (numpy.ndarray): The input image.
        min_area (int, optional): Minimum area of a contour to be considered
            as a valid object. Defaults to 15.

    Returns:
        numpy.ndarray: The image with small artifacts removed.
    """
    contours, _ = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    mask = np.ones_like(image) * 255
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            cv2.drawContours(mask, [cnt], -1, 0, -1)

    return cv2.bitwise_and(image, mask)


def preprocess_image(image, denoise_strength=10, min_smudge_size=15):
    """Enhanced preprocessing with smudge removal
    Args:
        image (numpy.ndarray): The input image.
        denoise_strength (int, optional): Denoising strength for
            cv2.fastNlMeansDenoising. Defaults to 10.
        min_smudge_size (int, optional): Minimum area for smudge removal.
            Defaults to 15.

    Returns:
        numpy.ndarray: The preprocessed image.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Advanced denoising
    denoised = cv2.fastNlMeansDenoising(
        gray,
        h=denoise_strength,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Smudge removal
    cleaned = remove_small_artifacts(thresh, min_area=min_smudge_size)

    # Morphological cleanup
    kernel = np.ones((2, 2), np.uint8)
    final = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    return final


def deskew(image):
    """Correct document skew/rotation
     Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The deskewed image.
    """
    coords = np.column_stack(np.where(image > 0))
    if len(coords) <= 20:
        return image

    try:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
    except:
        return image


def clean_text(text, cleaning_options):
    """Enhanced text cleaning with smudge artifact removal
    Args:
        text (str): The text to be cleaned.
        cleaning_options (dict): A dictionary of cleaning options.
            See the main function for available options.

    Returns:
        str: The cleaned text.
    """
    cleaned_text = text

    # Remove isolated symbols (common OCR noise)
    cleaned_text = re.sub(r'(?<!\w)[^\u0900-\u097F\s](?!\w)', '', cleaned_text)

    # Remove lines with excessive symbols/no text
    lines = cleaned_text.split('\n')
    cleaned_lines = []
    for line in lines:
        if re.search(r'[\u0900-\u097F]{2,}', line):  # Keep lines with Nepali text
            cleaned_lines.append(line)
    cleaned_text = '\n'.join(cleaned_lines)

    # Standard cleaning options
    if cleaning_options.get('remove_non_nepali', False):
        cleaned_text = re.sub(r'[^\u0900-\u097F0-9 .,!?;:()\[\]{}\-"\'_\n]', '', cleaned_text)

    if cleaning_options.get('remove_whitespace', False):
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        cleaned_text = re.sub(r'^ +', '', cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

    if cleaning_options.get('remove_special_chars', False):
        # Extended list of symbols to remove
        chars_to_remove = r'[|~`@#$%^&*+=<>©®™§¶¬]'
        cleaned_text = re.sub(chars_to_remove, '', cleaned_text)

    if cleaning_options.get('fix_common_errors', False):
        replacements = {
            'o': 'ो', 'O': 'ओ',
            '0': '०', '1': '१', '2': '२', '3': '३',
            '4': '४', '5': '५', '6': '६', '7': '७',
            '8': '८', '9': '९',
            **cleaning_options.get('custom_replacements', {})
        }
        for wrong, correct in replacements.items():
            cleaned_text = cleaned_text.replace(wrong, correct)

    return cleaned_text


def parse_pdf(pdf_path, processing_params, cleaning_options):
    """Process PDF with enhanced smudge handling
    Args:
        pdf_path (str): Path to the PDF file.
        processing_params (dict): A dictionary of image processing parameters.
        cleaning_options (dict): A dictionary of text cleaning options.

    Returns:
        tuple: A tuple containing the extracted text (str) and a list of
            processed images (PIL.Image.Image).
    """
    try:
        images = convert_from_path(
            pdf_path,
            dpi=300,
            thread_count=os.cpu_count(),
            use_pdftocairo=True,
        )

        full_text = ""
        processed_images = []
        custom_config = r'--oem 3 --psm 6 -l nep'

        progress_bar = st.progress(0, text="Processing PDF Pages...")  # Added text to progress bar
        for i, image in enumerate(images):
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            if processing_params['use_preprocessing']:
                img_cv = preprocess_image(
                    img_cv,
                    denoise_strength=processing_params['denoise_strength'],
                    min_smudge_size=processing_params['min_smudge_size']
                )

                if processing_params['use_deskewing'] and len(img_cv.shape) == 2:
                    img_cv = deskew(img_cv)

            processed_images.append(
                Image.fromarray(
                    img_cv if len(img_cv.shape) == 3
                    else cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
                )
            )

            text = pytesseract.image_to_string(img_cv, config=custom_config)
            full_text += f"--- Page {i + 1} ---\n{clean_text(text, cleaning_options)}\n\n"

            progress_bar.progress((i + 1) / len(images),
                                  text=f"Processed {i + 1} of {len(images)} Pages")  # Updated progress bar text

        return full_text, processed_images
    except Exception as e:
        return f"Error processing PDF: {str(e)}", []


def parse_image(image_file, processing_params, cleaning_options):
    """Process single image with enhanced cleaning
    Args:
        image_file (UploadedFile): Streamlit UploadedFile object.
        processing_params (dict): A dictionary of image processing parameters.
        cleaning_options (dict): A dictionary of text cleaning options.

    Returns:
        tuple: A tuple containing the extracted text (str) and the processed
            image (PIL.Image.Image).
    """
    try:
        image = Image.open(image_file)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if processing_params['use_preprocessing']:
            img_cv = preprocess_image(
                img_cv,
                denoise_strength=processing_params['denoise_strength'],
                min_smudge_size=processing_params['min_smudge_size']
            )

            if processing_params['use_deskewing'] and len(img_cv.shape) == 2:
                img_cv = deskew(img_cv)

        pil_processed = Image.fromarray(
            img_cv if len(img_cv.shape) == 3
            else cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
        )

        custom_config = r'--oem 3 --psm 6 -l nep'
        text = pytesseract.image_to_string(img_cv, config=custom_config)

        return clean_text(text, cleaning_options), pil_processed
    except Exception as e:
        return f"Error processing image: {str(e)}", None


def main():
    st.set_page_config(page_title="Nepali Document Parser", layout="wide")
    st.title("Nepali (Devnagari) Document Parser")

    # Sidebar controls
    st.sidebar.title("Processing Settings")

    # Image processing settings
    st.sidebar.subheader("Image Cleaning")
    processing_params = {
        'use_preprocessing': st.sidebar.checkbox("Enable Image Cleaning", True),
        'denoise_strength': st.sidebar.slider("Denoising Strength", 0, 20, 10,
                                              help="Higher values remove more noise but may blur the image."),
        'min_smudge_size': st.sidebar.slider("Min Smudge Size (pixels)", 0, 50, 15,
                                             help="Smallest area of a blob to be removed as smudge."),
        'use_deskewing': st.sidebar.checkbox("Auto Deskew", False,
                                             help="Automatically correct document rotation.")
    }

    # Text cleaning settings
    st.sidebar.subheader("Text Cleaning")
    cleaning_options = {
        'remove_non_nepali': st.sidebar.checkbox("Remove Non-Nepali Chars", True,
                                                 help="Remove any characters that are not Nepali."),
        'remove_whitespace': st.sidebar.checkbox("Clean Whitespace", True,
                                                 help="Remove extra spaces and newlines."),
        'remove_special_chars': st.sidebar.checkbox("Remove Special Chars", True,
                                                    help="Remove symbols like |, ~, @, #, etc."),
        'fix_common_errors': st.sidebar.checkbox("Fix Common OCR Errors", True,
                                                 help="Fix common OCR mistakes like replacing '0' with '०'.")
    }

    # Custom replacements
    st.sidebar.subheader("Custom Character Fixes")
    custom_replacements = {}
    for i in range(3):
        cols = st.sidebar.columns(2)
        wrong = cols[0].text_input(f"Wrong {i + 1}", key=f"wrong_{i}",
                                   placeholder="e.g.,  o")
        correct = cols[1].text_input(f"Correct {i + 1}", key=f"correct_{i}",
                                     placeholder="e.g., ो")
        if wrong and correct:
            custom_replacements[wrong] = correct
    cleaning_options['custom_replacements'] = custom_replacements

    # File upload
    uploaded_file = st.file_uploader(
        "Upload PDF or Image",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Upload a PDF or image containing Nepali text."
    )

    if uploaded_file:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Processed Pages")
            placeholder = st.empty()

        with col2:
            st.subheader("Extracted Text")
            text_placeholder = st.empty()
            download_btn = st.empty()

        file_ext = uploaded_file.name.split('.')[-1].lower()

        with st.spinner('Processing Document...'):  # Added more descriptive message
            if file_ext == 'pdf':
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                text, images = parse_pdf(tmp_path, processing_params, cleaning_options)
                os.unlink(tmp_path)

                with col1:
                    for i, img in enumerate(images):
                        st.image(img, caption=f"Page {i + 1}", use_column_width=True)
            else:
                text, processed_img = parse_image(
                    uploaded_file, processing_params, cleaning_options
                )
                with col1:
                    if processed_img:
                        st.image(processed_img, use_column_width=True)

            with col2:
                text_placeholder.text_area("Extracted Text", text, height=500)  # Added label to text area
                download_btn.download_button(
                    "Download Text",
                    io.BytesIO(text.encode()),
                    "extracted_text.txt",
                    "text/plain"
                )


if __name__ == "__main__":
    main()
