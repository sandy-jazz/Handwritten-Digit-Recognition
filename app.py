import base64
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # to stop the oneDNN message in the terminal

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered",
)

# paths for the model and the background image
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_FOLDER, "models", "digit_cnn_model.keras")
BACKGROUND_PATH = os.path.join(BASE_FOLDER, "backgroundimage.jpg")


# ---------------- helper functions ----------------

def show_html(code):
    # streamlit shows indented html as a code block,
    # so I strip the spaces from every line first
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


@st.cache_resource
def load_model():
    # cached so the model is not loaded again on every click
    return tf.keras.models.load_model(MODEL_PATH)


def get_background():
    # css can't read a local file path in streamlit,
    # so I convert the image to base64 and put it in the css
    if os.path.exists(BACKGROUND_PATH):
        with open(BACKGROUND_PATH, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f'url("data:image/jpeg;base64,{data}")'
    # if there is no image, use a simple gradient
    return "linear-gradient(135deg, #e8eefb, #f4f1fb)"


def prepare_image(image):
    # makes the image look like an MNIST image (28x28, white digit on black)

    # transparent PNGs turn black otherwise, so put them on a white background
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, image)

    image = ImageOps.exif_transpose(image).convert("L")
    pixels = np.array(image, dtype="float32")

    # MNIST is white digit on black. If the corners are bright it means
    # black ink on white paper, so invert it
    corners = [pixels[0, 0], pixels[0, -1], pixels[-1, 0], pixels[-1, -1]]
    if np.mean(corners) > 127:
        pixels = 255 - pixels

    # cut away the empty area around the digit
    rows, cols = np.where(pixels > 50)
    if len(rows) > 0:
        pixels = pixels[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

    # shrink the digit to fit in 20x20, then paste it in the middle of 28x28
    digit = Image.fromarray(pixels.astype("uint8"))
    digit.thumbnail((20, 20))
    final = Image.new("L", (28, 28), 0)
    final.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    return final


# ---------------- css ----------------

show_html(
    f"""
    <style>
    #MainMenu, footer {{ visibility: hidden; }}

    .stApp {{ background: transparent; }}

    /* blurred background image */
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: {get_background()};
        background-size: cover;
        background-position: center;
        filter: blur(3px);
        transform: scale(1.04);
        z-index: -2;
    }}

    /* light layer on top of the image */
    .stApp::after {{
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(245, 248, 252, 0.78);
        z-index: -1;
    }}

    .block-container {{
        max-width: 860px;
        padding: 2rem 1rem 3rem;
    }}

    /* white card that holds everything */
    .st-key-main_card {{
        background: rgba(255, 255, 255, 0.96);
        padding: 2.2rem;
        border-radius: 24px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 20px 60px rgba(15, 23, 42, 0.15);
    }}

    .badge {{ text-align: center; margin-bottom: 15px; }}
    .badge span {{
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        padding: 7px 16px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
    }}

    .title {{
        text-align: center;
        color: #111827;
        font-size: clamp(1.8rem, 5vw, 2.4rem);
        font-weight: 800;
        margin-bottom: 8px;
    }}

    .subtitle {{
        text-align: center;
        color: #64748b;
        font-size: 16px;
        line-height: 1.6;
        margin-bottom: 20px;
    }}

    .section-title {{
        color: #111827;
        font-size: 21px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 5px;
    }}

    .section-text {{
        color: #64748b;
        font-size: 14px;
        margin-bottom: 12px;
    }}

    [data-testid="stFileUploader"] section {{
        background: #f8fafc;
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
    }}

    [data-testid="stFileUploader"] section:hover {{
        border-color: #6366f1;
        background: #f8faff;
    }}

    /* tab text ("Upload image" and "Use camera") was not visible.
       I use button[role="tab"] because the class names change
       between streamlit versions */
    button[role="tab"],
    button[role="tab"] p,
    button[role="tab"] div,
    button[role="tab"] span,
    [data-baseweb="tab"],
    [data-baseweb="tab"] p,
    [data-baseweb="tab"] div,
    [data-baseweb="tab"] span,
    [data-testid="stTab"],
    [data-testid="stTab"] p {{
        color: #111827 !important;
        opacity: 1 !important;
        font-weight: 600;
    }}

    /* the tab that is currently selected */
    button[role="tab"][aria-selected="true"],
    button[role="tab"][aria-selected="true"] p,
    button[role="tab"][aria-selected="true"] div,
    button[role="tab"][aria-selected="true"] span,
    [data-baseweb="tab"][aria-selected="true"],
    [data-baseweb="tab"][aria-selected="true"] p {{
        color: #4338ca !important;
    }}

    /* line under the selected tab and the grey line under all tabs */
    [data-baseweb="tab-highlight"] {{
        background-color: #4338ca !important;
    }}
    [data-baseweb="tab-border"] {{
        background-color: #e5e7eb !important;
    }}

    /* file uploader and camera text (was white on white) */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] span,
    [data-testid="stCameraInput"] label,
    [data-testid="stCameraInput"] div,
    [data-testid="stCameraInput"] span {{
        color: #111827 !important;
    }}

    /* small grey text like "200MB per file" */
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] div,
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {{
        color: #475569 !important;
    }}

    /* name of the uploaded file */
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploaderFile"] small {{
        color: #111827 !important;
    }}

    /* browse files button */
    [data-testid="stFileUploader"] button {{
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #cbd5e1 !important;
    }}
    [data-testid="stFileUploader"] button:hover {{
        border-color: #6366f1 !important;
        color: #4338ca !important;
    }}
    [data-testid="stFileUploader"] button span,
    [data-testid="stFileUploader"] button p {{
        color: inherit !important;
    }}

    /* camera buttons */
    [data-testid="stCameraInput"] button {{
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #cbd5e1 !important;
    }}
    [data-testid="stCameraInput"] button span,
    [data-testid="stCameraInput"] button p {{
        color: inherit !important;
    }}

    /* captions under the images */
    [data-testid="stImageCaption"],
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {{
        color: #475569 !important;
    }}

    /* success, info and warning boxes */
    [data-testid="stAlert"] {{
        background: #f1f5f9 !important;
        border: 1px solid #e2e8f0 !important;
    }}
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] div,
    [data-testid="stAlert"] span {{
        color: #111827 !important;
    }}

    /* technical details expander */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {{
        color: #111827 !important;
    }}

    [data-testid="stMetric"] {{
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 15px;
        padding: 14px 16px;
        box-shadow: 0 5px 15px rgba(15, 23, 42, 0.06);
    }}

    [data-testid="stMetricLabel"] p {{ color: #64748b !important; }}
    [data-testid="stMetricValue"] {{ color: #111827 !important; }}

    .prediction-card {{
        background: linear-gradient(135deg, #eef2ff, #f5f3ff);
        border: 1px solid #c7d2fe;
        border-radius: 18px;
        padding: 22px;
        text-align: center;
        margin: 15px 0;
    }}

    .prediction-label {{
        color: #6366f1;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
    }}

    .prediction-number {{
        color: #111827;
        font-size: clamp(3.5rem, 12vw, 4.5rem);
        font-weight: 800;
        line-height: 1.1;
        margin-top: 8px;
    }}

    /* probability bars */
    .bar-row {{ display: flex; align-items: center; gap: 12px; margin: 7px 0; }}
    .bar-digit {{ width: 18px; font-weight: 800; color: #111827; text-align: center; }}
    .bar-bg {{ flex: 1; height: 12px; background: #edf0f7; border-radius: 10px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: #b8c0e8; border-radius: 10px; }}
    .bar-fill.best {{ background: linear-gradient(90deg, #4f46e5, #7c3aed); }}
    .bar-percent {{ width: 62px; text-align: right; font-size: 13px; color: #64748b; }}

    .empty-box {{
        text-align: center;
        padding: 30px 20px;
        background: #f8fafc;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        margin-top: 15px;
    }}
    .empty-title {{ color: #334155; font-size: 19px; font-weight: 700; margin-bottom: 6px; }}
    .empty-text {{ color: #64748b; font-size: 14px; }}

    .about-box {{
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px;
        margin-top: 30px;
    }}
    .about-heading {{ color: #111827; font-size: 19px; font-weight: 700; margin-bottom: 8px; }}
    .about-content {{ color: #64748b; font-size: 14px; line-height: 1.7; }}

    .tag {{
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        margin: 12px 5px 0 0;
    }}

    .footer {{
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        margin-top: 25px;
        padding-top: 15px;
        border-top: 1px solid #e5e7eb;
    }}

    @media (max-width: 640px) {{
        .st-key-main_card {{ padding: 1.1rem; border-radius: 18px; }}
    }}
    </style>
    """
)

# ---------------- load the model ----------------

try:
    model = load_model()
except Exception as error:
    st.error("Could not load the model. Check that models/digit_cnn_model.keras exists.")
    st.code(str(error))
    st.stop()

# ---------------- main page ----------------

with st.container(key="main_card"):

    show_html(
        """
        <div class="badge"><span>DEEP LEARNING • COMPUTER VISION</span></div>
        <div class="title">Handwritten Digit Recognition</div>
        <div class="subtitle">
            Upload or capture a handwritten digit and let our
            <strong>CNN model</strong> predict which number it is.
        </div>
        """
    )

    show_html(
        """
        <div class="section-title">Add Your Image</div>
        <div class="section-text">
            Use one digit in dark ink on plain paper. Formats: PNG, JPG, JPEG.
        </div>
        """
    )

    upload_tab, camera_tab = st.tabs(["Upload image", "Use camera"])

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Choose a handwritten digit image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )

    with camera_tab:
        camera_photo = st.camera_input(
            "Take a photo of the digit",
            label_visibility="collapsed",
        )

    # use whichever one the user gave
    image_file = uploaded_file or camera_photo

    if image_file is None:
        show_html(
            """
            <div class="empty-box">
                <div class="empty-title">Ready to recognize your digit</div>
                <div class="empty-text">Upload an image or use the camera to start.</div>
            </div>
            """
        )

    else:
        try:
            original_image = Image.open(image_file)
            processed_image = prepare_image(original_image)

            # scale to 0-1 and reshape to what the CNN expects
            image_array = np.array(processed_image, dtype="float32") / 255.0
            image_array = image_array.reshape(1, 28, 28, 1)

            prediction = model.predict(image_array, verbose=0)[0]
            probabilities = prediction * 100
            predicted_digit = int(np.argmax(prediction))
            confidence = float(np.max(prediction) * 100)

            # the digit with the second highest probability
            second_digit = int(np.argsort(prediction)[-2])
            second_value = float(probabilities[second_digit])

            # show both images
            show_html('<div class="section-title">Your Image</div>')
            col_original, col_processed = st.columns(2)

            with col_original:
                st.image(original_image, caption="Original image", use_container_width=True)

            with col_processed:
                st.image(
                    processed_image.resize((224, 224), Image.NEAREST),
                    caption="What the model sees (28 × 28)",
                    use_container_width=True,
                )

            # show the result
            show_html('<div class="section-title">Prediction Result</div>')
            show_html(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">PREDICTED DIGIT</div>
                    <div class="prediction-number">{predicted_digit}</div>
                </div>
                """
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Digit", predicted_digit)
            col2.metric("Confidence", f"{confidence:.2f}%")
            col3.metric("Second Guess", f"{second_digit} ({second_value:.1f}%)")

            # message depends on how sure the model is
            if confidence >= 90:
                st.success(f"The model is highly confident that the digit is {predicted_digit}.")
            elif confidence >= 70:
                st.info(f"The model predicts {predicted_digit} with moderate confidence.")
            else:
                st.warning(
                    f"The model predicts {predicted_digit}, but the confidence is low. "
                    "Try a thicker stroke or a clearer photo."
                )

            # probability bars for all 10 digits
            show_html(
                """
                <div class="section-title">Prediction Probabilities</div>
                <div class="section-text">Probability given by the model to each digit.</div>
                """
            )

            bars = ""
            for digit, probability in enumerate(probabilities):
                best = "best" if digit == predicted_digit else ""
                bars += f"""
                <div class="bar-row">
                    <div class="bar-digit">{digit}</div>
                    <div class="bar-bg">
                        <div class="bar-fill {best}" style="width:{min(float(probability), 100):.2f}%"></div>
                    </div>
                    <div class="bar-percent">{float(probability):.2f}%</div>
                </div>
                """
            show_html(bars)

        except Exception as error:
            st.error("Unable to process the image. Please try a clear PNG or JPG.")
            with st.expander("Technical Details"):
                st.code(str(error))

    show_html(
        """
        <div class="about-box">
            <div class="about-heading">About This Project</div>
            <div class="about-content">
                This application uses a Convolutional Neural Network (CNN) to
                recognize handwritten digits. The image is converted to grayscale,
                inverted and centred like MNIST, resized to
                <strong>28 × 28 pixels</strong>, normalized and passed to the
                trained model. The result shows the predicted digit, confidence
                and the probability of all ten classes.
            </div>
            <div>
                <span class="tag">Python</span>
                <span class="tag">TensorFlow</span>
                <span class="tag">Keras</span>
                <span class="tag">CNN</span>
                <span class="tag">Streamlit</span>
                <span class="tag">MNIST</span>
            </div>
        </div>
        <div class="footer">
            Handwritten Digit Recognition | CNN Deep Learning Project<br>
            Built with Python, TensorFlow and Streamlit
        </div>
        """
    )