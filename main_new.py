import io
import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from fpdf import FPDF
from google import genai
from PIL import Image
from torchvision import transforms

from model import ViT_UNet
import streamlit as st

st.set_page_config(page_title="Brain Tumor Analysis", layout="wide")
load_dotenv()


def inject_css():
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
        .hero-card {
            background: linear-gradient(135deg, rgba(10,32,59,0.98), rgba(24,85,128,0.95));
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 24px;
            padding: 24px 26px;
            box-shadow: 0 14px 40px rgba(0,0,0,0.24);
            margin-bottom: 1rem;
        }
        .hero-card h1 { margin: 0 0 0.35rem 0; font-size: 2rem; color: #f7fbff; }
        .hero-card p { margin: 0; color: rgba(247,251,255,0.82); font-size: 0.97rem; line-height: 1.55; }
        .eyebrow { display: inline-block; padding: 6px 10px; background: rgba(255,255,255,0.12); color: #dff4ff; border-radius: 999px; font-size: 0.8rem; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.7rem; }
        .panel { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 1rem 1.05rem; margin-bottom: 1rem; }
        .pill-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.8rem; }
        .pill { background: rgba(255,255,255,0.1); color: #eff7ff; border-radius: 999px; padding: 6px 10px; font-size: 0.82rem; }
        .stButton > button { border-radius: 12px; padding: 0.65rem 1rem; font-weight: 600; background: linear-gradient(135deg, #33a7ff, #1d6ed6); border: none; color: white; box-shadow: 0 10px 24px rgba(29,110,214,0.24); }
        .stTextInput > div > div > input, .stSelectbox > div > div > div, .stFileUploader > div { border-radius: 12px; background: rgba(255,255,255,0.04); }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


def get_credentials():
    sender_email = os.getenv("sender_email") or os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("sender_password") or os.getenv("SENDER_PASSWORD")
    gemini_api_key = os.getenv("gemini_api_key") or os.getenv("GEMINI_API_KEY")
    return sender_email, sender_password, gemini_api_key


# PDF

def generate_pdf(name, age, gender, report_text, original_image, overlay_image):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, txt="Brain Tumor Analysis Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(95, 10, txt=f"Patient Name: {name}", ln=False)
    pdf.cell(95, 10, txt=f"Age: {age}", ln=True, align="R")
    pdf.cell(95, 10, txt=f"Gender: {gender}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, txt="Imaging Results", ln=True)
    original_image.save("temp_original.png")
    overlay_image.save("temp_overlay.png")
    pdf.image("temp_original.png", x=10, y=pdf.get_y(), w=90)
    pdf.image("temp_overlay.png", x=110, y=pdf.get_y(), w=90)
    pdf.ln(95)
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.set_xy(10, pdf.get_y())
    pdf.cell(90, 10, "Original MRI Image", align="C")
    pdf.set_xy(110, pdf.get_y())
    pdf.cell(90, 10, "Segmented Tumor (Overlay)", align="C")
    pdf.ln(15)
    os.remove("temp_original.png")
    os.remove("temp_overlay.png")
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, txt="AI-Generated Analysis", ln=True)
    pdf.set_font("Arial", "", 11)
    cleaned = report_text.replace("*", "").replace("#", "")
    pdf.multi_cell(0, 7, txt=cleaned)
    output_path = "Brain_Tumor_Report.pdf"
    pdf.output(output_path)
    return output_path


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT_UNet().to(device)
    model.load_state_dict(torch.load("vit_unet.pth", map_location=device))
    model.eval()
    return model, device


def get_segmentation(image_bytes, model, device):
    preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(input_tensor)
        preds = torch.nn.functional.interpolate(preds, size=(224, 224), mode="bilinear", align_corners=False)
        pred_mask_tensor = (preds > 0.5).float()
    mask_np = pred_mask_tensor.squeeze().cpu().numpy()
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
    return img, mask_pil


def create_overlay(original_pil, mask_pil, color=(0, 255, 0), alpha=0.6):
    original_np = np.array(original_pil.convert("RGB"))
    mask_np = np.array(mask_pil.convert("L"))
    colored_mask = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
    colored_mask[mask_np > 0] = color
    overlay_np = cv2.addWeighted(original_np, 1, colored_mask, alpha, 0)
    return Image.fromarray(overlay_np)


def calculate_tumor_volume(mask_pil):
    mask_np = np.array(mask_pil)
    tumor_pixels = np.sum(mask_np > 0)
    volume = tumor_pixels * 0.005
    return f"{volume:.2f} cm³"


def get_gemini_report(prompt, original_pil, overlay_image, gemini_api_key):
    if not gemini_api_key:
        return (
            "FINDINGS:\n- Location: The AI could not confirm anatomical details without API access.\n- Size & Shape: A preliminary estimate is available from the segmentation mask.\n\n"
            "PROBABLE TUMOR TYPE:\n- Suggestion: An assisted review is recommended for a definitive classification.\n\n"
            "RECOMMENDED FOLLOW-UP:\n- Consider specialist review, advanced MRI, and biopsy planning as appropriate.\n\n"
            "DISCLAIMER:\n- This report is an AI-assisted preliminary summary and is not a substitute for professional medical diagnosis."
        )
    try:
        client = genai.Client(api_key=gemini_api_key)
        response = client.models.generate_content(model="gemini-2.5-flash", contents=[prompt, original_pil, overlay_image])
        return response.text
    except Exception as exc:
        st.error(f"Error calling the Gemini API: {exc}")
        return "Could not generate a report due to an API error."


def send_email_with_pdf(sender_email, sender_password, recipient, subject, body, pdf_path):
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    with open(pdf_path, "rb") as file:
        attachment = MIMEApplication(file.read(), _subtype="pdf")
    attachment.add_header("Content-Disposition", "attachment", filename=str(pdf_path))
    msg.attach(attachment)
    try:
        with st.spinner("Connecting to mail server..."):
            server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient, msg.as_string())
            server.close()
        st.success(f"Report sent to {recipient}.")
    except Exception as exc:
        st.error(f"Email delivery failed: {exc}")


st.markdown(
    """
    <div class="hero-card">
        <div class="eyebrow">Clinical imaging workspace</div>
        <h1>Brain tumor segmentation with a more premium, polished experience</h1>
        <p>Upload a brain MRI scan to receive a refined segmentation preview, a concise AI-assisted report, and a shareable PDF summary.</p>
        <div class="pill-row">
            <span class="pill">Secure workflow</span>
            <span class="pill">Elegant interface</span>
            <span class="pill">Streamlit-ready</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

sender_email, sender_password, gemini_api_key = get_credentials()

sidebar = st.sidebar
sidebar.header("Patient details")
uploaded_file = sidebar.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"], help="PNG or JPG files work best")
recipient_email = sidebar.text_input("Email", placeholder="name@example.com")
recipient_name = sidebar.text_input("Patient name", placeholder="Alex Morgan")
recipient_age = sidebar.number_input("Age", min_value=0, max_value=100, value=35, step=1)
recipient_gender = sidebar.selectbox("Gender", ["Male", "Female", "Other"])
analysis_button = sidebar.button("Run analysis", type="primary", use_container_width=True)

if not sender_email or not sender_password:
    sidebar.warning("Email delivery will be skipped unless SMTP credentials are set.")

if not uploaded_file:
    st.markdown("""
    <div class="panel">
        <h3 style="margin-top:0;">What this experience includes</h3>
        <ul>
            <li>A cleaner clinical-style intake panel</li>
            <li>A focused preview for the segmentation overlay</li>
            <li>A structured report and PDF package for sharing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.info("Choose an MRI image in the sidebar to begin.")
    st.stop()

if analysis_button:
    if not recipient_email or not recipient_name:
        st.warning("Please provide patient details and a valid email address before continuing.")
        st.stop()

    with st.spinner("Preparing the segmentation workflow..."):
        model, device = load_model()
        image_bytes = uploaded_file.getvalue()
        original_pil, mask_pil = get_segmentation(image_bytes, model, device)

    original_pil_resized = original_pil.resize((224, 224))
    overlay_image = create_overlay(original_pil_resized, mask_pil, color=(0, 255, 0), alpha=0.6)
    tumor_volume = calculate_tumor_volume(mask_pil)

    left_col, right_col = st.columns([1.1, 0.9])
    with left_col:
        st.markdown("<div class='panel'><h3 style='margin-top:0;'>Segmentation preview</h3></div>", unsafe_allow_html=True)
        st.image(overlay_image, caption="Overlay showing the predicted lesion region", use_container_width=True)
        st.caption("This visualization is assistive and should be reviewed with clinical context.")

    with right_col:
        st.markdown("<div class='panel'><h3 style='margin-top:0;'>Clinical summary</h3></div>", unsafe_allow_html=True)
        st.metric("Estimated tumor volume", tumor_volume)
        prompt = (
            f"You are a helpful radiological assistant AI. Analyze the provided brain MRI and its green tumor segmentation overlay for patient {recipient_name} (Age: {recipient_age}, Gender: {recipient_gender}). "
            f"The estimated tumor volume is {tumor_volume}."
            "\n\nGenerate a preliminary radiological description. Structure the report with the following headings. Do not use markdown:"
            "\n\n1. FINDINGS:\n- Location: Describe the anatomical location.\n- Size & Shape: Note the estimated volume and describe the shape.\n\n"
            "2. PROBABLE TUMOR TYPE:\n- Suggest a probable type and briefly explain why.\n\n"
            "3. RECOMMENDED FOLLOW-UP:\n- Suggest next medical tests or steps.\n\n"
            "4. DISCLAIMER:\n- End with a clear disclaimer that this is an AI-generated analysis and not a substitute for professional medical diagnosis."
        )
        with st.spinner("Generating a structured AI report..."):
            report_text = get_gemini_report(prompt, original_pil_resized, overlay_image, gemini_api_key)
        st.text_area("Report output", report_text, height=250)
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.info("Generating the PDF package and sending the report when email credentials are available.")
        with st.spinner("Creating the final report package..."):
            pdf_path = generate_pdf(recipient_name, recipient_age, recipient_gender, report_text, original_pil_resized, overlay_image)
            if sender_email and sender_password:
                send_email_with_pdf(sender_email, sender_password, recipient_email, "Your Brain Tumor Analysis Report", f"Dear {recipient_name},\n\nPlease find attached your AI-generated report.\n\nThis report is intended for preliminary informational purposes only and should be reviewed by a qualified professional.\n\nBest regards,\nThe AI Health Assistant Team", pdf_path)
            else:
                st.warning("Email delivery was skipped because the required credentials were not found.")
        st.markdown("</div>", unsafe_allow_html=True)
