from download_model import download_model

download_model()
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
import torch

model = torch.load(
    "vit_unet.pth",
    map_location="cpu"
)

st.set_page_config(
    page_title="Brain Tumor Analysis Workspace",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_dotenv()


def inject_css():
    st.markdown(
        """
        <style>
        /* Base page theme */
        .stApp {
            background-color: #f8fafc;
            color: #0f172a;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1300px;
        }

        /* Hero Header */
        .hero-card {
            background-color: #0f172a;
            border-radius: 12px;
            padding: 24px 28px;
            margin-bottom: 1.5rem;
            border: 1px solid #1e293b;
        }
        .hero-card h1 {
            margin: 0 0 0.5rem 0;
            font-size: 1.8rem;
            font-weight: 700;
            color: #ffffff !important;
        }
        .hero-card p {
            margin: 0;
            color: #94a3b8 !important;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        .eyebrow {
            display: inline-block;
            padding: 2px 10px;
            background-color: #1e293b;
            color: #60a5fa;
            border: 1px solid #334155;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
        }

        /* Panels */
        .panel {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1.2rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        }
        .panel h3 {
            color: #0f172a !important;
            font-size: 1.1rem;
            font-weight: 700;
            margin-top: 0;
            margin-bottom: 0.8rem;
        }

        /* Metric Styling */
        div[data-testid="stMetric"] {
            background-color: #f1f5f9 !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
            padding: 0.85rem 1rem !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: #475569 !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #1d4ed8 !important;
            font-size: 1.6rem !important;
            font-weight: 700 !important;
        }

        /* Summary Cards */
        .summary-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #2563eb;
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .summary-title {
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
            color: #0f172a;
        }
        .summary-body {
            font-size: 0.9rem;
            line-height: 1.6;
            color: #334155;
        }
        .status-pill {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
            background-color: #dbeafe;
            color: #1e40af;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
        }

        /* Button Styling */
        .stButton > button {
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 600 !important;
            background-color: #2563eb !important;
            border: none !important;
            color: #ffffff !important;
            transition: background-color 150ms ease !important;
        }
        .stButton > button:hover {
            background-color: #1d4ed8 !important;
        }

        /* FIXED SIDEBAR VISIBILITY & CONTRAST */
        section[data-testid="stSidebar"] {
            background-color: #1e293b !important;
            border-right: 1px solid #334155 !important;
        }

        /* Sidebar headers & labels */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] p {
            color: #f8fafc !important;
        }

        /* Form input elements in sidebar */
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] div[data-baseweb="select"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border-radius: 6px !important;
            border: 1px solid #cbd5e1 !important;
        }

        /* Number input adjust buttons */
        section[data-testid="stSidebar"] button[aria-label="Increase"],
        section[data-testid="stSidebar"] button[aria-label="Decrease"] {
            background-color: #e2e8f0 !important;
            color: #0f172a !important;
        }

        /* File uploader container in sidebar */
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
            background-color: #0f172a !important;
            border: 1px dashed #475569 !important;
            border-radius: 8px !important;
            padding: 0.5rem !important;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
            background-color: #334155 !important;
            color: #ffffff !important;
            border: none !important;
        }
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


# PDF Generation

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
            "FINDINGS:\n- Location: Anatomical location pending API integration.\n- Size & Shape: Lesion footprint calculated from segmentation mask.\n\n"
            "PROBABLE TUMOR TYPE:\n- Suggestion: Clinical evaluation required for differential diagnosis.\n\n"
            "RECOMMENDED FOLLOW-UP:\n- Specialist consultation and contrast MRI suggested.\n\n"
            "DISCLAIMER:\n- Automated preliminary assessment. Not a substitute for expert medical diagnosis."
        )
    try:
        client = genai.Client(api_key=gemini_api_key)
        response = client.models.generate_content(model="gemini-2.5-flash", contents=[prompt, original_pil, overlay_image])
        return response.text
    except Exception as exc:
        st.error(f"Error executing report request: {exc}")
        return "Report generation failed."


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
        st.success(f"Report emailed to {recipient}.")
    except Exception as exc:
        st.error(f"Email delivery failed: {exc}")


def build_compact_summary(report_text, tumor_volume, patient_name):
    lines = [line.strip() for line in report_text.splitlines() if line.strip()]
    bullets = []
    for line in lines:
        if line.upper().startswith(("FINDINGS", "PROBABLE", "RECOMMENDED", "DISCLAIMER")):
            continue
        if line.startswith("-"):
            bullets.append(line[1:].strip())
        elif line and not line.startswith(("1.", "2.", "3.", "4.")):
            bullets.append(line)
        if len(bullets) >= 4:
            break

    if not bullets:
        bullets = ["Primary scan segmentation complete.", "Clinical evaluation recommended."]

    summary_html = f"""
    <div class="summary-card">
        <div class="status-pill">Clinical Overview</div>
        <div class="summary-title">{patient_name}'s Scan Findings</div>
        <div class="summary-body">
            <strong>Estimated Volume:</strong> {tumor_volume}<br>
            <strong>Observations:</strong><br>
            • {bullets[0]}<br>
            • {bullets[1] if len(bullets) > 1 else 'Follow-up recommended.'}<br>
            • {bullets[2] if len(bullets) > 2 else 'Validation required.'}
        </div>
    </div>
    """
    return summary_html


# --- Interface Layout ---

st.markdown(
    """
    <div class="hero-card">
        <div class="eyebrow">Radiology Workstation</div>
        <h1>Brain Tumor Segmentation & Analysis</h1>
        <p>Upload a brain MRI scan to process image segmentation, view calculated lesion volume, and build a downloadable diagnostic summary.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

sender_email, sender_password, gemini_api_key = get_credentials()

# Sidebar Controls
sidebar = st.sidebar
sidebar.header("Patient Setup")
uploaded_file = sidebar.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"], help="JPG or PNG image format supported.")
recipient_email = sidebar.text_input("Patient Email", placeholder="patient@example.com")
recipient_name = sidebar.text_input("Patient Name", placeholder="Alex Morgan")
recipient_age = sidebar.number_input("Age", min_value=0, max_value=120, value=35, step=1)
recipient_gender = sidebar.selectbox("Gender", ["Male", "Female", "Other"])
analysis_button = sidebar.button("Run Segmentation", type="primary", use_container_width=True)

if not sender_email or not sender_password:
    sidebar.info("Email delivery is disabled until SMTP credentials are added to environment variables.")

if not uploaded_file:
    st.markdown("""
    <div class="panel">
        <h3>System Overview</h3>
        <p style="color: #475569; font-size: 0.95rem; line-height: 1.6; margin: 0;">
            This tool processes brain MRI scans using a Vision Transformer (ViT-UNet) model to assist clinicians with lesion boundary identification, volume estimates, and automated report packaging.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.info("Upload an MRI image from the sidebar panel to start.")
    st.stop()

if analysis_button:
    if not recipient_email or not recipient_name:
        st.warning("Please provide patient name and email address before proceeding.")
        st.stop()

    with st.spinner("Processing segmentation mask..."):
        model, device = load_model()
        image_bytes = uploaded_file.getvalue()
        original_pil, mask_pil = get_segmentation(image_bytes, model, device)

    original_pil_resized = original_pil.resize((224, 224))
    overlay_image = create_overlay(original_pil_resized, mask_pil, color=(0, 255, 0), alpha=0.6)
    tumor_volume = calculate_tumor_volume(mask_pil)

    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.markdown("<div class='panel'><h3>Segmentation Overlay</h3>", unsafe_allow_html=True)
        st.image(overlay_image, caption="ViT-UNet Segmentation Boundary Overlay", use_container_width=True)
        st.caption("Green overlay highlights identified lesion footprint.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='panel'><h3>Quantitative Assessment</h3>", unsafe_allow_html=True)
        st.metric("Estimated Lesion Volume", tumor_volume)

        prompt = (
            f"You are a helpful radiological assistant AI. Analyze the provided brain MRI and its green tumor segmentation overlay for patient {recipient_name} (Age: {recipient_age}, Gender: {recipient_gender}). "
            f"The estimated tumor volume is {tumor_volume}."
            "\n\nGenerate a preliminary radiological description. Structure the report with the following headings. Do not use markdown:"
            "\n\n1. FINDINGS:\n- Location: Describe the anatomical location.\n- Size & Shape: Note the estimated volume and describe the shape.\n\n"
            "2. PROBABLE TUMOR TYPE:\n- Suggest a probable type and briefly explain why.\n\n"
            "3. RECOMMENDED FOLLOW-UP:\n- Suggest next medical tests or steps.\n\n"
            "4. DISCLAIMER:\n- End with a clear disclaimer that this is an AI-generated analysis and not a substitute for professional medical diagnosis."
        )

        with st.spinner("Building summary report..."):
            report_text = get_gemini_report(prompt, original_pil_resized, overlay_image, gemini_api_key)

        st.markdown(build_compact_summary(report_text, tumor_volume, recipient_name), unsafe_allow_html=True)

        with st.expander("View Full Text Report", expanded=False):
            st.text(report_text)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'><h3>Export Options</h3>", unsafe_allow_html=True)
    with st.spinner("Compiling PDF document..."):
        pdf_path = generate_pdf(recipient_name, recipient_age, recipient_gender, report_text, original_pil_resized, overlay_image)
        if sender_email and sender_password:
            send_email_with_pdf(
                sender_email,
                sender_password,
                recipient_email,
                f"Brain Analysis Report - {recipient_name}",
                f"Dear {recipient_name},\n\nPlease find attached the preliminary brain tumor analysis report.\n\nBest regards,\nClinical Assistant Team",
                pdf_path
            )
        else:
            st.warning("Email transmission skipped: SMTP credentials not provided.")
    st.markdown("</div>", unsafe_allow_html=True)