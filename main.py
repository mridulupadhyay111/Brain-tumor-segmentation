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

import streamlit as st
from download_model import download_model
from model import ViT_UNet

# Pre-load/verify model weights
download_model()

# Must be the first Streamlit command
st.set_page_config(
    page_title="NEURO-HUD | Diagnostic Workstation",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)
load_dotenv()

def inject_light_modern_ui_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@500;700&display=swap');

        /* 1. Positive Light Background & Dark Text for Readability */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            font-family: 'Inter', sans-serif !important;
            background: linear-gradient(135deg, #f5f7fa 0%, #e4eaf5 100%) !important;
            color: #1e293b !important; /* Dark Slate for standard text */
        }

        /* Override markdown text to ensure paragraphs are visible */
        .stMarkdown p {
            color: #334155 !important; 
        }

        /* 2. Hide default Streamlit header */
        [data-testid="stHeader"] {
            background: transparent !important;
        }

        /* 3. Clinical Glassmorphic Top Nav (Light) */
        .glass-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-bottom: 1px solid rgba(14, 165, 233, 0.2);
            padding: 1rem 2rem;
            margin: -2rem -2rem 2rem -2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        }
        .nav-brand {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.6rem;
            font-weight: 700;
            background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.05em;
        }
        .nav-status {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: #0284c7;
            background: rgba(14, 165, 233, 0.1);
            border: 1px solid rgba(14, 165, 233, 0.3);
            padding: 6px 12px;
            border-radius: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 700;
        }
        .nav-status::before {
            content: '';
            width: 8px;
            height: 8px;
            background-color: #0ea5e9;
            border-radius: 50%;
            box-shadow: 0 0 8px #0ea5e9;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(14, 165, 233, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(14, 165, 233, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(14, 165, 233, 0); }
        }

        /* 4. Elegant Step Tracker (Light) */
        .step-container {
            display: flex;
            gap: 15px;
            margin-bottom: 2rem;
        }
        .step-pill {
            flex: 1;
            text-align: center;
            padding: 12px;
            background: rgba(255, 255, 255, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 12px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.9rem;
            font-weight: 600;
            color: #64748b;
            transition: all 0.3s ease;
        }
        .step-pill.active {
            background: rgba(14, 165, 233, 0.1);
            border: 1px solid rgba(14, 165, 233, 0.5);
            color: #0284c7;
            box-shadow: 0 4px 15px rgba(14, 165, 233, 0.15);
        }

        /* 5. Medical Blue Primary Buttons */
        .stButton > button {
            width: 100%;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: 1px !important;
            border-radius: 12px !important;
            padding: 0.8rem 1.5rem !important;
            background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%) !important;
            color: #ffffff !important;
            border: none !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.2) !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(37, 99, 235, 0.35) !important;
        }
        
        /* Secondary Button (Reset / New Scan) */
        button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.8) !important;
            color: #334155 !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
        }
        button[kind="secondary"]:hover {
            background: #ffffff !important;
            border-color: #94a3b8 !important;
        }

        /* 6. Form Inputs & Metric Cards (High Contrast Light) */
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, section[data-testid="stFileUploader"] {
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 12px !important;
            color: #0f172a !important; /* Ensure input text is very dark */
            transition: border-color 0.3s ease;
        }
        div[data-baseweb="input"] > div:focus-within, section[data-testid="stFileUploader"]:hover {
            border-color: #0ea5e9 !important;
            box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.2) !important;
        }
        
        /* Metric Cards */
        div[data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-left: 4px solid #0ea5e9 !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
        }
        div[data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
            color: #0284c7 !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #475569 !important;
            font-weight: 600 !important;
        }

        /* Section Titles */
        .section-title {
            font-family: 'Space Grotesk', sans-serif;
            color: #64748b;
            font-size: 0.9rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 1rem;
            margin-top: 1rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .section-title::after {
            content: "";
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, rgba(0,0,0,0.1) 0%, transparent 100%);
        }

        /* 7. Mobile Optimization Constraints */
        @media (max-width: 768px) {
            .glass-nav {
                flex-direction: column;
                gap: 15px;
                padding: 1rem;
                margin: -1rem -1rem 1.5rem -1rem;
                text-align: center;
            }
            .step-container {
                flex-direction: column;
                gap: 8px;
            }
            .block-container {
                padding: 1rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_light_modern_ui_css()

# --- Email Function Implementation ---
def send_email_with_pdf(sender_email, sender_password, receiver_email, subject, body, pdf_path):
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with open(pdf_path, "rb") as f:
            attach = MIMEApplication(f.read(), _subtype="pdf")
            attach.add_header('Content-Disposition', 'attachment', filename=os.path.basename(pdf_path))
            msg.attach(attach)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        st.toast("✅ Medical report successfully dispatched via secure email.", icon="📧")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# --- Helper Functions ---
def get_credentials():
    sender_email = os.getenv("sender_email") or os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("sender_password") or os.getenv("SENDER_PASSWORD")
    gemini_api_key = os.getenv("gemini_api_key") or os.getenv("GEMINI_API_KEY")
    return sender_email, sender_password, gemini_api_key

def generate_pdf(name, age, gender, report_text, original_image, overlay_image):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(15, 23, 42) # Slate 900
    pdf.cell(0, 10, txt="Clinical Imaging Analysis Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(51, 65, 85) # Slate 700
    pdf.cell(95, 10, txt=f"Patient Name: {name}", ln=False)
    pdf.cell(95, 10, txt=f"Age: {age}", ln=True, align="R")
    pdf.cell(95, 10, txt=f"Gender: {gender}", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, txt="Imaging Modality Results", ln=True)

    original_image.save("temp_original.png")
    overlay_image.save("temp_overlay.png")

    pdf.image("temp_original.png", x=15, y=pdf.get_y(), w=80)
    pdf.image("temp_overlay.png", x=115, y=pdf.get_y(), w=80)
    pdf.ln(85)

    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.set_xy(15, pdf.get_y())
    pdf.cell(80, 10, "Original Source MRI", align="C")
    pdf.set_xy(115, pdf.get_y())
    pdf.cell(80, 10, "AI Segmentation Overlay", align="C")
    pdf.ln(15)

    if os.path.exists("temp_original.png"): os.remove("temp_original.png")
    if os.path.exists("temp_overlay.png"): os.remove("temp_overlay.png")

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, txt="Automated AI Analysis", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(51, 65, 85)

    cleaned = report_text.replace("*", "").replace("#", "")
    pdf.multi_cell(0, 7, txt=cleaned)
    output_path = "Imaging_Analysis_Report.pdf"
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

def create_dynamic_overlay(original_pil, mask_pil, alpha=0.6):
    original_np = np.array(original_pil.convert("RGB"))
    mask_np = np.array(mask_pil.convert("L"))
    colored_mask = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
    # Vibrant Cyan/Teal for high visibility against grayscale MRI
    colored_mask[mask_np > 0] = (254, 242, 0) # BGR format
    overlay_np = cv2.addWeighted(original_np, 1, colored_mask, alpha, 0)
    return Image.fromarray(overlay_np)

def calculate_tumor_volume(mask_pil):
    mask_np = np.array(mask_pil)
    tumor_pixels = np.sum(mask_np > 0)
    volume = tumor_pixels * 0.005
    return f"{volume:.2f} cm³"

def get_clinical_report(prompt, original_pil, overlay_image, api_key):
    if not api_key:
        return "⚠️ Cloud reasoning engine offline. Please check API configuration."
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, original_pil, overlay_image]
        )
        return response.text
    except Exception as exc:
        st.error(f"Error executing report request: {exc}")
        return "Report generation failed."


# --- Core App Execution ---
sender_email, sender_password, api_key = get_credentials()

# Top Navigation (Light Glassmorphic)
st.markdown(
    """
    <div class="glass-nav">
        <div class="nav-brand">Clinical Vision Portal</div>
        <div class="nav-status">SECURE CONNECTION :: SYSTEM ACTIVE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "processed" not in st.session_state:
    st.session_state.processed = False

# Step Navigation Breadcrumbs
s1 = "active" if not st.session_state.processed else ""
s2 = "active" if not st.session_state.processed else ""
s3 = "active" if st.session_state.processed else ""

st.markdown(
    f"""
    <div class="step-container">
        <div class="step-pill {s1}">01. Patient Registry</div>
        <div class="step-pill {s2}">02. Imaging Upload</div>
        <div class="step-pill {s3}">03. Clinical Diagnostics</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- VIEW 1: DATA INGESTION ---
if not st.session_state.processed:
    st.markdown("<div class='section-title'>PATIENT ONBOARDING</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        recipient_name = st.text_input("Patient Full Name", placeholder="e.g. Alex Morgan")
        recipient_age = st.number_input("Age (Years)", min_value=0, max_value=120, value=30)
        recipient_gender = st.selectbox("Biological Sex", ["Male", "Female", "Other"])
        recipient_email = st.text_input("Report Delivery Email", placeholder="patient@clinic.com")

    with col2:
        uploaded_file = st.file_uploader(
            "Upload Brain MRI DICOM/Image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: .jpg, .png. Ensure scan is aligned."
        )
        st.markdown("<br>", unsafe_allow_html=True) 
        submit_btn = st.button("RUN CLINICAL ANALYSIS", use_container_width=True)

    if submit_btn:
        if not uploaded_file or not recipient_name or not recipient_email:
            st.warning("⚠️ Missing fields. Please complete registry and attach imaging data.")
        else:
            with st.spinner("Processing Neural Segmentation & Clinical Reasoning..."):
                model, device = load_model()
                image_bytes = uploaded_file.getvalue()
                original_pil, mask_pil = get_segmentation(image_bytes, model, device)

                original_pil_resized = original_pil.resize((224, 224))
                tumor_volume = calculate_tumor_volume(mask_pil)

                prompt = (
                    f"You are a helpful radiological assistant AI. Analyze the provided brain MRI and its cyan tumor segmentation overlay for patient {recipient_name} (Age: {recipient_age}, Gender: {recipient_gender}). "
                    f"The estimated tumor volume is {tumor_volume}."
                    "\n\nGenerate a preliminary radiological description. Structure the report with the following headings. Do not use markdown:"
                    "\n\n1. FINDINGS:\n- Location: Describe the anatomical location.\n- Size & Shape: Note the estimated volume and describe the shape.\n\n"
                    "2. PROBABLE TUMOR TYPE:\n- Suggest a probable type and briefly explain why.\n\n"
                    "3. RECOMMENDED FOLLOW-UP:\n- Suggest next medical tests or steps.\n\n"
                    "4. DISCLAIMER:\n- End with a clear disclaimer that this is an AI-generated analysis and not a substitute for professional medical diagnosis."
                )

                overlay_init = create_dynamic_overlay(original_pil_resized, mask_pil, alpha=0.6)
                report_text = get_clinical_report(prompt, original_pil_resized, overlay_init, api_key)

                st.session_state.processed = True
                st.session_state.recipient_name = recipient_name
                st.session_state.recipient_email = recipient_email
                st.session_state.recipient_age = recipient_age
                st.session_state.recipient_gender = recipient_gender
                st.session_state.original_pil = original_pil_resized
                st.session_state.mask_pil = mask_pil
                st.session_state.tumor_volume = tumor_volume
                st.session_state.report_text = report_text
                st.rerun()

# --- VIEW 2: DIAGNOSTIC DASHBOARD ---
if st.session_state.processed:
    
    # Header controls
    top_bar1, top_bar2 = st.columns([3, 1])
    with top_bar1:
        st.markdown(f"### 📋 Diagnostic Profile: **{st.session_state.recipient_name}**")
    with top_bar2:
        if st.button("NEW SCAN", type="secondary", use_container_width=True):
            st.session_state.processed = False
            st.rerun()

    # Visualizations & Metrics
    grid_a, grid_b = st.columns([1.2, 1], gap="large")

    with grid_a:
        st.markdown("<div class='section-title'>IMAGING TERMINAL</div>", unsafe_allow_html=True)
        overlay_alpha = st.slider("Segmentation Opacity", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        
        dynamic_overlay = create_dynamic_overlay(
            st.session_state.original_pil, 
            st.session_state.mask_pil, 
            alpha=overlay_alpha
        )
        st.image(dynamic_overlay, caption="Dynamic Multimodal Overlay View", use_container_width=True)

    with grid_b:
        st.markdown("<div class='section-title'>CLINICAL REASONING</div>", unsafe_allow_html=True)
        st.metric("Estimated Lesion Footprint", st.session_state.tumor_volume)
        
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("VIEW FULL RADIOLOGY REPORT", expanded=True):
            # Ensuring the text inside the expander is fully visible
            st.markdown(f"<div style='color: #334155;'>{st.session_state.report_text}</div>", unsafe_allow_html=True)

    # Export Area
    st.markdown("<div class='section-title'>EXPORT & COMMUNICATION</div>", unsafe_allow_html=True)
    
    with st.spinner("Compiling PDF Payload..."):
        pdf_path = generate_pdf(
            st.session_state.recipient_name,
            st.session_state.recipient_age,
            st.session_state.recipient_gender,
            st.session_state.report_text,
            st.session_state.original_pil,
            dynamic_overlay
        )

        col_out1, col_out2 = st.columns(2, gap="large")
        
        with col_out1:
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="DOWNLOAD REPORT (.PDF)",
                    data=pdf_file,
                    file_name=f"Report_{st.session_state.recipient_name.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        with col_out2:
            if sender_email and sender_password:
                if st.button("DISPATCH SECURE EMAIL TO PATIENT", use_container_width=True):
                    send_email_with_pdf(
                        sender_email,
                        sender_password,
                        st.session_state.recipient_email,
                        f"Secure Health Portal: Imaging Analysis Report - {st.session_state.recipient_name}",
                        f"Dear {st.session_state.recipient_name},\n\nPlease find attached your preliminary imaging analysis report.\n\nBest regards,\nAutomated Diagnostic Portal",
                        pdf_path
                    )
            else:
                st.info("Email credentials not configured in environment variables.", icon="ℹ️")