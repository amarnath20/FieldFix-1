import streamlit as st  # type: ignore[import-untyped]
import os
from PIL import Image
import io
import base64
import time
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np

# Set up environment variables and Gemini API
load_dotenv()

# Configure page settings
st.set_page_config(
    page_title="FeildFix - Agricultural Image Analysis",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define enhanced color scheme
COLORS = {
    "primary": "#1B5E20",  # Dark Green
    "secondary": "#795548",  # Brown
    "tertiary": "#C8E6C9",  # Very Light Green
    "background": "#F8F9FA",  # Light Gray
    "accent": "#FF8F00",  # Amber
    "text": "#212121",  # Almost Black
    "light_text": "#FFFFFF",  # White
    "result_bg": "#FFFFFF",  # White
    "result_border": "#E0E0E0",  # Light Gray
    "tab_inactive": "#81C784",  # Medium Green
}

# Custom CSS for improved styling
def local_css():
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {COLORS["background"]};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 10px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {COLORS["tab_inactive"]};
            border-radius: 8px 8px 0px 0px;
            padding: 12px 24px;
            color: {COLORS["light_text"]};
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {COLORS["primary"]};
            color: {COLORS["light_text"]};
            transform: translateY(-3px);
        }}
        h1, h2, h3 {{
            color: {COLORS["primary"]};
            font-weight: 700;
        }}
        .stButton>button {{
            background-color: {COLORS["accent"]};
            color: {COLORS["light_text"]};
            border-radius: 8px;
            font-weight: 600;
            border: none;
            padding: 8px 20px;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: #F57C00;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }}
        .upload-section {{
            background-color: {COLORS["result_bg"]};
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
            margin-bottom: 24px;
            border: 1px solid {COLORS["result_border"]};
        }}
        .results-section {{
            background-color: {COLORS["result_bg"]};
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
            margin-top: 24px;
            border-left: 5px solid {COLORS["accent"]};
            color: {COLORS["text"]};
            line-height: 1.6;
        }}
        .results-section h1, .results-section h2, .results-section h3 {{
            color: {COLORS["primary"]};
            margin-top: 16px;
            margin-bottom: 12px;
        }}
        .results-section ul {{
            margin-bottom: 16px;
        }}
        .info-box {{
            background-color: {COLORS["tertiary"]};
            padding: 18px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 5px solid {COLORS["primary"]};
        }}
        .app-header {{
            padding: 20px;
            text-align: center;
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, #388E3C 100%);
            border-radius: 12px;
            margin-bottom: 24px;
            color: white;
        }}
        .tab-content {{
            padding: 20px;
            background-color: white;
            border-radius: 0px 12px 12px 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }}
        .stTextInput>div>div>input {{
            border-radius: 8px;
        }}
        .stTextArea>div>div>textarea {{
            border-radius: 8px;
        }}
        .stFileUploader>div>button {{
            background-color: {COLORS["tertiary"]};
            color: {COLORS["primary"]};
        }}
        .divider {{
            height: 3px;
            background: linear-gradient(90deg, {COLORS["primary"]} 0%, {COLORS["tertiary"]} 100%);
            margin: 20px 0;
            border-radius: 3px;
        }}
        .container-card {{
            background-color: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
            border: 1px solid #E0E0E0;
        }}
        .spinner {{
            text-align: center;
            color: {COLORS["primary"]};
            font-weight: 600;
            margin: 20px 0;
        }}
    </style>
    """, unsafe_allow_html=True)

# Initialize Gemini API
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found. Please add it to your .env file.")
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-3-flash-preview')
except Exception as e:
    st.error(f"Error initializing Gemini API: {str(e)}")

# Helper function to process uploaded image
def process_uploaded_image(uploaded_file):
    if uploaded_file is None:
        return None
    
    try:
        # Read image
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Create container with border for the image
        with st.container():
            st.markdown('<div style="padding: 10px; border: 2px solid #E0E0E0; border-radius: 10px;">', 
                        unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        return image
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Helper function to get image data for Gemini API
def get_image_data(image):
    if image is None:
        return None
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
    
    return {
        "mime_type": "image/jpeg",
        "data": image_bytes
    }

# Analysis functions for each feature
def analyze_pest(image, additional_info=""):
    prompt = f"""
    Analyze this image and identify any agricultural pests present.
    
    For each pest identified, please provide:
    1. Name of the pest (common and scientific)
    2. Detailed description
    3. Potential damage they cause to plants/crops
    4. Recommended treatment options and prevention methods
    5. Natural predators if applicable
    
    Additional context provided by the user: {additional_info}
    
    Format your response with clear headings and bullet points for readability.
    """
    
    image_data = get_image_data(image)
    if not image_data:
        return "Failed to process image"
    
    try:
        response = model.generate_content([prompt, image_data])
        return response.text
    except Exception as e:
        return f"Error during analysis: {str(e)}"

def analyze_ripeness(image, fruit_type="", additional_info=""):
    prompt = f"""
    Analyze this image and assess the ripeness level of the fruit shown.
    
    Fruit type (if specified by user): {fruit_type}
    
    Please provide:
    1. Identification of the fruit (if not specified by user)
    2. Current ripeness level (e.g., underripe, ripe, overripe)
    3. Visual indicators of ripeness present in the image
    4. Estimated time until optimal ripeness (if underripe)
    5. Storage recommendations based on current state
    6. Expected shelf life
    7. Optimal uses based on current ripeness level
    
    Additional context provided by the user: {additional_info}
    
    Format your response with clear headings and bullet points for readability.
    """
    
    image_data = get_image_data(image)
    if not image_data:
        return "Failed to process image"
    
    try:
        response = model.generate_content([prompt, image_data])
        return response.text
    except Exception as e:
        return f"Error during analysis: {str(e)}"

def analyze_disease(image, plant_type="", additional_info=""):
    prompt = f"""
    Analyze this image and identify any plant diseases or disorders present.
    
    Plant type (if specified by user): {plant_type}
    
    For each disease identified, please provide:
    1. Name of the disease (common and scientific)
    2. Detailed description of symptoms visible in the image
    3. Pathogen or cause of the disease
    4. How the disease spreads and conditions that favor it
    5. Recommended treatment methods (chemical and organic options)
    6. Prevention strategies
    7. Potential impact on yield or plant health if left untreated
    
    Additional context provided by the user: {additional_info}
    
    Format your response with clear headings and bullet points for readability.
    """
    
    image_data = get_image_data(image)
    if not image_data:
        return "Failed to process image"
    
    try:
        response = model.generate_content([prompt, image_data])
        return response.text
    except Exception as e:
        return f"Error during analysis: {str(e)}"

def analyze_weed(image, location="", additional_info=""):
    prompt = f"""
    Analyze this image and identify any weed species present.
    
    Growing location/region (if specified by user): {location}
    
    For each weed identified, please provide:
    1. Name of the weed (common and scientific)
    2. Detailed description and identification features
    3. Life cycle and growth habits
    4. Impact on cultivated plants and agricultural systems
    5. Recommended control and removal techniques (mechanical, chemical, biological)
    6. Prevention strategies
    7. Any beneficial properties or uses if applicable
    
    Additional context provided by the user: {additional_info}
    
    Format your response with clear headings and bullet points for readability.
    """
    
    image_data = get_image_data(image)
    if not image_data:
        return "Failed to process image"
    
    try:
        response = model.generate_content([prompt, image_data])
        return response.text
    except Exception as e:
        return f"Error during analysis: {str(e)}"

# Main function to run the app
def main():
    local_css()
    
    # Attractive Header with Gradient
    st.markdown("""
    <div class="app-header">
        <h1 style='margin-bottom: 0px; color: white; font-size: 42px;'>🌱 FieldFix</h1>
        <h3 style='margin-top: 5px; color: white; opacity: 0.9;'>Smart Agricultural Image Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Introduction in an attractive box
    with st.expander("ℹ️ About FieldFix"):
        st.markdown("""
        <div class="info-box">
            <h3 style="margin-top: 0;">Welcome to FieldFix!</h3>
            <p>This intelligent tool helps farmers and gardeners identify common agricultural issues through advanced image analysis.</p>
            
            <p>With FieldFix, you can:</p>
            <ul>
                <li><strong>Identify agricultural pests</strong> and get treatment recommendations</li>
                <li><strong>Assess fruit ripeness</strong> to determine optimal harvest time</li>
                <li><strong>Diagnose plant diseases</strong> and learn prevention strategies</li>
                <li><strong>Identify weed species</strong> and understand control methods</li>
            </ul>
            
            <p>Simply upload clear images to the appropriate section and let our AI-powered system analyze them for you.</p>
            <p style="font-style: italic; margin-top: 15px;">Powered by Google's  gemini-3-flash-preview AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs for each feature with custom styling
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    tabs = st.tabs([
        "🐛 Pest Identification", 
        "🍎 Ripeness Assessment", 
        "🌡️ Disease Diagnosis", 
        "🌿 Weed Identification"
    ])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Pest Identification Tab
    with tabs[0]:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown(f"""
        <h2 style='color: {COLORS["primary"]}; display: flex; align-items: center;'>
            <span style='margin-right: 10px;'>🐛</span> Pest Identification
        </h2>
        <div class='divider'></div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown("""
            <div class="container-card">
                <h3 style='color: #1B5E20; margin-top: 0;'>Upload an image of a potential pest</h3>
                <p>Upload a clear image of the insect, larva, or evidence of pest activity on your plants.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                pest_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="pest_upload")
            
            with col2:
                additional_info = st.text_area("Additional information (optional)", 
                                            placeholder="e.g., Location, crop type, observed damage",
                                            key="pest_info",
                                            height=100)
            
            analyze_button = st.button("🔍 Analyze Pest", key="pest_button", use_container_width=True)
            
            if pest_file:
                pest_image = process_uploaded_image(pest_file)
                
                if analyze_button and pest_image:
                    st.markdown('<div class="spinner">Analyzing pest image... Please wait</div>', unsafe_allow_html=True)
                    with st.spinner(""):
                        result = analyze_pest(pest_image, additional_info)
                    
                    st.markdown(f"<div class='results-section'>{result}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Ripeness Assessment Tab
    with tabs[1]:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown(f"""
        <h2 style='color: {COLORS["primary"]}; display: flex; align-items: center;'>
            <span style='margin-right: 10px;'>🍎</span> Fruit Ripeness Assessment
        </h2>
        <div class='divider'></div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown("""
            <div class="container-card">
                <h3 style='color: #1B5E20; margin-top: 0;'>Upload an image of a fruit</h3>
                <p>Upload a clear image of the fruit to assess its ripeness level.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                ripeness_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="ripeness_upload")
                fruit_type = st.text_input("Fruit type (optional)", placeholder="e.g., Apple, Tomato, Banana", key="fruit_type")
            
            with col2:
                additional_info_ripeness = st.text_area("Additional information (optional)", 
                                                    placeholder="e.g., Variety, growing conditions",
                                                    key="ripeness_info",
                                                    height=100)
            
            analyze_ripeness_button = st.button("🔍 Analyze Ripeness", key="ripeness_button", use_container_width=True)
            
            if ripeness_file:
                ripeness_image = process_uploaded_image(ripeness_file)
                
                if analyze_ripeness_button and ripeness_image:
                    st.markdown('<div class="spinner">Analyzing fruit ripeness... Please wait</div>', unsafe_allow_html=True)
                    with st.spinner(""):
                        result = analyze_ripeness(ripeness_image, fruit_type, additional_info_ripeness)
                    
                    st.markdown(f"<div class='results-section'>{result}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Disease Diagnosis Tab
    with tabs[2]:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown(f"""
        <h2 style='color: {COLORS["primary"]}; display: flex; align-items: center;'>
            <span style='margin-right: 10px;'>🌡️</span> Plant Disease Diagnosis
        </h2>
        <div class='divider'></div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown("""
            <div class="container-card">
                <h3 style='color: #1B5E20; margin-top: 0;'>Upload an image of a plant with symptoms</h3>
                <p>Upload a clear image of plant leaves, stems, or whole plants showing symptoms of disease.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                disease_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="disease_upload")
                plant_type = st.text_input("Plant type (optional)", placeholder="e.g., Tomato, Rose, Wheat", key="plant_type")
            
            with col2:
                additional_info_disease = st.text_area("Additional information (optional)", 
                                                    placeholder="e.g., Symptoms progression, affected parts",
                                                    key="disease_info",
                                                    height=100)
            
            analyze_disease_button = st.button("🔍 Analyze Disease", key="disease_button", use_container_width=True)
            
            if disease_file:
                disease_image = process_uploaded_image(disease_file)
                
                if analyze_disease_button and disease_image:
                    st.markdown('<div class="spinner">Analyzing plant disease... Please wait</div>', unsafe_allow_html=True)
                    with st.spinner(""):
                        result = analyze_disease(disease_image, plant_type, additional_info_disease)
                    
                    st.markdown(f"<div class='results-section'>{result}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Weed Identification Tab
    with tabs[3]:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown(f"""
        <h2 style='color: {COLORS["primary"]}; display: flex; align-items: center;'>
            <span style='margin-right: 10px;'>🌿</span> Weed Identification
        </h2>
        <div class='divider'></div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown("""
            <div class="container-card">
                <h3 style='color: #1B5E20; margin-top: 0;'>Upload an image of a weed</h3>
                <p>Upload a clear image of the weed plant for identification.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                weed_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="weed_upload")
                location = st.text_input("Growing location (optional)", placeholder="e.g., North America, Tropical, Mediterranean", key="location")
            
            with col2:
                additional_info_weed = st.text_area("Additional information (optional)", 
                                                  placeholder="e.g., Growing conditions, surrounding plants",
                                                  key="weed_info",
                                                  height=100)
            
            analyze_weed_button = st.button("🔍 Analyze Weed", key="weed_button", use_container_width=True)
            
            if weed_file:
                weed_image = process_uploaded_image(weed_file)
                
                if analyze_weed_button and weed_image:
                    st.markdown('<div class="spinner">Analyzing weed image... Please wait</div>', unsafe_allow_html=True)
                    with st.spinner(""):
                        result = analyze_weed(weed_image, location, additional_info_weed)
                    
                    st.markdown(f"<div class='results-section'>{result}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; padding: 20px; color: #616161; font-size: 0.8em;">
        <p>FieldFix - Your Smart Agricultural Companion</p>
        <p>Powered by gemini-3-flash-preview API</p>
    </div>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()
