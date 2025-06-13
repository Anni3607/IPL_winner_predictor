import streamlit as st
import base64

st.set_page_config(layout="centered", page_title="Test App", page_icon="🧪")

# Test base64 image
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Image not found at {image_path}")
        return None

st.markdown("""
    <style>
    .stApp {
        background-color: #0000FF !important; /* Blue background */
        color: yellow !important; /* Yellow text */
    }
    .test-text {
        color: red !important;
        font-size: 30px !important;
        background-color: black !important;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Deployment Test App")
st.markdown("<p class='test-text'>This text should be red on a black background!</p>", unsafe_allow_html=True)
st.write("The main app background should be blue with yellow default text.")

# Try to load a known logo (e.g., 'csk.png' if you have it)
logo_path = "logos/csk.png" # Adjust to a logo you definitely have
encoded_logo = get_base64_image(logo_path)
if encoded_logo:
    st.markdown(f"<img src='data:image/png;base64,{encoded_logo}' width='100'>", unsafe_allow_html=True)
else:
    st.error("Logo display failed in test app.")

st.success("Test app loaded!")
