import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Streamlit App", page_icon=":shark:", layout="wide")

# ---- LOAD ASSETS ----
img_contact_form = Image.open("images/download.jpg")


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")
# -- Header ---
st.subheader("Hi, I am Nandeesh :wave:")
st.title("A Python Developer and ML Engineer")
st.write("I am a Python Developer and Machine Learning Engineer with 3 years of experience in building scalable and robust applications. I am looking for opportunities to work on challenging projects and grow my skills.")

# --- what i do ---
with st.container():
    st.write("--------")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("What I do")
        st.write("##")
        st.write(
            """
            On my YouTube channel I am creating tutorials for people who:
            - are looking for a way to leverage the power of Python in their day-to-day work.
            - are struggling with repetitive tasks in Excel and are looking for a way to use Python and VBA.
            - want to learn Data Analysis & Data Science to perform meaningful and impactful analyses.
            - are working with Excel and found themselves thinking - "there has to be a better way."

            If this sounds interesting to you, consider subscribing and turning on the notifications, so you don’t miss any content.
            """
        )
        st.write("[Youtube Channel](https://www.youtube.com/@krishnaik06)")


# ---- PROJECTS ----
with st.container():
    st.write("---")
    st.header("My Projects")
    st.write("##")
    image_column, text_column = st.columns((1, 2))
    with image_column:
        st.image(img_contact_form)
    with text_column:
        st.subheader("End To End Data Science Projects 2023")
        st.write(
            """
            End To End Data Science Projects 2023:
            - End To End Deep Learning Project Using MLOPS DVC Pipeline With Deployments Azure And AWS- Krish Naik.
            - End To End Machine Learning Project Implementation Using AWS Sagemaker.
            - End To End NLP Project Implementation With Deployment Github Action- Text Summarization- Krish Naik.
            - End To End Cell Segmentation Using Yolo V8 With Deployment- Part 1
            - End To End Cell Segmentation Using Yolo V8 With Deployment- Part 2
            """
        )
        st.markdown("[Watch Video...](https://www.youtube.com/watch?v=p1bfK8ZJgkE&list=PLZoTAELRMXVOjQdyqlCmOtq1nZnSsWvag)")
    

# ---- CONTACT ----
with st.container():
    st.write("---")
    st.header("Get In Touch With Me!")
    st.write("##")

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    contact_form = """
    <form action="https://formsubmit.co/YOUR@MAIL.COM" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()