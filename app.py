import streamlit as st
from about import about_page
from drowsiness_detection import drowsiness_detection_page

# Streamlit UI - Page Navigation
def main():
    st.set_page_config(page_title="Drowsiness Detection System", layout="wide")

    # Sidebar Navigation
    page = st.sidebar.selectbox("Select a Page", ["Drowsiness Detection", "About"])

    if page == "Drowsiness Detection":
        drowsiness_detection_page()
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()
