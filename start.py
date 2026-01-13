import streamlit as st
from PIL import Image
img = Image.open("bent.png")
st.image(img, width=150)
st.set_page_config(page_title = "This is a Multipage WebApp")
st.title("Chart_mobile")
st.write("Приложение для разработки ОГ")
st.sidebar.success("Выбрать категорию")

