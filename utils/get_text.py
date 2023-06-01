# import libraries
import requests
from bs4 import BeautifulSoup
import streamlit as st


# get text from the webpage and dump it into a text
@st.cache_data
def get_text(URL):
    page = requests.get(URL)
    soup = BeautifulSoup(page.content,"html.parser")
    poem = soup.get_text()
    return poem
