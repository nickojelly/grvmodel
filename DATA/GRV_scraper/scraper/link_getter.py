import random
import csv
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from tkinter import Tk

#Currently works getting hyperlinks of pictures
#Need to create something to login to stop getting blocked

def main():
    driver = webdriver.Chrome('C:/Users/Nick/OneDrive - The University of Melbourne/personal projects/instagram hottest 100/chromedriver_win32/chromedriver.exe')
    driver.get('https://fasttrack.grv.org.au/Meeting/Search?MeetingDateFrom=01%2F01%2F2015&MeetingDateTo=01%2F04%2F2020&Status=Results%20Finalised&DisplayAdvertisedEvents=false&AllTracks=False&SelectedTracks=Shepparton&searchbutton=Search&page=1')
    scroll_link

def scoll_link():
    elems = driver.find_elements_by_xpath("//a[@href]")
    for elem in elems:
        print(elem.get_attribute("href"))

if __name__== "__main__":
  driver()
