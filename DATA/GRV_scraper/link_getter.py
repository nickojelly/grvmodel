import random
import csv
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from tkinter import Tk
import os
from selenium.webdriver.chrome.options import Options
#Working again as of feburary 2
#This program opens a list of meeting numbers "meets.csv" and downloads the
# xml data file for them

#planning to iterate over serach by month to get all links
#now working getting meet numbers searching by day with function searchit using predifined strings from day-day_strings

driver_path = 'C:/Users/Nick/Documents/GitHub/grvmodel/grv scraper/chromedriver_win32/chromedriver.exe'

# do not use anymore
#depreciated
def race_numbers(url):
    driver = webdriver.Chrome(executable_path="C:\\chromedriver.exe")

    
    meets_list = []
    for i in range(10):
        driver.get(url+str(i))
        print(url+str(i))
        elems = driver.find_elements_by_xpath("//a[@href]")
        for elem in elems:
            #print(elem.get_attribute("href"), 1)
            if "Meeting/Details" in elem.get_attribute("href"):
                #print("this worked")

                if elem.get_attribute("href") not in meets_list:
                    meets_list.append(elem.get_attribute("href"))
                    print(elem.get_attribute("href"))
        #break
    with open('meets.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(meets_list)
    return meets_list

def searchit(day_strings):

    cur = os.getcwd()
    print(cur)

    driver = webdriver.Firefox(r'C:\Users\Nick\Documents\GitHub\grvmodel\grv scraper\csipt')
    meets_list = []
    print(day_strings)
    for i in day_strings:
        print(i)
        #creates url for specific day and fetches it
        url = "https://fasttrack.grv.org.au/Meeting/Search?MeetingDateFrom="+i+"&MeetingDateTo="+i+"&Status=Results+Finalised&TimeSlot=&DayOfWeek=&DisplayAdvertisedEvents=false&AllTracks=True&SelectedTracks=AllTracks&searchbutton=Search"
        driver.get(url)

        #searchs for meeting in that day and adds them to list
        elems = driver.find_elements_by_xpath("//a[@href]")
        for elem in elems:
            #print(elem.get_attribute("href"), 1)
            if "Meeting/Details" in elem.get_attribute("href"):
                #print("this worked")

                if elem.get_attribute("href") not in meets_list:
                    meets_list.append(elem.get_attribute("href"))
                    print(elem.get_attribute("href"))
        time.sleep(0.02)
    with open('meetsnewpart3.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(meets_list)


def downloader(meets_list):
    #mode 0 to download Results
    #mode 1 to download Form
    mode = 0

    if mode:
        xpath = '//button[text()="Download Full Format  (xml)"]'
        download_dir = 'I:\\greyhound model\\grvmodel\\grv scraper\\full race form'
    else:
        xpath = '//button[text()="Download Race Results Format  (xml)"]'
        download_dir = 'I:\\greyhound model\\grvmodel\\grv scraper\\full race results'
    print("\n\n\n --------- \n\n\n")
    #partial_url = 'https://fasttrack.grv.org.au/Meeting/Details/'

    print(xpath)
    profile = webdriver.FirefoxProfile()
    profile.set_preference("browser.download.folderList", 2)
    profile.set_preference("browser.download.manager.showWhenStarting", False)
    profile.set_preference("browser.download.dir", download_dir)
    profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/xml")
    profile.set_preference("browser.download.viewableInternally.enabledTypes", "")

    driver = webdriver.Firefox(firefox_profile=profile)

    for i in meets_list:
        print("\n"+i+"\n")
        driver.get(i)

        try:
            download_button = driver.find_element_by_xpath(xpath)
            download_button.click()
        except:
            print("no dl button found")




if __name__== "__main__":
    print("oiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
    #mode 0 to generate race race_numbers
    #mode 1 to download race meetings

    mode = 0

    if mode:
        #this part obtains the meeting race_numbers
        print("running in search mode")
        with open('daystrings.txt') as f:
            day_strings = f.read().splitlines()

        searchit(day_strings)
    else:
        #this part downloads the xml documents

        with open('meetsnewpart3.csv', newline='') as f:
            reader = csv.reader(f)
            meets_list = list(reader)[0]

        downloader(meets_list)