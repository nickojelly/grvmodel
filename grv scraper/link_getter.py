import random
import csv
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from tkinter import Tk
from selenium.webdriver.chrome.options import Options
#Working again as of feburary 2
#This program opens a list of meeting numbers "meets.csv" and downloads the
# xml data file for them

#planning to iterate over serach by month to get all links
#now working getting meet numbers searching by day with function searchit using predifined strings from day-day_strings

driver_path = 'C:/Users/Nick/OneDrive - The University of Melbourne/personal projects/instagram hottest 100/chromedriver_win32/chromedriver.exe'

# do not use anymore
#depreciated
def race_numbers(url):
    driver = webdriver.Chrome(driver_path)
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
        time.sleep(2)
        #break
    with open('meets.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(meets_list)
    return meets_list

def searchit(day_strings):
    driver = webdriver.Chrome(driver_path)
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
        time.sleep(0.2)
    with open('meetsnewpart2.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(meets_list)


def downloader(meets_list):
    print("\n\n\n --------- \n\n\n")
    #partial_url = 'https://fasttrack.grv.org.au/Meeting/Details/'
    chrome_options = Options()

    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-gpu")
    #chrome_options.add_argument("--no-sandbox") # linux only
    #chrome_options.add_argument("--headless")

    prefs = {'download.default_directory' : 'I:\\greyhound model\\grvmodel\\grv scraper\\new data',
             'download.prompt_for_download': False,
             'safebrowsing.enabled' : True,
             "download.directory_upgrade": True}
    chrome_options.add_experimental_option('prefs', prefs)

    driver = webdriver.Chrome(driver_path, options =chrome_options)
    driver.implicitly_wait(10)
    for i in meets_list:
        print("\n"+i+"\n")
        driver.get(i)

        try:
            download_button = driver.find_element_by_xpath('//button[text()="Download Race Results Format  (xml)"]')
        except:
            print("no dl button found")

        download_button.click()
        time.sleep(0.2)


if __name__== "__main__":

    #mode 0 to generate race race_numbers
    #mode 1 to download race meetings

    mode = 0

    if mode:
        #this part obtains the meeting race_numbers
        with open('daystrings.txt') as f:
            day_strings = f.read().splitlines()

        searchit(day_strings[1000:])
    else:
        #this part downloads the xml documents

        with open('meetsnewpart2.csv', newline='') as f:
            reader = csv.reader(f)
            meets_list = list(reader)[0]

        downloader(meets_list)
