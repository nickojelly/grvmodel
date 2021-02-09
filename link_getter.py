import random
import csv
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from tkinter import Tk

#Working again as of feburary 2
#This program opens a list of meeting numbers "meets.csv" and downloads the
# xml data file for them

driver_path = 'C:/Users/Nick/OneDrive - The University of Melbourne/personal projects/instagram hottest 100/chromedriver_win32/chromedriver.exe'


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
    with open('meets.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(meets_list)
    return meets_list

def downloader(meets_list):
    print("\n\n\n --------- \n\n\n")
    partial_url = 'https://fasttrack.grv.org.au/Meeting/Details/'
    chrome_options = webdriver.ChromeOptions()

    prefs = {'download.default_directory' : 'C:\\Users\\Nick\\Desktop\\greyhound results',
             'download.prompt_for_download': False,
             'safebrowsing.enabled' : True,
             "download.directory_upgrade": True}
    chrome_options.add_experimental_option('prefs', prefs)

    driver = webdriver.Chrome(driver_path, chrome_options=chrome_options)

    for i in meets_list:
        print("\n"+partial_url+i+"\n")
        driver.get(partial_url+str(i))

        try:
            download_button = driver.find_element_by_xpath('//button[text()="Download Race Results Format  (xml)"]')
        except:
            print("no dl button found")


        #print(download_button)
        download_button.click()
        time.sleep(1)


if __name__== "__main__":
    search_url = 'https://fasttrack.grv.org.au/Meeting/Search?MeetingDateFrom=05%2F07%2F2016&MeetingDateTo=05%2F05%2F2020&Status=Results+Finalised&TimeSlot=&DayOfWeek=&DisplayAdvertisedEvents=false&AllTracks=False&SelectedTracks=Shepparton&searchbutton=Search&page='
    download_url = 'https://fasttrack.blob.core.windows.net/fasttrackpublic/raceresults/1099030018/RaceResults.pdf'

    #meets_list = race_numbers(search_url)

    with open('meets.csv', newline='') as f:
        reader = csv.reader(f)
        meets_list = list(reader)[0]

    #
    print(meets_list)

    for i in range(len(meets_list)):
        meets_list[i] = meets_list[i][-9:]

    print(meets_list)

    downloader(meets_list)
