import random
import csv
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from tkinter import Tk
import urllib.request
from bs4 import BeautifulSoup


def listgen(start, end):
    race_numbers = range(start, end)
    return race_numbers

def urltester(possible_race_numbers):
    confirmed_race_numbers = []
    for i in range(len(possible_race_numbers)):
        req =  urllib.request.Request("https://fasttrack.grv.org.au/Meeting/Details/"+str(possible_race_numbers[i]))
        try:
            urllib.request.urlopen(req)
            print(possible_race_numbers[i], "passed")
            confirmed_race_numbers.append(possible_race_numbers[i])
        except urllib.error.URLError as e:
            print(possible_race_numbers[i], e.reason)
            possible_race_numbers = possible_race_numbers[i+100:]

if __name__ == "__main__":
    possible_race_numbers = listgen(228692176, 425059883)
    print(len(possible_race_numbers))
    urltester(possible_race_numbers)
    file1 = open("links.txt", "w")
    for i in possible_race_numbers:
        file1.write("https://fasttrack.grv.org.au/Meeting/Details/"+str(i))
        if i%100000 == 0:
            print(i)


    #print("title = ", soup.title.string)
