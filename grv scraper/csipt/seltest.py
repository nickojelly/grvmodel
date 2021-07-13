import time

from selenium import webdriver
import os

cur = os.getcwd()
print(cur)
driver = webdriver.Firefox(cur) 

