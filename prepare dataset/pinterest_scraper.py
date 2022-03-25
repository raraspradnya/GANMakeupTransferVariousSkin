from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests
import time
import os
import threading
from selenium.webdriver.support import ui

from ScrapingEssentials import ScrapingEssentials
t = ScrapingEssentials()


def page_is_loaded(driver):
    return driver.find_element_by_tag_name("body") != None

def download_pages(driver, valid_urls):
    list_counter = 0

    # Pinterest happens to change its HTML every once in a while to prevent botting.

    # This should account for all the differences
    # soup = BeautifulSoup(driver.page_source, "lxml")
    # for pinWrapper in soup.find_all("div", {"class": "pinWrapper"}):
    #     class_name = pinWrapper.get("class")
    #     print(class_name)
    #     if "_o" in class_name[0]:
    #         print(class_name)
    #         break
    #
    # #Finds the tags of the HTML and adjusts it
    # name = " ".join(class_name)
    # print(name)

    # Does this until you have 10000 items or the program has gone on for long enough, meaning that it reached the end of results
    beginning = time.time()
    end = time.time()

    while list_counter < 160:
        beginning = time.time()
        # ----------------------------------EDIT THE CODE BELOW------------------------------#
        # Locate all the urls of the detailed pins
        soup = BeautifulSoup(driver.page_source, "html.parser")
        # for c in soup.find_all("div", {"class": name}):
        # print("11")
        for pinLink in soup.find_all("div", {"data-test-id": "pinWrapper"}):
            # print("22")
            # for images in pinLink.find_all("div", {"data-test-id": "pinrep-image"}):
            # print("33")
            a = pinLink.find_all("a")
            # print("44")
            # print(a)
            url = ("https://pinterest.com" + str(a[0].get("href")))
            # Checks and makes sure that the pin isn't there already and that random urls are not invited
            if len(url) < 60 and url not in valid_urls and "A" not in url:
                # ---------------------------------EDIT THE CODE ABOVE-------------------------------#
                valid_urls.append(url)
                print("THREAD 1: " + str(list_counter))
                list_counter += 1
                end = time.time()
            time.sleep(.15)
            # Scroll down now
        driver.execute_script("window.scrollBy(0,300)")
    return

def login(driver):
    wait = ui.WebDriverWait(driver, 10)
    wait.until(page_is_loaded)
    email = driver.find_element_by_xpath("//input[@type='email']")
    password = driver.find_element_by_xpath("//input[@type='password']")
    email.send_keys("rarasprdny@gmail.com")
    password.send_keys("Nayahita02")
    password.submit()
    time.sleep(3)
    print("Teleport Successful!")

def get_pic(valid_urls, driver):
    get_pic_counter = 0
    time.sleep(5)
    print("get_pic_counter", get_pic_counter)
    print("valid_urls", len(valid_urls))
    while ((get_pic_counter < len(valid_urls)) or (get_pic_counter <160)):
        # print(0)
        # Now, we can just type in the URL and pinterest will not block us
        try:
            url =valid_urls[get_pic_counter]
            driver.get(url)
        except:
            time.sleep(5)

        # Wait until the page is loaded
        if driver.current_url == url:
            wait = ui.WebDriverWait(driver, 10)
            wait.until(page_is_loaded)
            loaded = True
        # print(1)
        print(url)
        # -----------------------------------EDIT THE CODE BELOW IF PINTEREST CHANGES---------------------------#
        # Extract the image url
        soup = BeautifulSoup(driver.page_source, "html.parser")
        # print(2)
        for mainContainer in soup.find_all("div", {"class": "mainContainer"}):
            # print(3)
            check = mainContainer.find_all("div", {"data-test-id": "closeup-image"})
            # print("check", check)
            if (len(check)==0):
                print("im here")
                get_pic_counter += 1
                break
            else:
                for closeupContainer in mainContainer.find_all("div", {"data-test-id": "closeup-image"}):
                    # print(4)
                    # for heightContainer in closeupContainer.find_all("div", {"class": "FlashlightEnabledImage Module"}):
                    # print(5)
                    for img in closeupContainer.find_all("img"):
                        print(6)
                        img_link = img.get("src")
                        if "564" in img_link:
                            print("THREAD 2: " + str(get_pic_counter))
                            # print("img_link = ", img_link)
                            get_pic_counter += 1
                            t.download_image(img_link)
                            break

            # ---------------------------------EDIT THE CODE ABOVE IF PINTEREST CHANGES-----------------------------#

try:
    os.chdir(os.path.join(os.getcwd(), 'images/coba3'))
except:
    pass

home= f'https://www.pinterest.com/login/?referrer=home_page'
url = f'https://id.pinterest.com/kaynuli/makeup-looks/'

driver1 = webdriver.Chrome(ChromeDriverManager().install())
driver2 = webdriver.Chrome(ChromeDriverManager().install())
driver1.get(home)
driver2.get(home)
login(driver1)
login(driver2)

loaded = False
while loaded == False:
    if driver1.current_url != "https://www.pinterest.com/login/?referrer=home_page":
        loaded = True

driver1.get(url)

time.sleep(3)
valid_urls = []
t1 = threading.Thread(target=download_pages, args=(driver1, valid_urls,))
t1.setDaemon(True)
t1.start()

time.sleep(5)

t2 = threading.Thread(target=get_pic, args=(valid_urls, driver2,))
t2.setDaemon(True)
t2.start()

t1.join()
t2.join()
print(t1.is_alive())
print(t2.is_alive())
t.reset()
print("done")