from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re
import time
import datetime


options = ChromeOptions()
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
options.add_argument('user_agent=' + user_agent)
options.add_argument("lang=ko_KR")
# options.add_argument('headless')
options.add_argument('window-size=1920x1080')

service = ChromeService(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

start_url = 'https://m.kinolights.com/discover/explore'
button_movie_tv_xpath = '//*[@id="contents"]/section/div[3]/div/div/div[3]/button'
button_movie_xpath = '//*[@id="contents"]/section/div[4]/div[2]/div[1]/div[3]/div[2]/div[2]/div/button[1]'
button_ok_xpath = '//*[@id="applyFilterButton"]'
button_post_xpath = '//*[@id="contents"]/div/div/div[3]/div[2]/div[1]/a'

driver.get(start_url)
time.sleep(1)

button_movie_tv = driver.find_element(By.XPATH, button_movie_tv_xpath)
driver.execute_script('arguments[0].click()', button_movie_tv)
time.sleep(1)

button_movie = driver.find_element(By.XPATH, button_movie_xpath)
driver.execute_script('arguments[0].click()', button_movie)
time.sleep(1)

button_ok = driver.find_element(By.XPATH, button_ok_xpath)
driver.execute_script('arguments[0].click()', button_ok)
time.sleep(1)


for i in range(25):
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(1)

list_review_url = []
movie_titles = []
for i in range(51,101):
    base = driver.find_element(By.XPATH, f'//*[@id="contents"]/div/div/div[3]/div[2]/div[{i}]/a').get_attribute("href")
    list_review_url.append(f"{base}/reviews")
    title = driver.find_element(By.XPATH, f'//*[@id="contents"]/div/div/div[3]/div[2]/div[{i}]/div/div[1]').text
    movie_titles.append(title)

print(list_review_url[:5])
print(len(list_review_url))
print(movie_titles[:5])
print(len(movie_titles))


reviews = []
for url in list_review_url:
    driver.get(url)
    time.sleep(0.5)
    review = ''
    for i in range(1,31):
        review_title_xpath = '//*[@id="contents"]/div[2]/div[2]/div[{}]/div/div[3]/a[1]/div'.format(i)
        review_more_xpath = '//*[@id="contents"]/div[2]/div[2]/div[{}]/div/div[3]/div/button'.format(i)
        try:
            review_more = driver.find_element(By.XPATH, review_more_xpath)
            driver.execute_script('arguments[0].click()', review_more)
            time.sleep(1)
            review_xpath = '//*[@id="contents"]/div[2]/div[1]/div/section[2]/div/div'
            review = review + ' ' + driver.find_element(By.XPATH, review_xpath).text
            print(review)
            driver.back()
            time.sleep(1)

        except NoSuchElementException as e:
            print('더보기',e)
            try:
                review = review + ' ' + driver.find_element(By.XPATH, review_title_xpath).text
            except:
                print('review title error')

        except StaleElementReferenceException as e:
            print('stale', e)
            time.sleep(1)
        except:
            print('error')
    print(review)
    reviews.append(review)
    time.sleep(1)

df = pd.DataFrame({'titles': movie_titles, 'reviews': reviews})
today = datetime.datetime.now().strftime('%Y%m%d')
df.to_csv('./crawling_data/reviews_100.csv', index=False)

