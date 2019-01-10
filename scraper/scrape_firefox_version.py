#!/usr/bin/python3
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from time import sleep
import sys
import json
import datetime

# Twitter web scraper (firefox version)
# Based on Brendan Brown's scraper : https://github.com/bpb27/twitter_scraping
# Kévin Dumanoir and Lea Uhliarova, Stockholm University, BIGDATA HT2018
# MIT License

# check user first argument
if len(sys.argv) < 2 or not sys.argv[1]:
    print('No twitter username provided as argument.',
          '\n',
          'Usage : scrape.py USERNAME [START YEAR] [END YEAR]')
    exit(120)

if len(sys.argv) < 3 or not sys.argv[2]:
    start_year = 2009
else:
    start_year = int(sys.argv[2])

if len(sys.argv) < 4 or not sys.argv[3]:
    end_year = 2019
else:
    end_year = int(sys.argv[3])

# edit these three variables
user = sys.argv[1]
start = datetime.datetime(start_year, 1, 1)  # year, month, day 2010 , 1 , 1
end = datetime.datetime(end_year, 1, 1)  # year, month, day

print('Start scraping tweets since ', start_year, ' for username: ', user)

# only edit these if you're having problems
delay = 1  # time to wait on each page load before reading the page
options = Options()
options.headless = len(sys.argv) < 5 or not sys.argv[4]
driver = webdriver.Firefox(options=options)  # options are Chrome() Firefox() Safari()

# don't mess with this stuff
twitter_texts_filename = 'all_tweets_{}_{}_{}.json'.format(user, start_year, end_year)
days = (end - start).days + 1
id_selector = '.time a.tweet-timestamp'
tweet_selector = 'div.js-stream-tweet'
user = user.lower()
tweets = []

print('Tweets will be stored in file: ', twitter_texts_filename)


def format_day(date):
    day = '0' + str(date.day) if len(str(date.day)) == 1 else str(date.day)
    month = '0' + str(date.month) if len(str(date.month)) == 1 else str(date.month)
    year = str(date.year)
    return '-'.join([year, month, day])


def form_url(since, until):
    p1 = 'https://twitter.com/search?f=tweets&vertical=default&q=from%3A'
    p2 = user + '%20since%3A' + since + '%20until%3A' + until + '%20include%3Aretweets&src=typd'
    return p1 + p2


def increment_day(date, i):
    return date + datetime.timedelta(days=i)


for day in range(days):
    d1 = format_day(increment_day(start, 0))
    d2 = format_day(increment_day(start, 1))
    url = form_url(d1, d2)
    print(url)
    print(d1)
    driver.get(url)
    sleep(delay)

    try:
        found_tweets = driver.find_elements_by_css_selector(tweet_selector)
        increment = 10

        while len(found_tweets) >= increment:
            print('scrolling down to load more tweets')
            driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            sleep(delay)
            found_tweets = driver.find_elements_by_css_selector(tweet_selector)
            increment += 10

        print('{} tweets found, {} total'.format(len(found_tweets), len(tweets)))

        for tweet in found_tweets:
            try:
                data = {'id': tweet.get_attribute('data-tweet-id'),
                        'screen_name': tweet.get_attribute('data-screen-name'),
                        'timestamp': tweet.find_element_by_css_selector('._timestamp').get_attribute('data-time'),
                        'text': tweet.find_element_by_css_selector('.tweet-text').text}
                tweets.append(data)
            except StaleElementReferenceException as e:
                print('lost element reference', tweet)

    except NoSuchElementException:
        print('no tweets on this day')

    start = increment_day(start, 1)

with open(twitter_texts_filename, 'w') as outfile:
    json.dump(tweets, outfile)

print('all done here')
driver.close()
