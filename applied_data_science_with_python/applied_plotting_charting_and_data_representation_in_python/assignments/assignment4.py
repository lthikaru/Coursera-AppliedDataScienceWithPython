
# https://en.wikipedia.org/wiki/List_of_National_Hockey_League_attendance_figures
# https://en.wikipedia.org/wiki/List_of_Detroit_Red_Wings_seasons

# 1) how have the average attendances for Detroit Red Wings develop compared to the rest of NHL over the period
# 2010-2016


from bs4 import BeautifulSoup
import urllib2

wiki = "https://en.wikipedia.org/wiki/List_of_National_Hockey_League_attendance_figures"
header = {'User-Agent': 'Mozilla/5.0'}  # Needed to prevent 403 error on Wikipedia
req = urllib2.Request(wiki, headers=header)
req = urllib2.Request(wiki)
page = urllib2.urlopen(req)
soup = BeautifulSoup(page)

area = ""
district = ""
town = ""
county = ""

table = soup.find("table", {"class": "wikitable sortable"})


urllib2.Request