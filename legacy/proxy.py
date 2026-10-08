#import requests
#import lxml.html as lh
#import pandas as pd

#http://free-proxy.cz/en/proxylist/country/US/http/uptime/all
#http://www.freeproxylists.net/?c=US&pt=&pr=HTTP&a%5B%5D=0&a%5B%5D=1&a%5B%5D=2&u=0
url='https://hidemy.name/en/proxy-list/?country=US&type=hs#list'#Create a handle, page, to handle the contents of the website

# Import necessary packages
from bs4 import BeautifulSoup
import requests
import pandas as pd
# Site URL

# Make a GET request to fetch the raw HTML content
html_content = requests.get(url).text

# Parse HTML code for the entire site
soup = BeautifulSoup(html_content, "lxml")
#print(soup.prettify()) # print the parsed data of html

# On site there are 3 tables with the class "wikitable"
# The following line will generate a list of HTML content for each table
print(soup)

#table_data = soup.find('table', class_ = 'table table-striped table-bordered table-hover table-condensed table-list')
