import os
import requests

headers = { 
    "apikey": os.environ["ZENSCRAPE_API_KEY"]  # key removed from source; it was committed earlier and should be revoked
}

params = (
   ("url","https://www.ebay.com/sch/i.html?_from=R40&_nkw=2014+panini+giannis+antetokounmpo+rookie+194&_sacat=0&rt=nc&LH_Sold=1&LH_Complete=1"),
   ("render","true"),
   ("premium","true"),
);

response = requests.get('https://app.zenscrape.com/api/v1/get', headers=headers, params=params);
print(response.text)