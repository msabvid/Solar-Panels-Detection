#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:08:41 2017

@author: msabate
"""

import os
import urllib.request as urllib2


url = 'https://scihub.copernicus.eu/dhus/odata/v1'
username='msabate'
password = 'camprodon1'

p = urllib2.HTTPPasswordMgrWithDefaultRealm()
p.add_password(None, url, username, password)
handler=urllib2.HTTPBasicAuthHandler(p)
opener = urllib2.build_opener(handler)
urllib2.install_opener(opener)

id = '5794cecc-d204-452c-93c1-632fe6824f39'

sentinel_link = "https://scihub.copernicus.eu/dhus/odata/v1/Products('b5416be0-e843-4f30-be11-df177d4a2e7a')/$value"
sentinel_link = "https://scihub.copernicus.eu/dhus/odata/v1/Products('"+id+"')/$value"
print(sentinel_link)
product_id = sentinel_link.split('/')[-2].replace("('", '_').replace("')", "")
print(product_id)

rest_api = urllib2.urlopen(sentinel_link)
data = rest_api.read()

destination_path = '/Users/msabate/Projects/CityFinancial/data/' + product_id + '.zip'

if os.path.exists(destination_path):
    print('The file was already downloaded!')
else:
    downloadfile = urllib2.urlopen(sentinel_link)
    data = downloadfile.read()
    
    with open(destinationpath, "wb") as code:
        code.write(data)
        

        



