"""This script processes the WorldCereal dataset to create global labels."""


import requests
import random
import pandas as pd

apiUrl = 'https://ewoc-rdm-api.iiasa.ac.at'


collectionResponse = requests.get(f'{apiUrl}/collections?SkipCount=0&MaxResultCount=10')
res = collectionResponse.json()

total = res['totalCount']
items = res['items']

print(f'Total Public Collections: {total}')
print(f'Current Response Collection Count: {len(items)}')

accumulatedCount = len(items)

while(accumulatedCount < total):
 reqUrl = f'{apiUrl}/collections?SkipCount={accumulatedCount}&MaxResultCount=10'
 collectionResponse = requests.get(reqUrl)
 res2 = collectionResponse.json()
 print(res2)
 total = res2['totalCount']
 items += res2['items']
 accumulatedCount = accumulatedCount + len(res2['items'])
 print(f'Accumulated Collections Count: {accumulatedCount}')

print(items[0])

# {'collectionId': '2017_af_oneacrefundmel_point_110', 'title': 'MEL agronomic survey eastern Africa, 2017', 'featureCount': 3374, 'type': 'Point', 'accessType': 'Public', 'typeOfObservationMethod': 'Unknown', 'confidenceLandCover': 0, 'confidenceCropType': 0, 'confidenceIrrigationType': 0, 'ewocCodes': [1000000000, 1101060000], 'irrTypes': [0], 'extent': {'spatial': {'bbox': [[34.05331037027546, -1.1181928808662311, 35.39540334981372, 1.1412347820301905]], 'crs': 'http://www.opengis.net/def/crs/OGC/1.3/CRS84'}, 'temporal': {'interval': [['2017-08-01T00:00:00', '2017-08-01T00:00:00']], 'trs': 'http://www.opengis.net/def/uom/ISO-8601/0/Gregorian'}}, 'additionalData': '', 'crs': ['http://www.opengis.net/def/crs/EPSG/0/4326'], 'lastModificationTime': None, 'lastModifierId': None, 'creationTime': '2025-01-24T06:53:26.991638', 'creatorId': None, 'id': '3a17a943-1e2a-f4ea-3224-c1d4f00796f1'}

# # go through items to find only type == 'Point'
# point_items = [item for item in items if item['type'] == 'Point']

# print(f'Total point items: {len(point_items)}')

# Put all items into a pandas dataframe
df = pd.DataFrame(items)
df.to_csv('worldcereal_collections.csv', index=False)

# Filter rows with confidenceLandCover >= 90 and confidenceCropType >= 90
df_filtered = df[df['confidenceLandCover'] >= 90]
df_filtered = df_filtered[df_filtered['confidenceCropType'] >= 90]

print(f'Total filtered rows: {len(df_filtered)}')  # 89 out of 140 collections

for index, row in df_filtered.iterrows():
    collectionId = row['collectionId']
    print(f'Processing collectionId: {collectionId}')
    # itemUrl = f'{apiUrl}/collections/{collectionId}/items?SkipCount=0&MaxResultCount=50'
    # itemsResponse = requests.get(itemUrl)
    # res = itemsResponse.json()

    # print(f'CollectionId={collectionId}')
    # print(f'Total Number of Features in Collection:{res["NumberMatched"]}')
    # print(f'Number Returned:{res["NumberReturned"]}')
    # print(f'SkipCount:{res["SkipCount"]}')

    # total = res['NumberMatched']
    # accumulatedItemsCount = res['NumberReturned']
    # features = res['features']

    # # use SkipCount and MaxResultCount to paginate and get all the features in the collection
    # while(accumulatedItemsCount < total):
    #     reqUrl =  f'{apiUrl}/collections/{collectionId}/items?SkipCount={accumulatedItemsCount}&MaxResultCount=50'
    #     itemsSkipResponse = requests.get(reqUrl)
    #     res2 = itemsSkipResponse.json()
    #     features += res2['features']
    #     accumulatedItemsCount = accumulatedItemsCount + len(res2['features'])
    #     print(f'accumulatedItemsCount: {accumulatedItemsCount}')
    #     # if(accumulatedItemsCount > 100):
    #     #     break
    
    # print(f'Total Number of Features in Collection:{len(features)}')

    metadataUrl = f'{apiUrl}/collections/{collectionId}/metadata/items'
    metadata = requests.get(metadataUrl)
    metadata = metadata.json()
    # print value fields
    # search for item in the metadata with id == 36
    # for item in metadata:
    #     if item['id'] == 36:
    #         print(item['value'])
    #         break
    # go through all items in the metadata, check if item has a value field, with sth ends in https://ewocstorage.blob.core.windows.net/public/2019afnhicropharvestpoly100/2019_AF_NHI-CROP-HARVEST_POLY_100.geoparquet
    for item in metadata:
        if item['value'] and item['value'].endswith('.geoparquet'):
            print(item['value'])
            # Download the file to the current directory
            fileUrl = item['value']
            fileName = fileUrl.split('/')[-1]
            response = requests.get(fileUrl)
            with open(fileName, 'wb') as file:
                file.write(response.content)









