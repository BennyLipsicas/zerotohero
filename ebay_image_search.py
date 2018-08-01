import base64
import requests
import requests_cache
import multiprocessing as mp
from multiprocessing import Manager
import traceback
import itertools
from utils.WebService import WebService
from werkzeug.contrib.fixers import ProxyFix
import logging
from logging.handlers import RotatingFileHandler
import os
import datetime
import sys
import cStringIO
from StringIO import StringIO
from PIL import Image
from flask_cors import CORS
import time
import threading
import boto
import uuid
import copy

conn = boto.connect_s3('AKIAIEOBSULUK4NEP7XA', '4RDa0vA7LLGVx8m7XgS7r7Qr+iiy0pFLKNYxIner')
b = conn.get_bucket('caffemodels')
path = 'ebay-image-search'

def upload_b64_to_s3(b64_img, img_id):
    k = b.new_key(path + '/' + img_id + '.jpg')
    k.set_contents_from_string(base64.b64decode(b64_img), headers={"Content-Type": "image/jpeg"})
    k.set_acl('public-read')
# requests_cache.install_cache()

if not os.path.isdir("log"):
    os.makedirs("log")

logger = logging.getLogger("Rotating Log")
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler('log/ebay_image_search.log', maxBytes=2000000, backupCount=10)
formatter = logging.Formatter(
    "%(asctime)s [%(processName)s:%(process)d] [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

app = WebService(__name__)
CORS(app)

ENDPOINT = 'https://api.ebay.com/image_search/v1/search'

TOKEN_TTL = 0

QUALITY = 93
size_to_net = 224

BODY_TEMPLATE = {
    "imageRequest": {
        "userContext": {
            "userLocation": {
                "country": "US",
                "postalCode": "95125"
            }
        },
        "sortOrder": "BestMatch",
        "shipToLocation": {
            "country": "US",
            "postalCode": "95125"
        },
        "paginationInput": {
            "entriesPerPage": 20,
            "pageNumber": 1
        },
        "outputSelector": {
            "omitFalse": True,
            "omitUrls": True,
            "item": {
                "seller": {},
                "condition": {},
                "shipping": {},
                "distance": {}
            },
            "rewrite": {},
            "refinement": {
                "priceFilterHistogram": {},
                "conditionHistogram": {},
                "categoryHistogram": {},
                "aspectHistogram": {},
                "locationHistogram": {}
            }
        },
        "requestConfig": [
            {
                "name": "SearchServiceDictionary.UserVisibleConstraints.Enabled",
                "value": [
                    "true"
                ]
            },
            {
                "name": "SearchServiceDictionary.WHOLE_PATH_LN_ENABLED",
                "value": [
                    "true"
                ]
            }
        ]
    },
    "answersPlatformRequest": {
        "supportedNavDestinationTypes": [
            "IMAGE_SEARCH"
        ],
        "appName": "SEARCH",
        "serviceConfig": {}
    }
}


def query_worker(b_64_img, page, res, url, category=None):
    global HEADERS
    try:
        logger.info("page [%d] for query [%s] - process started" % (page, url))
        response = requests.post(ENDPOINT, headers=HEADERS, json=request_body(b_64_img, page, category=category))
        res.append(response.json())
        logger.info("page [%d] for query [%s] - successfully completed" % (page, url))
    except:
        logger.error("page [%d] query [%s]" % (page, url))
        logger.error("error details:\n%s" % traceback.format_exc())


def request_body(image, num_results, page_num=None, category=None):
    b = copy.deepcopy(BODY_TEMPLATE)
    b['imageRequest']['paginationInput']['entriesPerPage'] = num_results
    b['imageRequest']['image'] = image
    if page_num:
        b['imageRequest']['paginationInput']['pageNumber'] = page_num
    if category:
        b['imageRequest']['categoryId'] = [category]
    return b


def download_b64(url):
    response = requests.get(url, verify=False, timeout=(5, 30))
    response.raise_for_status()
    dl_img = Image.open(StringIO(response.content))
    dl_img = resize(dl_img)
    buff = cStringIO.StringIO()
    dl_img.save(buff, quality=QUALITY, format='JPEG')
    return base64.b64encode(buff.getvalue())


def generator(responses, url):
    resp_list = list(responses)
    actual_pages = map(lambda x: x['meta']['pagination']['pageNumber'], resp_list)
    logger.info("query [%s]: actual pages returned: %s" % (url, sorted(actual_pages)))
    by_page = [None] * max(actual_pages)
    
    for resp in resp_list:
        # import pdb; pdb.set_trace()
        by_page[resp['meta']['pagination']['pageNumber'] - 1] = {"items": [x['item'][0]['item'] for x in resp['answers']['labeledItem']['answer']], "categoryId": {"category": resp['items']['categoryInfo']['firstAppliedCategory'], 'site': 0}, "categories": {'categories': map(lambda y: y['queryAction']['categoryId'][0],  resp['answers']['searchQuery']['answer'][0]['query']), "site": 0}}
    # for y in filter(lambda x: x is not None, itertools.chain.from_iterable(resp_list)):
        # yield y
        # by_page[resp['meta']['pagination']['pageNumber'] - 1] = resp
    for y in by_page:
        yield y
    logger.info("query [%s]: process completed successfully\n----------------------------------------" % url)

def resize(img):
    if (img.size[0] <= size_to_net and img.size[1] <= size_to_net):
        return img
    ratio = max(img.size) / float(size_to_net)
    return img.resize(tuple([int(i / ratio) for i in img.size]), Image.ANTIALIAS)


def api_query(post=None):
    global HEADERS
    if post is None:
        return "no data"
    logger.info('getting token')
    token = requests.get('http://localhost:8555/api/get_token')
    token.raise_for_status()
    HEADERS = {'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token.json()['results']['token'], 'X-EBAY-C-MARKETPLACE-ID': 'EBAY-US'}
    return [y for y in query(post.get('url'), post.get('num_results'), post.get('b64'), post.get('category'))]
    
def query(url, num_results, b64=None, category=None):
    if num_results == -1:
        num_results = sys.maxint
    manager = Manager()
    responses = manager.list()
    logger.info("downloading [%s]" % url)
    if not b64:
    	b_64_img = download_b64(url)
    else:
        b_64_img = b64
    logger.info("[%s] downloaded successfully. Sending initial query." % url)
    response = requests.post(ENDPOINT, headers=HEADERS, json=request_body(b_64_img, num_results, category=category), timeout=60)
    response.raise_for_status()
    # import pdb; pdb.set_trace()
    responses.append(response.json())
    pagination = response.json()['meta']['pagination']
    # 0 here used to disable paging. Remove to restore
    additional_pages = min(0, num_results/pagination['entriesPerPage'], pagination['totalPages'] - pagination['pageNumber'])
    logger.info("query [%s]: %d additional pages to query" % (url, additional_pages))
    if additional_pages > 0:
        processes = min(mp.cpu_count(), additional_pages)
        logger.info("query [%s]: using [%d] processes" % (url, processes))
        pool = mp.Pool(processes=processes)
        for x in xrange(2, 2 + additional_pages):
            pool.apply_async(query_worker, args=[b_64_img, x, responses, url, category])
        pool.close()
        pool.join()
        logger.info("query [%s]: additional pages query complete. Processing results." % url)
    return generator(responses, url)


# Crop and return b64 image
def crop_image(url, coords, margin):
    response = requests.get(url)
    if response.status_code == requests.codes.ok:
        dl_img = Image.open(StringIO(response.content))
        dl_img = dl_img.crop((max(coords['x1']-margin, 0), max(coords['y1']-margin, 0), min(coords['x1'] + coords['w'] + margin, dl_img.size[0]), min(coords['y1'] + coords['h'] + margin, dl_img.size[1])))
        dl_img = resize(dl_img)
        # dl_img = dl_img.resize((224, 224))
        buff = cStringIO.StringIO()
        dl_img.save(buff, quality=QUALITY, format='JPEG')
        return base64.b64encode(buff.getvalue())
    else:
	    return None
	
def api_ebay_image_search(url = None, num_results=20, coords=None, margin=10, category=None):
    global HEADERS
    print ">>>", url
    start = datetime.datetime.now()

    num_results = min(100, num_results)
    
    logger.info('getting token') 
    token = requests.get('http://localhost:8555/api/get_token')
    token.raise_for_status()
    HEADERS = {'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token.json()['results']['token'], 'X-EBAY-C-MARKETPLACE-ID': 'EBAY-US'}
    logger.info('token received')
    logger.info("----------------------------------------\nReceived query: [%s]" % url)
    if not url:
        logger.info("query empty. returning empty result")
        return []
    try:
        if coords:
            cropped = map(lambda x: {"img": crop_image(url, x, margin), "coords": x}, coords)
            results = []
            for entry in cropped:
                img_id = uuid.uuid1().get_hex()
                item = {"coords": entry["coords"], "margin": margin, 's3_url': 'https://s3.amazonaws.com/caffemodels/ebay-image-search/'+img_id+'.jpg'}
                if not entry.get("img"):
                    # item["results"] = []
                    pass
                else:
                    upload_b64_to_s3(entry.get("img"), img_id)                
                    # item["results"] = [g for g in query(url, num_results, entry["img"], category)]
                    item.update([g for g in query(url, num_results, entry["img"], category)][0])
                    results.append(item)
        else:
            results = [g for g in query(url, num_results, category=category)]
    except:
        logger.error("error processing %s" % url)
        logger.error(traceback.format_exc())
        raise
    finally:
        delta = datetime.datetime.now() - start

    # return {'items': results, 'total items': len(results), 'milliseconds': int(delta.total_seconds() * 1000)}
    return results

app.wsgi_app = ProxyFix(app.wsgi_app)

if __name__ == '__main__':
    app.run('0.0.0.0', port=4444)
    # res_gen = query('https://cdn.shopify.com/s/files/1/0377/2037/products/WhiteTanLeather.Front_1024x.jpg?v=1510683461')
    # for n, x in enumerate([g for g in res_gen]):
    #     print "(%d) %s" % (n + 1, x)
