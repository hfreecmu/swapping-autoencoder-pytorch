import flickrapi
from PIL import Image
import urllib
import os
import argparse
import cv2
import json

flickr_json_path = 'flickr_jsons'
flickr_cred_suffix = '_creds.json'

method_choices = ['download_images', 'save_flickr_creds']
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=method_choices)
    parser.add_argument('--flickr_cred_name', required=True)

    #required for download_images
    parser.add_argument('--keyword', default=None)
    parser.add_argument('--output_dir', default=None)

    parser.add_argument('--num_images', type=int, default=100)

    args = parser.parse_args()
    return args

def read_json(path):
    with open(path) as f:    
        data = json.load(f)
    return data

def write_json(path, data):
    json.dump(data, open(path,'w'))

def read_flickr_json(flickr_cred_name):
    path = os.path.join(flickr_json_path, flickr_cred_name + flickr_cred_suffix)
    if not os.path.exists(path):
        raise RuntimeError('flickr_cred_name not found: ' + flickr_cred_name)
    data = read_json(path)
    return data['api_key'], data['api_secret']

def write_flickr_json(api_key, api_secret, flickr_cred_name):
    data = {'api_key': api_key, 'api_secret': api_secret}
    path = os.path.join(flickr_json_path, flickr_cred_name + flickr_cred_suffix)
    write_json(path, data)

def save_flickr_creds(flickr_cred_name):
    api_key = input('Enter api key: ')
    api_secret = input('Enter api secret: ')
    write_flickr_json(api_key, api_secret, flickr_cred_name)

def download_images(flickr_cred_name, keyword, output_dir, num_images):
    api_key, api_secret = read_flickr_json(flickr_cred_name)

    flickr = flickrapi.FlickrAPI(api_key, api_secret, cache=True)

    keyword_no_space = keyword.replace(" ", "_")

    photos = flickr.walk(text=keyword,
                    tag_mode='all',
                    tags=keyword,
                    extras='url_c',
                    per_page=200,
                    sort='relevance')

    urls = []
    for i, photo in enumerate(photos):    
        url = photo.get('url_c')
        urls.append(url)
    
        if i > num_images:
            break

    for i in range(num_images):
        try:
            print('Downloading image: ', i)
            jpg_path = os.path.join(output_dir, keyword_no_space + "_{:06d}.jpg".format(i))
            urllib.request.urlretrieve(urls[i], jpg_path)
        except:
            continue

        im = cv2.imread(jpg_path)
        png_path = jpg_path.replace(".jpg", ".png")
        cv2.imwrite(png_path, im)
        os.remove(jpg_path)


if __name__ == "__main__":
    args = parse_args()

    method = args.method
    flickr_cred_name = args.flickr_cred_name
    keyword = args.keyword
    output_dir = args.output_dir
    num_images = args.num_images

    if method == 'download_images':
        if keyword is None:
            raise RuntimeError('keyword required when method is download_images')
        
        if output_dir is None:
            raise RuntimeError('output_dir required when method is download_images')

        if not os.path.exists(output_dir):
            raise RuntimeError('Invalid output dir')

        download_images(flickr_cred_name, keyword, output_dir, num_images)
    elif method == 'save_flickr_creds':
        save_flickr_creds(flickr_cred_name)
    else:
        raise RuntimeError('Illegal method specified: ' + method)

