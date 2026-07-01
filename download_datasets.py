import urllib.request
import concurrent.futures

urls = {
    'ts_with_dup_b97d3.tar.gz': 'https://zenodo.org/records/3715478/files/ts_with_dup_b97d3.tar.gz?download=1',
    'ts_with_dup_wb97xd3.tar.gz': 'https://zenodo.org/records/3715478/files/ts_with_dup_wb97xd3.tar.gz?download=1',
    'wb97xd3.tar.gz': 'https://zenodo.org/records/3715478/files/wb97xd3.tar.gz?download=1'
}

def d(item):
    name, url = item
    print(f'Downloading {name}...')
    urllib.request.urlretrieve(url, name)
    print(f'Done {name}')

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as e:
    e.map(d, urls.items())
