import sys; sys.stdout.reconfigure(encoding='utf-8')
import requests
from pathlib import Path
from PIL import Image
import hashlib

video_id = 'vJ6FhP_qMkk'

# Fetch the thumbnail YouTube has stored using signed URL from last upload
# We need to re-upload thumbnail to get the signed URL, or use public endpoints
# Public endpoints don't work for private videos, so let's use the upload API response

# Actually, let's try fetching via the thumbnail set endpoint to see what's there
from youtube.uploader import load_upload_config, _refresh_access_token

cfg = load_upload_config()
token = _refresh_access_token(cfg)

# Try to get video snippet to see thumbnail URLs
# This requires read scope which we don't have, but let's try
r = requests.get(
    'https://www.googleapis.com/youtube/v3/videos',
    params={'part': 'snippet', 'id': video_id},
    headers={'Authorization': 'Bearer ' + token},
    timeout=30,
)
print(f'Videos.list status: {r.status_code}')
if r.status_code == 200:
    items = r.json().get('items', [])
    if items:
        thumbs = items[0].get('snippet', {}).get('thumbnails', {})
        for key, val in thumbs.items():
            print(f'  {key}: {val.get("url", "?")}')
            print(f'       width={val.get("width")}, height={val.get("height")}')
    else:
        print('  No items found')
else:
    print(f'  Error: {r.text[:200]}')

# Try downloading the thumbnail directly from YouTube
# For private videos, the standard URLs return 404, but let's try the i9 signed URLs
# We'll need to re-upload to get fresh signed URLs
print()
print('Re-uploading thumbnail to get fresh signed URL...')
thumb_path = Path(r'C:\Users\harsh\Youtube_Automation_system\output\thumbnails\thumbnail.jpg')
data = thumb_path.read_bytes()

r2 = requests.post(
    'https://www.googleapis.com/upload/youtube/v3/thumbnails/set',
    data=data,
    headers={
        'Authorization': 'Bearer ' + token,
        'Content-Type': 'image/jpeg',
        'Content-Length': str(len(data)),
    },
    params={'videoId': video_id},
    timeout=60,
)
print(f'Thumbnail upload status: {r2.status_code}')
resp = r2.json()
items = resp.get('items', [])
if items:
    for key in ['default', 'medium', 'high', 'standard', 'maxres']:
        if key in items[0]:
            url = items[0][key].get('url', '')
            print(f'  {key}: {url[:120]}')
            # Download this thumbnail
            if key == 'maxres':
                r3 = requests.get(url, timeout=15)
                yt_path = Path(r'C:\Users\harsh\Youtube_Automation_system\output\yt_thumb_v2.jpg')
                yt_path.write_bytes(r3.content)
                img = Image.open(yt_path)
                print(f'  Downloaded: {len(r3.content)} bytes, {img.size}')
                print(f'  MD5: {hashlib.md5(r3.content).hexdigest()}')
                
                # Compare with local
                local_hash = hashlib.md5(thumb_path.read_bytes()).hexdigest()
                print(f'  Local MD5:  {local_hash}')
                print(f'  Match: {local_hash == hashlib.md5(r3.content).hexdigest()}')
