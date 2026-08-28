import os
import requests
from dotenv import load_dotenv

load_dotenv()

response = requests.post(
    "https://oauth2.googleapis.com/token",
    data={
        "client_id": os.environ["YOUTUBE_CLIENT_ID"],
        "client_secret": os.environ["YOUTUBE_CLIENT_SECRET"],
        "refresh_token": os.environ["YOUTUBE_REFRESH_TOKEN"],
        "grant_type": "refresh_token",
    },
)

print("HTTP Status:", response.status_code)

if response.ok:
    data = response.json()
    print("✅ YouTube refresh token is valid")
    print("Access token received:", bool(data.get("access_token")))
else:
    print("❌ Token refresh failed")
    print(response.text)