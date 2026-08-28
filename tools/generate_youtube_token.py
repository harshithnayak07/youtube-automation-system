from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload"
]

flow = InstalledAppFlow.from_client_secrets_file(
    "config/client_secret.json",
    SCOPES
)

credentials = flow.run_local_server(
    port=0,
    access_type="offline",
    prompt="consent"
)

print("\nREFRESH TOKEN:")
print(credentials.refresh_token)