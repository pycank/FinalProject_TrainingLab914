import requests

# Zenodo access token
access_token = 'cN901ggej5JQ0KgoaNbHt9JLx8TDSKMba5OayofxpmMxqwARykRhJsKa4uQ8'

# Path to the checkpoint file
# checkpoint_path = 'checkpoint.pth'
checkpoint_path = './loader.py'

# Zenodo API endpoint for creating a new deposition
deposition_url = 'https://zenodo.org/api/deposit/depositions'

# Create a new deposition
headers = {"Content-Type": "application/json"}
params = {'access_token': access_token}
r = requests.post(deposition_url, params=params, headers=headers)
print(r.json())
deposition_id = r.json()['id']
print(f"Deposition ID: {deposition_id}")

# Upload the checkpoint file
files = {'file': open(checkpoint_path, 'rb')}
upload_url = f'https://zenodo.org/api/deposit/depositions/{deposition_id}/files'
r = requests.post(upload_url, params=params, files=files)

# Publish the deposition
publish_url = f'https://zenodo.org/api/deposit/depositions/{deposition_id}/actions/publish'
r = requests.post(publish_url, params=params)

print("Checkpoint uploaded and published successfully!")
