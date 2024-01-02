import requests

url = 'http://localhost:5000/api/'

payload = {
	'exp':1.8
}

r = requests.post(url,json=payload)

print(r.json())
