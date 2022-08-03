#!/usr/bin/env python
# coding: utf-8

import requests

ride = {
    "PULocationID": 20,
    "DOLocationID": 30,
    "trip_distance": 40
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print()
print(response.json())
