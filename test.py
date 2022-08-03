#!/usr/bin/env python
# coding: utf-8

import requests

adult = {
    "age": 25,
    "workclass": " Private",
    "fnlwgt": 226802,
    "education": " 11th",
    "education-num": 7,
    "marital-status": " Never-married",
    "occupation": " Machine-op-inspct",
    "relationship": " Own-child",
    "race": " Black",
    "sex": " Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": " United-States"
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=adult)
print()
print(response.json())
