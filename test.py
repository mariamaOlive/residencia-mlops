#!/usr/bin/env python
# coding: utf-8

import requests

'''adult = {
    "age": 25,
    "workclass": " Private",
    "fnlwgt": 226802,
    "education": " 11th",
    "education-num": 7,
    "marital-status": " Never-married",
    "occupation": " Exec-managerial",
    "relationship": " Own-child",
    "race": " Black",
    "sex": " Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": " United-States"
}'''

# Entrada para fazer a predição
adult = {
    "age": 50,
    "workclass": " Private",
    "fnlwgt": 289551,
    "education": " Doctorate",
    "education-num": 16,
    "marital-status": " Married-civ-spouse",
    "occupation": " Exec-managerial",
    "relationship": " Husband",
    "race": " White",
    "sex": " Male",
    "capital-gain": 55000,
    "capital-loss": 1000,
    "hours-per-week": 40,
    "native-country": " United-States"
}

# Faz a requisição
url = 'http://localhost:9696/predict'
response = requests.post(url, json=adult)
print()
print(response.json())
