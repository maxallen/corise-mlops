import os
from fastapi.testclient import TestClient
from .app.server import app

os.chdir('app')

"""
We've built our web application, and containerized it with Docker.
But imagine a team of ML engineers and scientists that needs to maintain, improve and scale this service over time. 
It would be nice to write some tests to ensure we don't regress! 

  1. `Pytest` is a popular testing framework for Python. If you haven't used it before, take a look at https://docs.pytest.org/en/7.1.x/getting-started.html to get started and familiarize yourself with this library.

  2. How do we test FastAPI applications with Pytest? Glad you asked, here's two resources to help you get started:
    (i) Introduction to testing FastAPI: https://fastapi.tiangolo.com/tutorial/testing/
    (ii) Testing FastAPI with startup and shutdown events: https://fastapi.tiangolo.com/advanced/testing-events/
"""


def test_root():
    """
    Test the root ("/") endpoint, which just returns a {"Hello": "World"} json response
    """
    with TestClient(app) as client:
        resp = client.get("/") 
        assert resp.status_code == 200
        assert resp.json() == {"Hello": "World"}


def test_predict_empty():
    """
    Test the "/predict" endpoint, with an empty request body
    """
    with TestClient(app) as client:
        resp = client.post("/predict", json={})
        assert resp.status_code == 500


def test_predict_en_lang():
    """
    Test the "/predict" endpoint, with an input text in English (you can use one of the test cases provided in README.md)
    """
    with TestClient(app) as client:
        resp = client.post("/predict", json={"source": "WESH.com", "url": "http://www.rockdalecitizen.net/sc/archive/2004/5214.htm", "title": "Hazard Level Upped At Mount St. Helens", "description": "VANCOUVER, Wash. -- A second steam eruption at Mount St. Helens and changes in seismic signals prompted the government to raise the volcano alert to Level Three on Saturday afternoon."})
        assert resp.status_code == 200


def test_predict_es_lang():
    """
    Test the "/predict" endpoint, with an input text in Spanish. 
    Does the tokenizer and classifier handle this case correctly? Does it return an error?
    """
    with TestClient(app) as client:
        resp = client.post("/", json={"source": "Me", "url": "", "title": "", "description": "VEVEY, Suiza—Nestlé SA suspendió su meta de crecimiento anual de ventas al menos durante los próximos tres años, en momentos en que el gigante suizo de alimentos procesados lucha junto al resto del sector contra una inflación sumamente baja y los rápidos cambios en los gustos de los consumidores."})
        assert resp.status_code == 200

def test_predict_non_ascii():
    """
    Test the "/predict" endpoint, with an input text that has non-ASCII characters. 
    Does the tokenizer and classifier handle this case correctly? Does it return an error?
    """
    with TestClient(app) as client:
        resp = client.post("/predict", json={"source": "Me", "url": "", "title": "", "description": "VEVEY, Suiza—Nestlé SA suspendió su meta de crecimiento anual de ventas al menos durante los próximos tres años, en momentos en que el gigante suizo de alimentos procesados lucha junto al resto del sector contra una inflación sumamente baja y los rápidos cambios en los gustos de los consumidores."})
        assert resp.status_code == 200