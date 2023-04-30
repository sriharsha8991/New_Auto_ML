from setuptools import find_packages,setup
from typing import List

E = "-e ."
def find_req(file_path:str)->List[str]:
    '''
    helps to return the llist of reqirements
    '''
    req = []
    with open(file_path) as file_obj:
        req = file_obj.readlines()
        req = [i.replace('\n','') for i in req]

        if E in req:
            req.remove(E)

    return req


setup(
    name="Model",
    version="0.0.0.1",
    author = ["Nalin","Sri","Kavin","Nandu"],
    packages = find_packages(),
    install_requires = find_req('Requirements.txt')
)