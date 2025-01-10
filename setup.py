from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    This function reads the requirements from a file and returns them as a list.
    '''
    requirement = []

    with open(file_path) as file_object:
        requirement = file_object.readlines()
        requirement = [ req.replace('\n',"") for req in requirement] 
    return requirement


setup(
    name='Hotel-Booking-Prediction',
    version='1.0',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    author='Yash Patel',
    author_email='yashjpatel2003@example.com',

)