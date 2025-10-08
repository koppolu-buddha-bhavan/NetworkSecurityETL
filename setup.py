from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path)->List[str]:
    packages_lst:List[str]=[]
    with open(file_path,"r") as obj:
        lines = obj.readlines()
        for line in lines:
            line = line.strip()
            if line and line!="-e .":
                packages_lst.append(line)
    
    return packages_lst

setup(
    name="NetworkSecurity",
    version= "0.0.1",
    author="Surya",
    author_email="surya19teja.sripati@gmail.com",
    packages= find_packages(),
    install_requires = get_requirements("requirements.txt")
)



