from setuptools import setup

setup(
    name="dta_ocr",
    version="1.0",
    install_requires=["calamari-ocr>=2.1.4", "kraken>=3.0", "Pillow>=8.4"],
    packages=["dta_ocr"]
)
