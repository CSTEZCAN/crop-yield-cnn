# Crop Yield Prediction with Deep Image Detection Algorithms

This is the codebase for the article named after the title by me, Petteri Nevavuori, and prof. Tarmo Lipping. 

A multitude of things are omitted, such as the generated dabases (~0.5 GB to +5GB), generated models (~40MB a piece) and all the original files from which the databases were generated from. What is however included are the codebase found under ``python/field_analysis/`` and the Jupyter Notebooks utlizing the codebase in conjuction with the data files found at the root of ``python/``.

There are a few gotchas. Let's go through them.

## Project Rundown

The whole course of the project with all the intermediary steps are given in the Jupyter Notebooks. In the first phase the creation of the datasets is covered (``GDAL*.ipynb``). In the second phase we go through the creation and the training of the model (``ML*.ipynb``). The classes and functions referenced from the package ``field_analysis`` were developed by me, Petteri Nevavuori.

## Separate Python Envs

To succesfully run the project, there were two separate Python environments made. 

The first was used to utilize the functionality of the ``osgeo`` package for handling geo-referenced data. During the period of coding the project the package couldn't be run in Python version higher than 3.4.*. The notebooks utilizing this env are prefixed as ``GDAL`` according to the ``osgeo.gdal`` package for reading and manipulating GeoTIFF-rasters. These notebooks relate exclusively to the building of the datasets.

The second was then created to use the PyTorch build for Windows effectively. The notebooks utilizing this environment are prefixed ``ML``.

---

Any comments or questions are welcome!