# Crop Yield Prediction with Deep Image Detection Algorithms

This is the codebase for the article named after the title by me, Petteri Nevavuori, and prof. Tarmo Lipping. 

A multitude of things are omitted, such as the generated dabases (~0.5 GB to +5GB), generated intermediary models (~40MB a piece) and all the original files from which the databases were generated from. What is however included are the codebase found under ``python/field_analysis/`` and the Jupyter Notebooks utilizing the codebase in conjuction with the data files found at the root of ``python/``. Also the final models are also present in this repository.

*Sometimes the pages fail to load. This is due to failure in serving large ipynb-files from Github to an external preview service. This is a Github-side problem, bu reload should alleviate it.*


## Contents

The notebooks prefixed `GDAL` were run in an environment compliant with the `gdal` package. The other notebooks with the `ML` prefix have been run with an environment which had PyTorch installed. The former has an upper limit for the Python version, while the latter is aimed and updated for the latest Python version.

- [GDAL I. Field-Wise Image Dataset Extraction](https://nbviewer.jupyter.org/github/karmus89/crop-yield-cnn/blob/master/GDAL%20I.%20Field-Wise%20Image%20Dataset%20Extraction.ipynb)
- [GDAL II. Weather Dataset Retrieval](https://nbviewer.jupyter.org/github/karmus89/crop-yield-cnn/blob/master/GDAL%20II.%20Weather%20Dataset%20Retrieval.ipynb)
- [GDAL III. Field-Wise Yield Dataset Extraction](https://nbviewer.jupyter.org/github/karmus89/crop-yield-cnn/blob/master/GDAL%20III.%20Field-Wise%20Yield%20Dataset%20Extraction.ipynb)
- [GDAL IV. Dataset DB Creation](https://nbviewer.jupyter.org/github/karmus89/crop-yield-cnn/blob/master/GDAL%20IV.%20Dataset%20DB%20Creation.ipynb)
- [ML I. Drone Datasets](https://nbviewer.jupyter.org/github/karmus89/crop-yield-cnn/blob/master/ML%20I.%20Drone%20Datasets.ipynb)
- [ML II. Building the CNN](https://nbviewer.jupyter.org/github/karmus89/crop-yield-cnn/blob/master/ML%20II.%20Building%20the%20CNN.ipynb)
- [ML III. CNN Optimization](https://nbviewer.jupyter.org/github/karmus89/crop-yield-cnn/blob/master/ML%20III.%20CNN%20Optimization.ipynb)

---

*To use and edit these files an Anaconda installation of Python with Peewee, OSGEO and PyTorch are required. The contents are viewable within this Github repo.*

*All diagrams have been made with [draw.io](https://www.draw.io/).*
