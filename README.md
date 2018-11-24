# Crop Yield Prediction with Deep Image Detection Algorithms

This is the codebase for the article named after the title by me, Petteri Nevavuori, and prof. Tarmo Lipping. 

A multitude of things are omitted, such as the generated dabases (~0.5 GB to +5GB), generated intermediary models (~40MB a piece) and all the original files from which the databases were generated from. What is however included are the codebase found under ``python/field_analysis/`` and the Jupyter Notebooks utilizing the codebase in conjuction with the data files found at the root of ``python/``. Also the final models are also present in this repository.

## Contents

The notebooks prefixed `GDAL` were run in an environment compliant with the `gdal` package. The other notebooks with the `ML` prefix have been run with an environment which had PyTorch installed. The former has an upper limit for the Python version, while the latter is aimed and updated for the latest Python version.

- [GDAL I. Field-Wise Image Dataset Extraction](http://htmlpreview.github.io/?https://github.com/karmus89/crop-yield-cnn/blob/master/html/GDAL%20I.%20Field-Wise%20Image%20Dataset%20Extraction.html)
- [GDAL II. Weather Dataset Retrieval](http://htmlpreview.github.io/?https://github.com/karmus89/crop-yield-cnn/blob/master/html/GDAL%20II.%20Weather%20Dataset%20Retrieval.html)
- [GDAL III. Field-Wise Yield Dataset Extraction](http://htmlpreview.github.io/?https://github.com/karmus89/crop-yield-cnn/blob/master/html/GDAL%20III.%20Field-Wise%20Yield%20Dataset%20Extraction.html)
- [GDAL IV. Dataset DB Creation](http://htmlpreview.github.io/?https://github.com/karmus89/crop-yield-cnn/blob/master/html/GDAL%20IV.%20Dataset%20DB%20Creation.html)
- [ML I. Drone Datasets](http://htmlpreview.github.io/?https://github.com/karmus89/crop-yield-cnn/blob/master/html/ML%20I.%20Drone%20Datasets.html)
- [ML II. Building the CNN](http://htmlpreview.github.io/?https://github.com/karmus89/crop-yield-cnn/blob/master/html/ML%20II.%20Building%20the%20CNN.html)
- [ML III. CNN Optimization](http://htmlpreview.github.io/?https://github.com/karmus89/crop-yield-cnn/blob/master/html/ML%20III.%20CNN%20Optimization.html)
- [Appendix A. Assessing the Model](http://htmlpreview.github.io/?https://github.com/karmus89/crop-yield-cnn/blob/master/html/Appendix%20A.%20Assessing%20the%20Model.html)

- [ML III. CNN Optimization](https://github.com/karmus89/crop-yield-cnn/blob/master/html/ML%20III.%20CNN%20Optimization.html)


---

*To use and edit these files an Anaconda installation of Python with Peewee, OSGEO and PyTorch are required. The contents are viewable within this Github repo.*

*All diagrams have been made with [draw.io](https://www.draw.io/).*
