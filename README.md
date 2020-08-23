# Open-Source License Validator

## Table of Contents

- [Overview](#Overview)
- [How to Load Project](#how-to-load-project)
- [Features](#Features)
- [Technologies Used](#technologies-used)
- [Acknowledgements](#acknowledgements)

## Overview

The open-source compliance team at Blackberry QNX is responsible for determining open-source licenses since it is used to comply with the legal terms of theses licenses before a product can be released into the public. This is done through a software pipeline that gets the source code information from a package found in a Jenkins server where comment blocks are then extracted and then predicted upon by a machine learning model called fastText to determine the specific license a comment block is referring to. These comment blocks along with any license information is then inserted into a MongoDB database. The problem that is being faced is that there are many false positives found in the database where certain comment blocks have absolutely no information about licensing, which are considered invalid licenses. 

This project and provided report aims to provide the decision process and analysis of a machine learning experiment to determine whether machine learning can be used to accurately identify whether a comment block found in source code contains open-source license information. This is a binary classification natural language processing task since the potential prediction of a model is that given a comment block (which is text), determine if it contains license information or not. The two approaches taken were unsupervised and supervised learning. 

Unsupervised learning was initially used since the dataset that will be used to train models may contain misclassifications because the data in the database is not actively monitored. Since this type of learning does not use labels, it can bypass this problem. One approach that was used was a K-means clustering algorithm that would attempt to cluster the comments into groups. This approach did not cluster groups into the groups wanted since the ideal number of groups found using the elbow method was about 54. Another approach that was used was an autoencoder that would attempt to replicate comments that contained license information. From testing, it was clear that the model was not able to generalize on a comment containing a license compared to ones that didn’t. This can be due to the various open-source licenses that can be found in software.

Supervised learning was then used where the data was initially cleaned up as much as possible and then preprocessed. Three different models that may perform well on this NLP task was used. Hyperparameter tuning was conducted either through Cross Validation or automatically if the model allowed it. Precision and recall along with a fast prediction time were used to determine how well a model performed and after testing on a training and test dataset, it was determining that the fastText model performed quite well and satisfied the constraints and criteria placed. Future training and testing should be used to further validate the obtained results and placing the model in a live production environment is a way to do so. 

**Note: The data and code has been altered significantly so that this project can be open-sourced. If any problems arise when working with this project, please raise an issue and I will look into it as soon as possible.**

## How to Load Project



## Features

- See how dots' movements evolve as they find their way around obstacles to reach the goal
- Can add more obstacles to see how dots behave
- Implemented the Genetic Algorithm from scratch (fitness calculation, natural selection, cross-over, mutation)
- Use of OOP in Python can be used as template for future projects

## Technologies Used

- [PyCharm](https://www.jetbrains.com/pycharm/) - IDE used to build the game
- [Python 3.8.3](https://www.python.org/downloads/) - Programming language used
- [Pandas](https://pandas.pydata.org/) - Data analysis and manipulation library
- [Matplotlib](https://matplotlib.org/) - Plotting/graphing library
- [Seaborn](https://seaborn.pydata.org/) - Statistical data visualization library
- [Scikit-learn](https://scikit-learn.org/stable/) - Machine learning library
- [fastText](https://fasttext.cc/) - One of the supervised ML models used

## Acknowledgements

- Thanks to [Blackberry](https://www.blackberry.com/us/en) and Mike Koch for providing me with the tools and skills necessary to complete this task
