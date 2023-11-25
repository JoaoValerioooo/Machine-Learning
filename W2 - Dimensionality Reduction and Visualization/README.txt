# Introduction to machine learning: Work 2

## Team members

* Hasnain Shafqat
* João Francisco Agostinho Valério
* Eirik Grytøyr

## Running scripts for the first time
In order to run our project for the first time, we need to prepare our computer in order to 
be able to execute the project. First of all, we need to install python 3.7 (extremely important) in our system. Then, we need to create a virtual environment:
1. Open in terminal the folder where our project is located
```bash
cd <root_folder_of_project>/
```
2. Once we have opened the folder, we create virtual environment

In linux operative system execute the following line of code:
```bash
python3.7 -m venv venv/
```

In Windows operative system execute the following line of code:
```bash
python -m venv venv/
```

3. We proceed to open the virtual environment

In linux operative system execute the following line of code:
```bash
source venv/bin/activate
```
In Windows operative system (Important installing Virtualenv) execute the following line of code:
```bash
venv/Scripts/activate
```
4. Then, we install the required dependencies (you may need to update your pip version, if needed, the console will tell you how)
```bash
pip install -r requirements.txt
```
you can check if dependencies were installed by running next
command, it should print list with installed dependencies
```bash
pip list
```
5. Finally, we close the virtual environment
```bash
deactivate
```

## Project execution

In order to execute our project, we need to run the main python file. For that, we need to open
the virtual environment we created previously.

1. Open the virtual environment

In linux operative system execute the following line of code:
```bash
source venv/bin/activate
```
In Windows operative system (Important installing Virtualenv) execute the following line of code:
```bash
venv/Scripts/activate
```
2. Then, to execute the project, we run the ``main.py`` file.

In linux operative system execute the following line of code:
```bash
python3.7 main.py
```
In Windows operative system execute the following line of code:
```bash
python main.py
```

3. In the main we will have the possible actions we can do in this project. The possible actions
  are the following ones:
   * **Apply feature reduction to a particular dataset**: We can apply dimensionality reduction using any dataset and 
     any method described in this work. There are some techniques for feature reduction that don't use eigenvectors, so 
     for these we will just plot the first 5 rows of the transformed data. For the others, we will print the eigenvectors also.
   * **Apply feature reduction to a dataset and compute the clustering**: In this option, we first apply feature reduction and
     then we apply clustering. The clustering methods available are K-means and agglomerative clustering. For KMeans, we will display the
     cluster centers, and for the agglomerative clustering, the dendrogram.
   * **Run Experiments**: Finally, we can also run the experiments we did for the project. These experiments are 
     described in the menu. Please be aware that some experiments may generate a lot of windows; be patient, please.
   
Finally, we would like to mention that we have not commented about the whole menu. In order to get a better understanding of the project, 
we just need to follow the instructions on the terminal.

4. At the end, we close virtual environment.
```bash
deactivate
```
