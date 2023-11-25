# Introduction to machine learning: Work 1

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
4. Then, we install the required dependencies
```bash
pip install -r requirements.txt
```
you can check if dependencies were installed by running next
command, it should print list with installed dependencies
```bash
pip list
```
5. Finally we close the virtual environment
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
   * **Compute cluster centroids**: we can compute the centroids of with any dataset with any method we 
     want from the list. Obvioulsy, we cannot get the centers of agglomerative clustring.
   * **Compute dendrogram of a dataset**: We can compute the dendrogram of a dataset by using this option.
     We only need to select the dataset we want.
   * **Compute the clustering validations plot for a model**: We can compute the clustering methods for any
     model and dataset. Here we have avoided MeanShift because we cannot decide the number of clusters for MeanShift
   * **Compute the confusion matrix**: Similarly to the previous action, we can compute the confusion of a model given
    a dataset.
   * **Run Experiments**: Finally, we can also run the experiments we did to compute the best parameters for
    each model. There are 7 possible experiments. These experiments are described in the menu.
   
Finally, we would like to mention that we have not commented about the whole menu. In order to get a better understanding of the project, 
we just need to follow the instructions on the terminal.

4. At the end, we close virtual environment.
```bash
deactivate
```
