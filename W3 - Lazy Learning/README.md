# Introduction to machine learning: Work 3

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
.\venv\Scripts\activate.bat
```
4. Then, we install the required dependencies (you may need to update your pip version, if needed the console will tell you how)
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
.\venv\Scripts\activate.bat
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
    * **Compute the Knn over a particular dataset**: In this option, we can select to apply the KNN with the possible parameters over a particular dataset. In our directory, we have two datasets: satimage and cmc. Moreover, the possible options for the KNN are the ones described in the work assignment.
    * **Apply instance reduction techniques and do the Knn**:  Here, we can select a particular KNN and apply it to a dataset after doing instance reduction. There are three possible techniques: GCNN, ENNTh, and DROP3. Please enter the parameters that the program demands. Furthermore, keep in mind that some instance reduction techniques may take a long time.
    * **Do Knn over previously reduced datasets**: In this option, we can apply Knn over a dataset that has previously been reduced using instance reduction techniques. We would like to add that the DROP3 option only has one reduced dataset for every dataset. This is due to the fact that it takes a lot of time to run. The reduced datasets that are available are the ones that we computed during the experimentation part.
    * **Run Experiments**: Finally, we can also run the experiments we did for the project. These experiments are described in the menu. Please be aware that some experiments may generate a lot of windows and may take a lot of time; be patient, please.
   
Finally, we would like to mention that we have not commented about the whole menu. In order to get a better understanding of the project, we just need to follow the instructions on the terminal.

4. At the end, we close virtual environment.
```bash
deactivate
```
