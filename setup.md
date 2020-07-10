# Setup
> [!Warning]
> Python 3.8 is needed for this project to work correctly. Get it <a href="https://www.python.org/downloads/">here</a>.

## Required libraries 
We use <a href="https://pipenv.pypa.io/en/latest/">pipenv</a> to ensure, that everything will be setup automatically. 

Just install <a href="https://pipenv.pypa.io/en/latest/">pipenv</a> with 

```shell
pip install pipenv
```
and use 
```shell
git clone https://github.com/LuanAdemi/GroceryListPredictor.git
```
to clone the repository conataining the project.

Afterwards, enter the directory, you cloned the project into and enter the pipenv shell with

```shell
pipenv shell
```
This will build a virtual environment for the project, which is completely independent from your local installation of python.

Now, simply type in
```shell
pipenv install
```
and all the needed libraries will be installed within the virtual environment. 

## Run the webapp

To start the webapp, enter the pipenv shell (if you haven't already) and simply run

```shell
python run.py
```