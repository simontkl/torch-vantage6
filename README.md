
# PyTorch model implementation into vantage6 infrastructure
This implementation uses the (https://vantage6.ai) solution. Vantage6 allows to execute computations on federated datasets. 

As of now, the implementation consists of file structure and slight adjustments to the code examples and guideline as outlined in the vantage6 documentation (https://docs.vantage6.ai/).

It should change to include or be replaced by simplemodel and coordination.


### Dockerfile
The dockerfile contains the algorithm package `ARG PKG_NAME='v6-ppsdg-py`. 

## {v6-ppsdg-py} - Algorithm package

### {v6-ppsdg-py}/__init__.py
Contains all the methods that can be called at the nodes. All __regular__ definitions in this file that have the prefix `RPC_` are callable by an external party. If you define a __master__ method, it should *not* contain the prefix! The __master__ and __regular__ definitions both have there own signature. __Master__ definitions have a __client__ and __data__ argument (and possible some other arguments), while the __regular__ definition only has the __data__ argument. The data argument is a [pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html?highlight=dataframe#pandas.DataFrame) and the client argument is a `ClientContainerProtocol` or `ClientMockProtocol` from the [vantage6-toolkit](https://github.com/IKNL/vantage6-toolkit). The master and regular definitions signatures should look like:
```python
def some_master_name(client, data, *args, **kwargs):
    # do something
    pass

def RPC_some_regular_method(data, *args, **kwargs):
    # do something
    pass
```
This example is taken from (https://github.com/IKNL/v6-boilerplate-py). The functions are the same as in (https://github.com/IKNL/vantage6-client/blob/master/vantage6/tools/mock_client.py) and (https://github.com/IKNL/vantage6-client/blob/master/vantage6/client/__init__.py).

### client.py

Contains the client functions.

### v6simplemodel.py

Contains the PyTorch model (Author: Carlijn Nijhuis).

### v6train.py

The training of the model at node. Original makes use of coordination (which would be my client.py), hence does it need to be incorporated in client.py? 

### db.py

Will contain the database partitioning algorithm (according to Yu).

--------------------

## Util
Note: .yaml files are stored elsewhere. These are copies for comparison.

### entities.yaml

This file contains the organisations/entities.

### node_configuration.yaml

Contains the configuration of the nodes. This is a copy of the actual file that is in C:\Users\simon\AppData\Local\vantage6\node\node_configuration and change it there as well

### v6server.yaml

This file contains the server configuration.

### setup.py
In order for the Docker image to find the methods the algorithm needs to be installable. Make sure the *name* matches the `ARG PKG_NAME` in the Dockerfile.

-------

# TODO



Package structure:

```python
vantage6-v2.0
├── Dockerfile
├── setup.py
└── algorithm_pkg
    └── __init__.py
    └── client.py
    └── db.py
    └── v6simplemodel.py
    └── v6train.py
```

Here, __init__.py is the algorithm which will import v6simplemodel.py, v6train.py, and db.py.

