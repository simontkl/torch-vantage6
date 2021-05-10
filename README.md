
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

### v6coordination.py

REVIEW!! "Description: This module contains functions for coordinating the workers and
server."; But also: average_gradients, average_parameters, average_parameters_weighted which are torch functions and not torch.distributed. Are these needed?? I assume they're calculated locally?

### v6train.py

The training of the model at node. Original makes use of coordination (which would be my client.py), hence does it need to be incorporated in client.py? 

### db.py

Will contain the database partitioning algorithm (according to Yu).

--------------------

## Util

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

- connect dots between torch.distributed and vantage6

torch.distributed: simplemodel, train, coordination, data, parser, main
  
v6: v6simplemodel, v6train, v6coordination (???), client, db, dockerfile, entities, node_configuration, v6server, setup

| dist        | v6           | comment  |
| ------------- |-------------| -----|
| simplemodel    | v6simplemodel | part of __init__; node or server? |
| train   | v6train      |   - node or server? |
| coordination | v6coordination     |    node or server? need help; reuse, where does it fit into vantage6 as the infrastructure design differs (docker, yaml, client, etc.) |
| data | db | - at node |
| parser | - | can I use the same? |
| main | client?, setup? | ???|
| - | dockerfile, entitites, node_configuration, v6server | where does that fit in? |

Steps as in example: 

1. Mathematically decompose the model (model already there)
2. Implement and test locally (http://localhost I suppse)
3. Vantage6 algorithm wrapper 
4. Dockerize and push to a registry
   
central part of the algorithm needs to be able to create tasks (client.create_new_task). These subtasks are responsible to execute the federated part of the algorithm. node provides the algorithm with a JWT token so that the central part of the algorithm has access to the server to post these subtasks.)


Package structure:

```python
project_folder
├── Dockerfile
├── setup.py
└── algorithm_pkg
    └── __init__.py
```

in this case: 

```python
vantage6-v2.0
├── Dockerfile
├── setup.py
└── algorithm_pkg
    └── __init__.py
    └── client.py
    └── db.py
    └── v6coordination.py
    └── v6simplemodel.py
    └── v6train.py
```

Here, __init__.py is the algorithm which will import v6simplemodel.py, v6train.py (and v6coordination.py?)

-------- 

### Useful Links

not sure if this is required as RESTful API is an alternative:
https://blog.jetbrains.com/pycharm/2017/12/building-an-api-using-flask-restful-and-using-the-pycharm-http-client/ \
https://medium.com/@daniel.carlier/how-to-build-a-simple-flask-restful-api-with-docker-compose-2d849d738137 \
https://www.jetbrains.com/pycharm/guide/tutorials/intro-aws/crud/ \
https://en.wikipedia.org/wiki/Remote_procedure_call \

Docker: 
https://www.jetbrains.com/help/idea/run-debug-configuration-docker.html#docker_image_run_config \
https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html
