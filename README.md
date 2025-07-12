# <img src="docs/imgs/logo-green.png" alt="icon" height="24" style="vertical-align:sub;"/> PFL Framework for Non-IID Data: Balancing Global and Local Adaptation

We propose

## Contents

- [ PFL Framework for Non-IID Data: Balancing Global and Local Adaptation](#-pfl-framework-for-non-iid-data-balancing-global-and-local-adaptation)
  - [Contents](#contents)
    - [Getting Started](#getting-started)
          - [Requirements](#requirements)
          - [Installation](#installation)
    - [Deployment](#deployment)
    - [Extend new algorithms and datasets](#extend-new-algorithms-and-datasets)
    - [Frameworks Used](#frameworks-used)
    - [Author](#author)

### Getting Started

###### Requirements

1. Ubuntu 24.04.02 LTS

###### Installation

1. Upgrade package

```sh
sudo apt-get update
```

2. Install Conda (If you have already installed this command or Anaconda , you can skip this step!!!!)

```sh
chmod 777 Install_miniconda.sh
./Install_miniconda.sh
```

### Deployment

1. Create a virtual environment and install the Python libraries

```sh
conda env create -f env.yaml
conda activate PFL
```

2. Generate the dataset based on the data distribution you personally want to test, for example FashionMNIST

**Pathological non-IID**: In this case, each client only holds a subset of the labels, for example, just 2 out of 10 labels from the FashionMNIST dataset, even though the overall dataset contains all 10 labels. This leads to a highly skewed distribution of data across clients.

**Practical non-IID**:  
Clients still see samples from _all_ labels, but with realistic heterogeneity in how the data is distributed and generated. We simulate this using:
   - **Label distribution skew**  
     Clients share the same label set but with very different class frequencies. Typically implemented by sampling class–client proportions from a Dirichlet(\<α\>) distribution (smaller α ⇒ more skew).
   - **Quantity skew**  
     Clients have disparate dataset sizes (e.g. one client with 10 000 samples vs. another with only 500).

```sh
cd ./dataset
python generate_FashionMNIST.py noniid - pat # for pathological noniid and unbalanced scenario
python generate_FashionMNIST.py noniid - dir # for practical noniid and unbalanced scenario
```

3. Run evaluation

```sh
cd ./system
python main.py -data FashionMNIST -m CNN -algo FedAvg -gr 100 -did 0 # using the FashionMNIST dataset, the FedAvg algorithm, and the 4-layer CNN model, communication round 100

python main.py -data Cifar10 -m CNN -algo FedAvg -gr 100 -did 0

python main.py -data Cifar100 -ncl 100 -m CNN -algo FedAvg -gr 100 -did 

python main.py -data TinyImagenet -ncl 200 -m CNN -algo FedAvg -gr 100 -did 0

python main.py -data FashionMNIST -m CNN -algo FedAvg -gr 100 -did 0,1,2,3 # running on multiple GPUs
```

### Extend new algorithms and datasets

- **New Dataset**: To add a new dataset, simply create a `generate_DATA.py` file in `./dataset` and then write the download code and use the [utils](https://github.com/TsingZ0/PFLlib/tree/master/dataset/utils) as shown in `./dataset/generate_MNIST.py` (you can consider it as a template):
  ```python
  # `generate_DATA.py`
  import necessary pkgs
  from utils import necessary processing funcs

  def generate_dataset(...):
    # download dataset as usual
    # pre-process dataset as usual
    X, y, statistic = separate_data((dataset_content, dataset_label), ...)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, statistic, ...)

  # call the generate_dataset func
  ```
  
- **New Algorithm**: To add a new algorithm, extend the base classes **Server** and **Client**, which are defined in `./system/flcore/servers/serverbase.py` and `./system/flcore/clients/clientbase.py`, respectively.
  - Server
    ```python
    # serverNAME.py
    import necessary pkgs
    from flcore.clients.clientNAME import clientNAME
    from flcore.servers.serverbase import Server

    class NAME(Server):
        def __init__(self, args, times):
            super().__init__(args, times)

            # select slow clients
            self.set_slow_clients()
            self.set_clients(clientAVG)
        def train(self):
            # server scheduling code of your algorithm
    ```
  - Client
    ```python
    # clientNAME.py
    import necessary pkgs
    from flcore.clients.clientbase import Client

    class clientNAME(Client):
        def __init__(self, args, id, train_samples, test_samples, **kwargs):
            super().__init__(args, id, train_samples, test_samples, **kwargs)
            # add specific initialization
        
        def train(self):
            # client training code of your algorithm
    ```
  
- **New Model**: To add a new model, simply include it in `./system/flcore/trainmodel/models.py`.
  
- **New Optimizer**: If you need a new optimizer for training, add it to `./system/flcore/optimizers/fedoptimizer.py`.

### Frameworks Used

- [PyTorch](https://pytorch.org/)

### Author

611221201@gms.ndhu.edu.tw

Egor Alekseyevich Morozov