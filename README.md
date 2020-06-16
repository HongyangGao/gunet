PyTorch Implementation of Graph U-Nets
======================================

Created by [Hongyang Gao](http://people.tamu.edu/~hongyang.gao/), and
[Shuiwang Ji](http://people.tamu.edu/~sji/) at Texas A&M University.

About
-----

PyTorch implementation of Graph U-Nets. Check http://proceedings.mlr.press/v97/gao19a/gao19a.pdf for more information.

Methods
-------

### Graph Pooling Layer

![gPool](./doc/GPool.png)

### Graph Unpooling Layer

![gPool](./doc/GUnpool.png)

### Graph U-Net

![gPool](./doc/GUnet.png)

Installation
------------


Type

    ./run_GNN.sh DATA FOLD GPU
to run on dataset using fold number (1-10). You can run

    ./run_GUNet.sh DD 0 0
to run on DD dataset with 10-fold cross
validation on GPU #0.


Code
----

The detail implementation of Graph U-Net is in src/utils/ops.py.


Datasets
--------

Check the "data/README.md" for the format. 


Results
-------


| Models   | DD              | IMDBMULTI       | PROTEINS        |
| -------- | --------------- | --------------- | --------------- |
| PSCN     | 76.3 ± 2.6%     | 45.2 ± 2.8%     | 75.9 ± 2.8%     |
| DIFFPOOL | 80.6%           | -               | 76.3%           |
| SAGPool  | 76.5%           | -               | 71.9%           |
| GIN      | 82.0 ± 2.7%     | 52.3 ± 2.8%     | 76.2 ± 2.8%     |
| g-U-Net  | **83.0 ± 2.2%** | **56.7 ± 2.9%** | **78.7 ± 4.2%** |

Reference
---------

If you find the code useful, please cite our paper:

    @inproceedings{gao2019graph,
        title={Graph U-Nets},
        author={Gao, Hongyang and Ji, Shuiwang},
        booktitle={International Conference on Machine Learning},
        pages={2083--2092},
        year={2019}
    }
