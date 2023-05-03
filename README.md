# Branch prediction using deep learning

This repository contains the source code for my FYP project, Branch prediction using deep learning. (Liu Dongxu)

This project explores the topic of branch prediction, a technique commonly used in modern computer processors to optimize instruction execution. The current state-of-the-art branch predictor, TAGE, struggles to identify correlated branches deep within a noisy global branch history, and fundamental breakthroughs in branch prediction have become increasingly rare. To address this issue and further improve branch prediction, this report suggests relaxing the constraint of runtime-only training and adopting more sophisticated prediction mechanisms that incorporate deep learning. Building on the work of Building on the work of BranchNethttps://github.com/siavashzk/BranchNet, a convolutional neural networks (CNNs) with a practical on-chip inference engine tailored to the needs of branch prediction, the report proposes a better CNN model that requires fewer resources. In an effort to reduce the expensive computational cost of multiplication operations, the report implements a novel deep learning model called Spiking Neural Networks (SNNs) for prediction, which only utilizes addition operations while maintaining higher accuracy than TAGE.

I'm Liu Dongxu, and if you have any question, feel free to mp me.
My email is: ldx123000@gmail.com

## Dependencies 

* Linux (I mainly use Ubuntu 22.04.2 LTS)

* Python packages, with the versions that I've used:

```
Package         Version
--------------- -------
h5py            3.8.0
matplotlib      3.7.1
numpy           1.24.2
PyYAML          6.0
torch           1.13.1
torchvision     0.14.1
```

* Intel Pin Tool for generating branch traces (I've tested with 3.23)

* cmake3 and make

## Repository Overview

### src/branchnet

This directory contains the source code of the original big branchnet Convolutional Neural Network model.

**dataset_loader.py**: reads datasets in a custom format from files and produces PyTorch Datasets.

**model.py**: defines a BranchNet PyTorch model.

**run.py**: the main script that connects everything together to do training and evaluation.

**original.yaml**: this subdirectory contains knobs for defining BranchNet models.

For running jobs:

```
python run.py
```

### src/My_deep_learning_model

This directory contains the source code of my Convolutional Neural Network model and Spiking Neural Network model.

**dataset_loader.py**: reads datasets in a custom format from files and produces PyTorch Datasets.

**model.py**: defines my CNN and SNN models.

**CNN_run.py**: the main script for my CNN model that connects everything together to do training and evaluation.

**SNN_run.py**: the main script for my SNN model that connects everything together to do training and evaluation.

**config.yaml**: this subdirectory contains knobs for defining my CNN and SNN models.

For running CNN jobs:

```
python CNN_run.py
```

For running SNN jobs:

```
python SNN_run.py
```

### src/tracer

This directory is a pintool for creating branch traces. You could invoke *make* to build it.

### src/tage

This directory contains the source code for runtime predictors: TAGE-SC-L and MTAGE-SC. You could use *cmake* to build them.

### src/lib

Common headers (defines branch trace format).

### environment_setup

This file contains the global paths that you need to define for using the helper scripts. I have committed example dummy version. Maybe all You need to do is to change *train_traces_dir* and *test_traces_dir* to your trace paths.

### SimpleExamples

Here is an example to use Pintool.

1. move this directory to pin/source/tools

```
mv ~/Branch-prediction-using-deep-learning/SimpleExamples/ ~/pin-3.23-98579-gb15ab7903-gcc-linux/source/tools/
```

2. cd to SimpleExamples

```
cd ~/pin-3.23-98579-gb15ab7903-gcc-linux/source/tools/SimpleExamples/
```

3. "make clean; make" to test it out
4. run pin directly to get the trace file

```
../../../pin -t obj-intel64/tracer.so -- /bin/ls
```

This will run /bin/ls (to use pin must specify the app’s full path) and create the trace file “tracer.out”.

5. compress the trace to give tracer.out.bz2

```
bzip2 tracer.out
```

6. To get the traces, we can just run

```
./src/build/tracer/tagescl64 tracer.out.bz2 try.out
```

​      or run this to get Tage results

```
./src/build/tage/tagescl64 tracer.out.bz2 try.out
```

### 

### bin

This directory contains helper scripts for launching experiments. 

## How to run experiments?

### Create branch traces

First, build the tracer

```
./bin/build_tracer.py
```

In *./environment_setup/paths.yaml*, define *train_trace_dir* and *test_traces_dir* : relative path to a directory for storing branch traces for benchmarks.

You can get traces by pintool or I have provided some traces for you to use directly.

Run:

```
./bin/run_traces.py
```

This will create trace output in the *output_traces_dir* directory, with the following structure. 

```
traces/
traces/training
traces/test
```


### Evaluate runtime predictors on the traces

First, build the runtime predictors

```
./bin/build_tage.py
```

In *./environment_setup/paths.yaml*, define *tage_stats_dir*: relative path to a directory for storing the results of runtime predictors for each benchmark.

Open *./bin/run_tage.py* and edit TARGET_BENCHMARKS to include your benchmarks of interest, BINARY_NAME to any of the binary names produces by *build_tage.py*, CONFIG_NAME to any name you like (this will be used by other script to refer to the runtime predictor results), and set NUM_THREADS according to your system capacity.

Run:

```
./bin/run_tage.py
```

### Identify hard-to-predict branches 

In *./environment_setup/paths.yaml*, define *hard_br_dir*: relative path to a directory for storing the PCs of hard-to-predict branches for each benchmark and *hard_br_acc_dir*: relative path to a directory for storing the accuracy of hard-to-predict branches for each benchmark. 

Run:

```
./bin/get_hard_brs.py
```

Note that the hard_br files are simple text files, where each line is a branch PC in hexadecimal. You can modify these manually, too, without running the script, which is useful when targetting specific branches for short experiments.

### Create BranchNet datasets

We need to convert the branch traces to a format that is more suitable for training. The key idea is to find the occurances of the hard-to-predict branches and store their positions in the trace along with the trace. We store the traces in *hdf5* file format.

In *./environment_setup/paths.yaml*, define *dataset_dir*: relative path to a directory for storing the branch traces, ready for running ML jobs in *hdf5* file format.

Open *./bin/create_ml_datasets.py* and edit  PC_BITS to the number of least significant bits in the PC to keep, and set NUM_THREADS according to your system capacity.

Run:

```
./bin/create_ml_datasets.py
```

### Run BranchNet training and evaluation jobs

Finally we're ready to actually training and evaluate a neural network model!

In *./environment_setup/paths.yaml*, define *result_BranchNet*, *result_CNN* and *result_SNN*: relative path to a directory for results of the ML jobs.

Open *./src/branchnet/run.py* , *./src/My_deep_learning_model/CNN_run* and *./src/My_deep_learning_model/SNN_run* to run corresponding ML jobs.
```
./src/branchnet/run.py
```
```
./src/My_deep_learning_model/CNN_run
```
```
./src/My_deep_learning_model/SNN_run
```