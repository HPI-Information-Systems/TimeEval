# TimeEval

Evaluation Tool for Anomaly Detection Algorithms on Time Series

## Installation (from source)

Currently, we only support installing `timeeval` from source.

**tl;dr**

```bash
git clone git@gitlab.hpi.de:bp2020fn1/timeeval.git
cd timeeval/
conda env create --file environment.yml
conda activate timeeval
python setup.py install
```

### Prerequisites

The following tools are required to install `timeeval` from source:

- git
- conda (anaconda or miniconda)

### Steps

1. Clone this repository using git and change into its root directory.
2. Create a conda-environment and install all required dependencies.
   Use the file [`environment.yml`](./environment.yml) for this:
   `conda env create --file environment.yml`.
3. Activate the new environment and install `timeeval` using _setup.py_:
   `python setup.py install`.
4. If you want to make changes to `timeeval` or run the tests, you need to install the development dependencies from `requirements.dev`:
   `pip install -r requirements.dev`.

## Tests

Run tests in `./tests/` as follows

```bash
python setup.py test
```

or

```bash
pytest
```

## Usage

### Datasets

A single dataset should be provided in a/two (with labels) Numpy-readable text file. The labels must be in a separate text file. Hereby, the label file can either contain the actual labels for each point in the data file or only the line indices of the anomalies.

Moreover, a single file can be used. It should be a csv without header having the label column at the most right and all data columns at the left.

#### Example

Data file
```csv
12751.0
8767.0
7005.0
5257.0
4189.0
```

Labels file (actual labels)
```csv
1
0
0
0
0
```

or Labels file (line indices)
```csv
0
```

or Combined Dataset file
```csv
12751.0,1
8767.0,0
7005.0,0
5257.0,0
4189.0,0
```

#### Registering Dataset

To tell the TimeEval tool where it can find which dataset, a configuration file is needed that contains all required datasets organized by their identifier which is used later on.

An element in the config file can either contain two different file paths for data and labels files, or use one combined file.

Config file
```json
{
  "dataset_name": {
    "data": "dataset.ts",
    "labels": "labels.txt"
  },
  "other_dataset": {
    "dataset": "dataset2.csv"
  }
}
```

### Algorithms

Any algorithm that can be called with a numpy array as parameter and a numpy array as return value can be evaluated. However, so far only __unsupervised__ algorithms are supported.

#### Registering Algorithm

```python
from timeeval import TimeEval, Algorithm
from pathlib import Path
import numpy as np

def my_algorithm(data: np.ndarray) -> np.ndarray:
    return np.zeros_like(data)

datasets = ["webscope", "mba", "eeg"]
algorithms = [
    # Add algorithms to evaluate...
    Algorithm(
        name="MyAlgorithm",
        function=my_algorithm,
        data_as_file=False
    )
]

timeeval = TimeEval(datasets, algorithms, dataset_config=Path("dataset.json"))
timeeval.run()
print(timeeval.results)
```
