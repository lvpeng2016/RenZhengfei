You learned that a compute target is the compute resource used to run a training script or host service deployment. Here you create an Azure Machine Learning Compute (AmlCompute) as our compute resource. The first step is to create a remote compute target

```python
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
import os

# Step 1: name the cluster, set minimal and maximal number of nodes 
compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "cpucluster")
min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 3)

# Step 2: choose environment variables 
vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")


provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size,
                                                                min_nodes = min_nodes, 
                                                                max_nodes = max_nodes)

# create the cluster
compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)
```

## Load Data and Create the Modeling Script

Depending on the location and format of the data source, there are various ways to pipe data into the model. In this example, you upload the data to Azure portal using the code below.

```python
#upload data using get_default_datastore()
ds = ws.get_default_datastore()

ds.upload(src_dir='./data_mnist', target_path='mnist', overwrite=True, show_progress=True)
```

Then you create a directory to save our training Python code.

```python
#import os if you haven't done so
Import os
#create folder
folder _training_script= './trial_model_mnist'
os.makedirs(folder _training_script, exist_ok=True)
```

Finally, let's prepare our model training script. Note that in this script, you are defining three parameters. The first parameter is for finding the data stored on the cloud, or for the path. The other two parameters are used to define the parameter k in the kNN algorithm. 'kmax' limits the maximum value of k, and 'kinterval' decides the interval between each k.

```python
%%writefile $folder_training_script/train.py

import argparse
import os
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

from utils import load_data

# Create 3 parameters, the location of the data files, and maximun value of k and the interval
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--kmax', type=float, dest='kmax', default=15, help='max k value')
parser.add_argument('--kinterval', type=float, dest='kinterval', default=2, help='k interval')
args = parser.parse_args()


data_folder = os.path.join(args.data_folder, 'mnist')
print('Data folder:', data_folder)

# load train and test set into numpy arrays
X_train = load_data(os.path.join(data_folder, 'train-images.gz'), False) / 255.0
X_test = load_data(os.path.join(data_folder, 'test-images.gz'), False) / 255.0
#print variable set dimension
print(X_train.shape, X_test.shape,  sep = '\n')

y_train = load_data(os.path.join(data_folder, 'train-labels.gz'), True).reshape(-1)
y_test = load_data(os.path.join(data_folder, 'test-labels.gz'), True).reshape(-1)

#print response variable dimension
print( y_train.shape, y_test.shape, sep = '\n')

# get hold of the current run
run = run.get_context()

print('Train KNN models with k equals to', range(1,args.kmax,args.kinterval))


# generating a wide range of K and find the best models
# also create a list to store the evaluation result for each value of k
kVals = range(1,args.kmax,args.kinterval)
evaluation = []

# loop over models with different parameter to find the one with lowest error rate
for k in kVals:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # using test dataset for evaluation and append the result to the evaluation list
    score = model.score(X_test, y_test)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    evaluation.append(score)

# find the value of k with best performance
i = int(np.argmax(evaluation))
print("k=%d with best performance with %.2f%% accuracy given current testset" % (kVals[i], evaluation[i] * 100))

model = KNeighborsClassifier(n_neighbors=kVals[i])


run.log('Best_k', kVals[i])
run.log('accuracy', evaluation[i])

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=model, filename='outputs/knn_mnist_model.pkl')
```

Now, you must add a utils script as shown below for loading data and to create an estimator so that it's easier to scale our work in the future. An estimator object is used to submit the run. Create your estimator by running the following code to define these items:

- The name of the estimator object, est.
- The directory that contains your scripts. All the files in this directory are uploaded into the cluster nodes for execution.
- The compute target. In this case, you use the Azure Machine Learning compute cluster you created.
- The training script name, **train.py**.
- Parameters required from the training script.
- Python packages needed for training.

```python
import shutil
shutil.copy('utils.py', folder_training_script)

from azureml.train.estimator import Estimator

script_params = {
    '--data-folder': ds.as_mount(),
    '--kmax': 5,
    '--kinterval':2
}

#import scikit-learn package 
est = Estimator(source_directory=folder_training_script,
                script_params=script_params,
                compute_target=compute_target,
                entry_script='train.py',
                conda_packages=['scikit-learn'])

```

## Submit Model, Monitor Run and Retrieve Results

The last step is running the model. Log in with your Azure account if you're asked.

```python
#run
run = experiment.submit(config=est)
run
```

![Screenshot of Experiment Status](../media/7-experiment-status.png)

You could use widgets module from 'azureml' package to monitor our run.

```python
# monitor the run
from azureml.widgets import RunDetails
RunDetails(run).show()
```

This Screenshot shows the status when the remote resources are running.

![Screenshot of Job Running](../media/7-job-running.png)

This Screenshot shows the status of the job completed. Highlighted in the red box, you can see that you got the same results as you did earlier in the local machine.

![Screenshot of Job Running](../media/7-job-completion.png)

After this run finishes, you can print the results. The results were logged since you wrote the code in the training script.

```python
#get result
print(run.get_metrics())
```