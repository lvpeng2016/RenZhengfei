Run the code below in your IDE to get the MNIST dataset.

```python
# import packages
import os
import urllib.request

#create a fold for the dataset
os.makedirs('./data', exist_ok = True)
# load dataset to the directory, as you can see, you need to load train sets and test sets seperately 
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', filename='./data/train-images.gz')
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', filename='./data/train-labels.gz')
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', filename='./data/test-images.gz')
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', filename='./data/test-labels.gz')
```

Now run the code below to split the data into training and test sets.

```python
#import dataset to 

# To help the model to converge faster , you shrink the intensity values (X) from 0-255 to 0-1
X_train = load_data('./data/train-images.gz', False) / 255.0
y_train = load_data('./data/train-labels.gz', True).reshape(-1)

X_test = load_data('./data/test-images.gz', False) / 255.0
y_test = load_data('./data/test-labels.gz', True).reshape(-1)
```

Here is an example of the dataset. The numbers on top are the labels, and the handwritten pictures are in the second row.

![Screenshot of Hard-written picture Example](../media/2-hand-written.png)

#### Video: Parts of a Computer
> [!VIDEO https://www.youtube.com/embed/sUssQdoEYb4]

// one blank

> [!VIDEO https://www.youtube.com/embed/sUssQdoEYb4]

> [!VIDEO https://www.youtube.com/embed/ApKlrP1LqZw]

// two blanks

> [!VIDEO https://www.youtube.com/embed/ApKlrP1LqZw]

> [!VIDEO https://www.youtube.com/embed/ApKlrP1LqZw]

// many blanks

> [!VIDEO https://www.youtube.com/embed/ApKlrP1LqZw]

> [!VIDEO https://www.youtube.com/embed/ApKlrP1LqZw]

// pre blank

> [!VIDEO https://www.youtube.com/embed/ApKlrP1LqZw]

> [!VIDEO https://www.youtube.com/embed/ApKlrP1LqZw]

// many pre blanks

> [!VIDEO https://www.youtube.com/embed/ApKlrP1LqZw]

> [!VIDEO https://www.youtube.com/embed/ApKlrP1LqZw]

// many pre and latter blanks

> [!VIDEO https://www.youtube.com/embed/ApKlrP1LqZw]

edX_intro_final.mp4
**Watch the following video to learn about the hardware component