# Load the dataset as Numpy array
from examples.dataset import load_dataset
x_train, x_test, y_train, y_test = load_dataset('dataset/1200Tex')

# Build a CNN model with a feature extraction function
from examples.extractor import EfficientNet
cnn = EfficientNet(device='cuda')

# Build the set of local features and estimate the Gaussian Mixture
from sklearn.decomposition import PCA
from src.model import MultiFisherEncoder

fv = MultiFisherEncoder(extractor = cnn.forward, reducer = PCA, n_kernels = 16)
fv.fit(x_train)

# Fit a classifier, in this case SVM, with the computed Fisher Vectors and evalute it using accuracy and macro-averaged F1-score measures
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

classifier = LinearSVC(dual='auto')
classifier.fit( fv.transform(x_train), y_train )
y_pred = classifier.predict( fv.transform(x_test) )
print("Accuracy: {:.4f}, F1-Score: {:.4f}".format(
    accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro')))
