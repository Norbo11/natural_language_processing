# inspired of https://realpython.com/python-keras-text-classification/

from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences


from sklearn.model_selection import RandomizedSearchCV



def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Importing the dataset
df = pd.read_csv('offenseval-training-v1.tsv',sep='\t')
print(df.shape)


sentences = df['tweet'].values
y = df['subtask_a'].values


# Main settings
epochs = 50
embedding_dim = 50
maxlen = 100
output_file = 'data/output.txt'

# Run grid search for each source (yelp, amazon, imdb)
# Train-test split
sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
y_test = lb.fit_transform(y_test)
y_train = lb.fit_transform(y_train)

print(y_train)
print(y_test)

# Tokenize words
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

print(X_train)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

# Pad sequences with zeros
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

from keras.callbacks import ModelCheckpoint, Callback

# Parameter grid for grid search
param_grid = dict(num_filters=[32, 64, 128],
                  kernel_size=[3, 5, 7],
                  vocab_size=[vocab_size],
                  embedding_dim=[embedding_dim],
                  maxlen=[maxlen])
model = KerasClassifier(build_fn=create_model,
                        epochs=epochs, batch_size=10,
                        verbose=True)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                          cv=4, verbose=1, n_iter=1)

print(X_train.shape)
print(y_train.shape)


params = dict(
    num_filters=32,
    kernel_size=3,
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    maxlen=maxlen
)

#grid_result = grid.fit(X_train, y_train, validation_data=(X_test, y_test))
#grid_result = grid.fit(X_train, y_train)

model = create_model(**params)
grid_result = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)
test_accuracy = model.evaluate(X_test, y_test)


# Evaluate testing set
#test_accuracy = grid.score(X_test, y_test)

# Save and evaluate results
for i, metric in enumerate(test_accuracy):
    print(f'{model.metrics_names[i]}: {metric}')

#s = ('Running data set\nBest Accuracy : {:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')

# output_string = s.format(
#     grid_result.best_score_,
#     grid_result.best_params_,
    # test_accuracy)
