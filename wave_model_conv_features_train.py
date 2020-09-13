from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import h5py
import pickle

NUM_TRAIN_IMAGES = 72
feature_out_dir = r'Features\wave_features.hdf5' #args['output']

db = h5py.File(feature_out_dir, "r")
print("[INFO] tuning hyperparams...")
params = {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]}
model = GridSearchCV(LogisticRegression(solver="lbfgs",
                                        multi_class="auto"),
                     params,
                     cv=3,
                     n_jobs=-1)
model.fit(db["features"][:NUM_TRAIN_IMAGES], db["labels"][:NUM_TRAIN_IMAGES])
print("[INFO] best hyperparams: {}".format(model.best_params_))

print("[INFO] evaluating...")
preds = model.predict(db["features"][NUM_TRAIN_IMAGES:])
print(classification_report(db["labels"][NUM_TRAIN_IMAGES:], preds,
                            target_names=db["label_names"]))

acc = accuracy_score(db["labels"][NUM_TRAIN_IMAGES:], preds)
print("[INFO] score: {}".format(acc))

print("[INFO] saving model to disk...")
f = open(f'models\\lr_features_wave.pickle', "wb")
f.write(pickle.dumps(model))
f.close()
