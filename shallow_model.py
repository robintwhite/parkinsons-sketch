import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from pathlib import Path
from utils.process_images import *
from utils.process_data import *
from tabulate import tabulate
import matplotlib.pyplot as plt


def pdtabulate(df):

    return tabulate(df,headers='keys',tablefmt='psql')


data_dir = Path(r'D:\Docs\Python_code\ParkinsonsSketch\178338_401677_bundle_archive\drawings')

print('[INFO] loading data...')
df = pd.DataFrame({'path': list(data_dir.glob('*/*/*/*.png'))})
df['img_id'] = df['path'].map(lambda x: x.stem)
df['disease'] = df['path'].map(lambda x: x.parent.stem)
df['validation'] = df['path'].map(lambda x: x.parent.parent.stem)
df['activity'] = df['path'].map(lambda x: x.parent.parent.parent.stem)
print(f'[INFO] {df.shape[0]} images loaded.')

print('[INFO] processing images to create features...')
df['thresh_img'] = df['path'].map(lambda x: read_and_thresh(x, resize=False))
df['clean_img'] = df['thresh_img'].map(lambda x: closing(label_sort(x)>0, disk(1)))
df['thickness'] = df['clean_img'].map(lambda x: stroke_thickness(x))
df['mean_thickness'] = df['thickness'].apply(np.mean)
df['std_thickness'] = df['thickness'].apply(np.std)
df['num_pixels'] = df['clean_img'].map(lambda x: sum_pixels(skeleton_drawing(x)))
df['num_ep'] = df['clean_img'].map(lambda x: number_of_end_points(x, k_nn))
df['num_inters'] = df['clean_img'].map(lambda x: number_of_intersection_points(x, k_nn))
# draw_df['nn_img'] = draw_df['clean_img'].map(lambda x: get_cleaned_nn_and_label(x, k_nn)[0])
# draw_df['label_img'] = draw_df['clean_img'].map(lambda x: get_cleaned_nn_and_label(x, k_nn)[1])
print('done.')

#spiral and wave separately.
activities = ['wave', 'spiral']
for activity in activities:
    print(f"[INFO] creating dataset for {activity}...")
    draw_df = df.loc[df['activity'] == activity]

    feature_columns = ['mean_thickness',
                       'std_thickness',
                       'num_pixels',
                       'num_ep',
                       'num_inters']
    target_column = ['disease']

    train_df = draw_df.loc[draw_df['validation'] == 'training']
    train_df = shuffle(train_df, random_state=42)
    print(f"[INFO] training samples for {activity}: {len(train_df.index)}")
    test_df = draw_df.loc[draw_df['validation'] == 'testing']
    test_df = shuffle(test_df, random_state=42)
    print(f"[INFO] testing samples for {activity}: {len(test_df.index)}...")

    X_train, y_train = train_df[feature_columns], train_df[target_column].to_numpy().ravel()
    X_test, y_test = test_df[feature_columns], test_df[target_column].to_numpy().ravel()

    #add interaction terms for all i != j columns: xi*xj
    X_train = create_interactions(X_train)
    X_test = create_interactions(X_test)

    X_train, X_test = standardize(X_train, X_test, verbose=False)

    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train).ravel()
    #print(lb.classes_)
    y_test = lb.transform(y_test).ravel()

    print(pdtabulate(X_train.sample(random_state = 42)))

    # print("[INFO] tuning hyperparams for LR...")
    # params_lr = {"C": [0.001, 0.01, 1.0, 10.0]}
    # lr_cls = GridSearchCV(LogisticRegression(solver="lbfgs",
    #                                         multi_class="auto",
    #                                         random_state=42),
    #                      params_lr,
    #                      cv=5,
    #                      n_jobs=-1)
    print("[INFO] fitting LR...")
    lr_cls = LogisticRegression(solver="lbfgs",
                                multi_class="auto",
                                C=1.0,
                                random_state=42)
    lr_cls.fit(X_train, y_train)
    # print("[INFO] best hyperparams for LR: {}".format(lr_cls.best_params_))

    print("[INFO] fitting RF...")
    rf_cls = RandomForestClassifier(n_estimators=100,
                                    random_state=42)
    rf_cls.fit(X_train, y_train)
    important_features_list = feature_importance(rf_cls)
    print(f'[INFO] Feature impact (in order of importance): {X_train.columns[important_features_list].values}')

    print("[INFO] evaluating...")
    rf_preds = rf_cls.predict(X_test)
    rf_probs = rf_cls.predict_proba(X_test)
    rf_probs_max = np.array([max(x,y) for x,y in rf_probs]).ravel()
    lr_preds = lr_cls.predict(X_test)
    lr_probs = lr_cls.predict_proba(X_test)
    lr_probs_max = np.array([max(x, y) for x, y in lr_probs]).ravel()

    print(classification_report(y_test, rf_preds, target_names=lb.classes_))
    print(f'[INFO] RF accuracy: {rf_cls.score(X_test, y_test)}')
    #print(f'[INFO] RF AUC: {roc_auc_score(y_test, rf_probs[:,1].ravel())}')
    print(classification_report(y_test, lr_preds, target_names=lb.classes_))
    print(f'[INFO] LR accuracy: {lr_cls.score(X_test, y_test)}')
    #print(f'[INFO] LR AUC: {roc_auc_score(y_test, lr_probs[:,1].ravel())}')
    # for AUC, score of the class with the greatest label from documentation for binary case

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    rfc_disp = plot_roc_curve(rf_cls, X_test, y_test, ax=ax, alpha=0.8)
    lr_disp = plot_roc_curve(lr_cls, X_test, y_test, ax=ax, alpha=0.8)
    plt.show()

