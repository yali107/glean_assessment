import os

import pandas as pd

DATA_PATH = r'C:\LiYuan\Job support\assessments\glean\data'


def data_prep():
    raw_train = pd.read_excel(
        os.path.join(DATA_PATH, 'question-python-data-science-project-mwsr7tgbeo-mapping_challenge.xlsx'),
        sheet_name='train'
    )

    raw_eval = pd.read_excel(
        os.path.join(DATA_PATH, 'question-python-data-science-project-mwsr7tgbeo-mapping_challenge.xlsx'),
        sheet_name='eval'
    )

    canon_item_table = pd.read_excel(
        os.path.join(DATA_PATH, 'question-python-data-science-project-mwsr7tgbeo-mapping_challenge.xlsx'),
        sheet_name='canonical_line_item_table'
    )

    X_train = raw_train[['line_item_name', 'line_item_description', 'canonical_vendor_name']]
    y_train = raw_train[['canonical_line_item_name']]

    X_eval = raw_eval[['line_item_name', 'line_item_description', 'canonical_vendor_name']]

    return X_train, y_train, X_eval, canon_item_table


if __name__ == '__main__':
    x_tr, y_tr, x_ev, table = data_prep()
    x_tr.to_csv(r'../data/X_train.csv', index=False)
    y_tr.to_csv(r'../data/y_train.csv', index=False)
    x_ev.to_csv(r'../data/X_eval.csv', index=False)
    table.to_csv(r'../data/canon_table.csv', index=False)
