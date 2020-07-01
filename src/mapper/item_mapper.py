import os

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.matcher.edit_based_matcher import EditMatcher
from src.matcher.token_based_matcher import TokenMatcher


class ItemMapper:
    def __init__(self):
        self.root_path = os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
        self.feature_cols = ['line_item_name', 'line_item_description', 'canonical_vendor_name']
        self.target_col = ['canonical_line_item_name']
        self.edit_matcher = EditMatcher()
        self.token_matcher = TokenMatcher()
        self.config = {
            "token": ["jac_score", "cos_score", "monge_elkan_score"],
            "edit": ["jaro_winkler_score", "lev_score"],
        }
        self.features_train = pd.read_csv(os.path.join(self.root_path, 'data', 'X_train.csv')).to_records(index=False)
        self.canonical_table = pd.read_csv(os.path.join(self.root_path, 'data', 'canon_table.csv'))['canonical_line_item_name'].tolist()
        self.y_train = pd.read_csv(os.path.join(self.root_path, 'data', 'y_train.csv'))['canonical_line_item_name'].tolist()
        self.features_eval = pd.read_csv(os.path.join(self.root_path, 'data', 'X_eval.csv')).to_records(index=False)
        self.X_train_ml = None
        self.y_train_ml = None
        self.X_eval_ml = None
        self.model = None

    @staticmethod
    def _check_input(s1, s2):
        if not isinstance(s1, str):
            s1 = ''
        if not isinstance(s2, str):
            s2 = ''
        return s1.lower(), s2.lower()

    @staticmethod
    def _tokenize(s):
        if isinstance(s, str):
            return s.split(' ')
        else:
            return ''

    def _generate_scores(self, feature, target_label):
        str_pair = self._check_input(feature, target_label)
        token_pair = [self._tokenize(s) for s in str_pair]
        scores = {
            'string pair': '-'.join(str_pair),
            'scores': {}
        }

        for m in self.config['edit']:
            method = getattr(self.edit_matcher, m)
            scores['scores'][m] = method(str_pair)
        for m in self.config['token']:
            method = getattr(self.token_matcher, m)
            scores['scores'][m] = method(token_pair)
        return scores

    @staticmethod
    def _calc_final_score(scores):
        final_score = sum([v for k, v in scores['scores'].items()])
        return final_score

    def _get_max_comp_score(self, feature, column):
        scores = []
        for element in self.canonical_table:
            scores.append((element, self._calc_final_score(self._generate_scores(feature, element))))
        return {
            'column': column,
            'max score': max(scores, key=lambda x: x[1])
        }

    def _get_mapping_label(self, record):
        info = []
        for val, col in zip(record, self.feature_cols):
            info.append(self._get_max_comp_score(val, col))
        return max(info, key=lambda x: x['max score'][1])['max score'][0]

    def train_rule_based(self, save=False):
        train_output = []
        for record in self.features_train:
            print(record)
            train_output.append(self._get_mapping_label(record))

        df_res = pd.DataFrame({'true_label': self.y_train, 'predict_label': train_output})
        if save:
            df_res.to_csv(os.path.join(self.root_path, 'results', 'train_results3.csv'), index=False)
        return df_res

    def eval_rule_based(self, save=False):
        eval_output = []
        for record in self.features_eval[:10]:
            print(record)
            eval_output.append(self._get_mapping_label(record))
        df_res = pd.DataFrame({'predict_label': eval_output})
        if save:
            df_res.to_csv(os.path.join(self.root_path, 'results', 'eval_results3.csv'), index=False)
        return df_res

    def transform_to_ml(self, mode='train'):
        ml_transformed = []
        if mode == 'train':
            for record, label in zip(self.features_train, self.y_train):
                for e in self.canonical_table:
                    class_label = 1 if e == label else 0
                    ml_transformed.append(list(record) + [e] + [label] + [class_label])

            self.X_train_ml = pd.DataFrame([i[:-2] for i in ml_transformed], columns=self.feature_cols + ['possible_canonical_name'])
            self.y_train_ml = pd.DataFrame({'Label': [i[-1] for i in ml_transformed]})
            return self.X_train_ml, self.y_train_ml
        elif mode == 'eval':
            for record in self.features_eval:
                for e in self.canonical_table:
                    ml_transformed.append(list(record) + [e])

            self.X_eval_ml = pd.DataFrame(ml_transformed, columns=self.feature_cols + ['possible_canonical_name'])
            return self.X_eval_ml

    def _generate_features(self):
        features = np.array([])
        count = 0
        for col1, col2, col3, col4 in self.X_train_ml.to_records(index=False):
            if count % 10000 == 0:
                print(count)
            count += 1
            features_one = np.array([])
            for col in [col1, col2, col3]:
                features_one = np.concatenate([features_one, np.array(list(self._generate_scores(col, col4)['scores'].values()))])
            features = np.vstack([features, features_one]) if features.size else features_one
        return features

    def train_ml(self):
        self.transform_to_ml()
        feature_matrix = self._generate_features()
        clf = RandomForestClassifier()
        clf.fit(feature_matrix, self.y_train_ml)
        self.model = clf
        return self.model

    def eval_ml(self):
        if not self.model:
            raise ValueError('Please train the model first!')
        self.transform_to_ml(mode='eval')
        pred = self.model.predict(self.X_eval_ml)
        return pred


if __name__ == '__main__':
    mapper = ItemMapper()

    mapper.train_rule_based()
    pred = mapper.eval_rule_based()

