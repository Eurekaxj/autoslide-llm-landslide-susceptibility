"""Execute the actual PSLSA training method with a documented, minimal protocol patch.

The vendored PSLSA source and LICENSE are retained. The method body is extracted
without importing the GUI and its unrelated GDAL/R dependencies. The controlled
variant changes only the input/split/preprocessing/search adapter. It is not
represented as an unmodified PSLSA GUI run.
"""
import ast
import difflib
import hashlib
import textwrap
from pathlib import Path
from types import SimpleNamespace, MethodType

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import exp_lib as L


class FactorEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, factors):
        self.factors = factors

    def fit(self, X, y=None):
        _, self.encoder_ = L.build_matrix(X, list(self.factors), fit=True)
        return self

    def transform(self, X):
        z, _ = L.build_matrix(X, list(self.factors), enc=self.encoder_, fit=False)
        return z


def sources():
    p = Path(L.configured_path('pslsa_source', required=True))
    source = p.read_text(encoding='utf-8-sig')
    methods = [n for n in ast.walk(ast.parse(source))
               if isinstance(n, ast.FunctionDef) and n.name == 'model_train']
    if len(methods) != 1:
        raise ValueError('Expected exactly one original model_train method')
    node = methods[0]
    original = textwrap.dedent('\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])) + '\n'
    replacements = {
        'ss_x = StandardScaler()': '# Preprocessing is fitted within each training split by Pipeline.',
        'x1 = ss_x.fit_transform(x)': 'x1 = x',
        'x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.3, random_state=13)':
            'x_train, y_train = x1, y  # outer split is fixed by the shared protocol',
        'estimator = GridSearchCV(model, param_grid=para_dict, cv=5)':
            "model = Pipeline([('factors', FactorEncoder(self.factors)), ('model', model)])\n"
            "    para_dict = {'model__' + k: v for k, v in para_dict.items()}\n"
            "    estimator = GridSearchCV(model, param_grid=para_dict, cv=self.inner_splits,\n"
            "                             scoring='roc_auc', n_jobs=1, refit=True,\n"
            "                             return_train_score=False, error_score='raise')",
    }
    controlled = original
    for old, new in replacements.items():
        if controlled.count(old) != 1:
            raise ValueError(f'Original source changed; cannot apply patch: {old}')
        controlled = controlled.replace(old, new)
    return p, original, controlled


def export_adapter():
    p, original, controlled = sources()
    out = Path(L.EXP)/'vendor/PSLSA_extracted'; out.mkdir(parents=True, exist_ok=True)
    (out/'model_train_original.py').write_text(original, encoding='utf-8')
    (out/'model_train_controlled.py').write_text(controlled, encoding='utf-8')
    diff = ''.join(difflib.unified_diff(original.splitlines(True), controlled.splitlines(True),
                                      fromfile='PSLSA:model_train:original',
                                      tofile='PSLSA:model_train:controlled'))
    (out/'protocol_adapter.diff').write_text(diff, encoding='utf-8')
    return {'source_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'original_method_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'controlled_method_sha256': hashlib.sha256(controlled.encode()).hexdigest(),
            'diff': str((out/'protocol_adapter.diff').relative_to(L.EXP))}


def training_method(controlled=False, factors=(), inner_splits=None):
    _, original, patched = sources()
    namespace = {'StandardScaler': StandardScaler, 'train_test_split': train_test_split,
                 'GridSearchCV': GridSearchCV, 'Pipeline': Pipeline, 'FactorEncoder': FactorEncoder}
    exec(compile(patched if controlled else original, 'PSLSA_source_method', 'exec'), namespace)
    instance = SimpleNamespace(factors=factors, inner_splits=inner_splits)
    return MethodType(namespace['model_train'], instance)
