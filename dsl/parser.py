from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from lark import Lark, Transformer, v_args


dsl_grammar = r"""
?start: train_stmt

train_stmt: "TRAIN" "MODEL" NAME "USING" algorithm "FROM" NAME "PREDICT" NAME features  # noqa: E501

algorithm: NAME ("(" param_list? ")")?
param_list: param ("," param)*
param: NAME "=" value
value: NUMBER | ESCAPED_STRING | NAME

features: "WITH" "FEATURES" "(" feature_list? ")"
feature_list: NAME ("," NAME)*

%import common.CNAME -> NAME
%import common.NUMBER
%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"""


@dataclass
class TrainModel:
    name: str
    algorithm: str
    params: List[tuple[str, Any]]
    source: str
    target: str
    features: List[str]


class TreeToModel(Transformer):
    def NAME(self, token):
        return str(token)

    def NUMBER(self, token):
        text = token.value
        return float(text) if "." in text else int(text)

    def ESCAPED_STRING(self, token):
        return token.value.strip('"')

    def value(self, items):
        return items[0]

    def param(self, items):
        name, value = items
        return name, value

    def param_list(self, items):
        return items

    def algorithm(self, items):
        alg_name = items[0]
        if len(items) == 1 or items[1] is None:
            params = []
        else:
            params = items[1]
        return alg_name, params

    def feature_list(self, items):
        return list(items)

    def features(self, items):
        return items[0] if items else []

    @v_args(inline=True)
    def train_stmt(self, model_name, algorithm, source, target, features):
        alg_name, params = algorithm
        return TrainModel(
            name=model_name,
            algorithm=alg_name,
            params=params,
            source=source,
            target=target,
            features=features,
        )


def parse(text: str) -> TrainModel:
    parser = Lark(dsl_grammar, start="start", parser="lalr")
    tree = parser.parse(text)
    model = TreeToModel().transform(tree)
    return model


def compile_sql(model: TrainModel) -> str:
    import json

    feature_cols = ", ".join(model.features)
    params_dict = {k: v for k, v in model.params}
    params_json = json.dumps(params_dict)
    training_query = (
        f"SELECT {feature_cols}, {model.target} FROM {model.source}"
    )
    feature_array = ", ".join(repr(f) for f in model.features)
    sql = (
        "SELECT ml_train_model("
        f"model_name := {repr(model.name)}, "
        f"algorithm := {repr(model.algorithm)}, "
        f"algorithm_params := {repr(params_json)}, "
        f"training_data := {repr(training_query)}, "
        f"target_column := {repr(model.target)}, "
        f"feature_columns := ARRAY[{feature_array}]"
        ")"
    )
    return sql
