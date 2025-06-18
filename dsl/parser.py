from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lark import Lark, Transformer, v_args


dsl_grammar = r"""
?start: train_stmt

train_stmt: "TRAIN" "MODEL" NAME "USING" algorithm "FROM" NAME "PREDICT" NAME features option*

algorithm: NAME ("(" param_list? ")")?
param_list: param ("," param)*
param: NAME "=" value
value: SIGNED_NUMBER | ESCAPED_STRING | NAME

features: "WITH" "FEATURES" "(" feature_list? ")"
feature_list: NAME ("," NAME)*

option: validate_stmt
      | optimize_stmt
      | stop_stmt
      | split_stmt

validate_stmt: "VALIDATE" ("USING" NAME ("(" param_list? ")")? | "ON" NAME)
optimize_stmt: "OPTIMIZE" "FOR" NAME
stop_stmt: "STOP" "WHEN" condition_expr
split_stmt: "SPLIT" "DATA" split_entries

split_entries: split_entry ("," split_entry)*
split_entry: NAME "=" SIGNED_NUMBER

?condition_expr: or_expr
?or_expr: and_expr ("OR" and_expr)*
?and_expr: comparison ("AND" comparison)*
comparison: NAME COMP_OP value
COMP_OP: ">=" | "<=" | ">" | "<" | "!=" | "="

%import common.CNAME -> NAME
%import common.SIGNED_NUMBER
%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"""


@dataclass
class DataSplit:
    ratios: Dict[str, float]


@dataclass
class ValidationOption:
    method: Optional[str] = None
    params: Optional[List[tuple[str, Any]]] = None
    on: Optional[str] = None


@dataclass
class OptimizeOption:
    metric: str

@dataclass
class TrainModel:
    name: str
    algorithm: str
    params: List[tuple[str, Any]]
    source: str
    target: str
    features: List[str]
    split: Optional[DataSplit] = None
    validate: Optional[ValidationOption] = None
    optimize_metric: Optional[str] = None
    stop_condition: Optional[str] = None


class TreeToModel(Transformer):
    def NAME(self, token):
        return str(token)

    def SIGNED_NUMBER(self, token):
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

    def option(self, items):
        return items[0]

    def split_entry(self, items):
        name, value = items
        return name, value

    def split_entries(self, items):
        return dict(items)

    def split_stmt(self, items):
        return DataSplit(items[0])

    def validate_stmt(self, items):
        if len(items) == 1:
            return ValidationOption(on=items[0])
        else:
            method = items[0]
            params = items[1] if len(items) > 1 else None
            return ValidationOption(method=method, params=params)

    def optimize_stmt(self, items):
        return OptimizeOption(metric=items[0])

    def comparison(self, items):
        left, op, right = items
        return f"{left} {op} {right}"

    def and_expr(self, items):
        expr = items[0]
        for part in items[1:]:
            expr += f" AND {part}"
        return expr

    def or_expr(self, items):
        expr = items[0]
        for part in items[1:]:
            expr += f" OR {part}"
        return expr

    def stop_stmt(self, items):
        return items[0]

    @v_args(inline=True)
    def train_stmt(self, model_name, algorithm, source, target, features, *options):
        alg_name, params = algorithm
        model = TrainModel(
            name=model_name,
            algorithm=alg_name,
            params=params,
            source=source,
            target=target,
            features=features,
        )
        for opt in options:
            if isinstance(opt, DataSplit):
                model.split = opt
            elif isinstance(opt, ValidationOption):
                model.validate = opt
            elif isinstance(opt, OptimizeOption):
                model.optimize_metric = opt.metric
            elif isinstance(opt, str):
                model.stop_condition = opt
        return model


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
    args = [
        f"model_name := {repr(model.name)}",
        f"algorithm := {repr(model.algorithm)}",
        f"algorithm_params := {repr(params_json)}",
        f"training_data := {repr(training_query)}",
        f"target_column := {repr(model.target)}",
        f"feature_columns := ARRAY[{feature_array}]",
    ]
    if model.split:
        args.append(f"data_split := {repr(json.dumps(model.split.ratios))}")
    if model.validate:
        if model.validate.on:
            args.append(f"validate_on := {repr(model.validate.on)}")
        if model.validate.method:
            args.append(f"validate_method := {repr(model.validate.method)}")
            if model.validate.params:
                args.append(
                    f"validate_params := {repr(json.dumps(dict(model.validate.params)))}"
                )
    if model.optimize_metric:
        args.append(f"optimize_metric := {repr(model.optimize_metric)}")
    if model.stop_condition:
        args.append(f"stop_condition := {repr(model.stop_condition)}")

    sql = "SELECT ml_train_model(" + ", ".join(args) + ")"
    return sql
