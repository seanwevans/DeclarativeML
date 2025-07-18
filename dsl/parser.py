from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lark import Lark, Transformer, v_args

dsl_grammar = r"""
?start: train_stmt
       | compute_stmt

train_stmt: "TRAIN" "MODEL" NAME "USING" algorithm "FROM" NAME \
            "PREDICT" NAME features option*

compute_stmt: "COMPUTE" NAME compute_from? compute_into? compute_every? \
              "USING" NAME kernel_opt*

compute_from: "FROM" "table" "(" name_list ")"
compute_into: "INTO" "column" "(" NAME ")"
compute_every: "EVERY" SIGNED_NUMBER "TICKS"
kernel_opt: block_opt
          | grid_opt
          | shared_opt

block_opt: "BLOCK" SIGNED_NUMBER
grid_opt: "GRID" NAME
shared_opt: "SHARED" size_spec

name_list: NAME ("," NAME)*
size_spec: SIGNED_NUMBER NAME?

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
      | balance_stmt

balance_stmt: "BALANCE" "CLASSES" "BY" NAME

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
class BalanceOption:
    method: str


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
    balance_method: Optional[str] = None


@dataclass
class ComputeKernel:
    name: str
    inputs: Optional[List[str]] = None
    output: Optional[str] = None
    schedule_ticks: Optional[int] = None
    kernel: str = ""
    options: Dict[str, Any] | None = None


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

    def name_list(self, items):
        return list(items)

    def compute_from(self, items):
        return ("inputs", items[0])

    def compute_into(self, items):
        return ("output", items[0])

    def compute_every(self, items):
        return ("schedule", int(items[0]))

    def size_spec(self, items):
        if len(items) == 2:
            return f"{items[0]}{items[1]}"
        return str(items[0])

    def kernel_opt(self, items):
        # should not be called directly
        return items[0]

    def block_opt(self, items):
        return ("BLOCK", items[0])

    def grid_opt(self, items):
        return ("GRID", items[0])

    def shared_opt(self, items):
        return ("SHARED", items[0])

    def split_entry(self, items):
        name, value = items
        return name, value

    def split_entries(self, items):
        return dict(items)

    def split_stmt(self, items):
        return DataSplit(items[0])

    def balance_stmt(self, items):
        return BalanceOption(method=items[0])

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
    def train_stmt(
        self,
        model_name,
        algorithm,
        source,
        target,
        features,
        *options,
    ):
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
            elif isinstance(opt, BalanceOption):
                model.balance_method = opt.method
            elif isinstance(opt, str):
                model.stop_condition = opt
        return model

    @v_args(inline=True)
    def compute_stmt(
        self,
        name,
        *parts,
    ):
        inputs = None
        output = None
        schedule = None
        kernel_name = None
        options: Dict[str, Any] = {}

        for part in parts:
            if isinstance(part, tuple) and part[0] == "inputs":
                inputs = part[1]
            elif isinstance(part, tuple) and part[0] == "output":
                output = part[1]
            elif isinstance(part, tuple) and part[0] == "schedule":
                schedule = part[1]
            elif isinstance(part, str) and kernel_name is None:
                kernel_name = part
            elif isinstance(part, tuple):
                key, val = part
                options[key] = val
            else:
                # ignore unexpected parts
                pass
        if kernel_name is None:
            raise ValueError("Kernel name missing")
        return ComputeKernel(
            name=name,
            inputs=inputs,
            output=output,
            schedule_ticks=schedule,
            kernel=kernel_name,
            options=options or None,
        )


def parse(text: str) -> Any:
    parser = Lark(dsl_grammar, start="start", parser="lalr")
    tree = parser.parse(text)
    model = TreeToModel().transform(tree)
    return model


def compile_sql(model: Any) -> str:
    import json

    if isinstance(model, TrainModel):
        feature_cols = ", ".join(model.features)
        params_dict = {k: v for k, v in model.params}
        params_json = json.dumps(params_dict)
        training_query = (
            "SELECT " + f"{feature_cols}, {model.target} " + f"FROM {model.source}"
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
                    params_json = json.dumps(dict(model.validate.params))
                    args.append(f"validate_params := {repr(params_json)}")
        if model.optimize_metric:
            args.append(f"optimize_metric := {repr(model.optimize_metric)}")
        if model.stop_condition:
            args.append(f"stop_condition := {repr(model.stop_condition)}")
        if model.balance_method:
            args.append(f"balance_method := {repr(model.balance_method)}")

        sql = "SELECT ml_train_model(" + ", ".join(args) + ")"
        return sql

    if isinstance(model, ComputeKernel):
        args = [f"kernel_name := {repr(model.kernel)}", f"name := {repr(model.name)}"]
        if model.inputs:
            inputs_array = ", ".join(repr(i) for i in model.inputs)
            args.append(f"inputs := ARRAY[{inputs_array}]")
        if model.output:
            args.append(f"output := {repr(model.output)}")
        if model.schedule_ticks is not None:
            args.append(f"schedule_ticks := {model.schedule_ticks}")
        if model.options:
            args.append(f"options := {repr(json.dumps(model.options))}")
        sql = "SELECT ml_register_compute(" + ", ".join(args) + ")"
        return sql

    raise TypeError("Unsupported model type")
