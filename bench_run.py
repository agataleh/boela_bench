import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from boela import problems
from boela.constants import METRICS
from boela.log import set_logger
from boela.model import analyze_sample_features
from boela.optimizer import solve_problem
from boela.predictor import build_predictor, load_predictor
from boela.problems.problem import ProblemBase
from boela.timer import timers

LOG = set_logger("bench_run")
DATA_FOLDER = "data"
MODELS_FOLDER = "models"
SAMPLES_FOLDER = "samples"

data_folder = Path(DATA_FOLDER)
data_folder.mkdir(parents=True, exist_ok=True)
models_folder = Path(MODELS_FOLDER)
models_folder.mkdir(parents=True, exist_ok=True)
samples_folder = Path(SAMPLES_FOLDER)
samples_folder.mkdir(parents=True, exist_ok=True)

timer = timers.start("Solve problems using metrics")
metrics_by_problem: Dict[str, Dict[METRICS, float]] = {}
for dim, n_init, n_restarts in [(2, 10, 10)]:  # settings
    for i, problem_class in enumerate(problems.bbob._all):
        for metric in [
            METRICS.NONE,
            METRICS.R2_CV,
            METRICS.SUIT_EXT,
            METRICS.VM_ANGLES_EXT,
        ]:
            problem: ProblemBase = problem_class.Problem(dim)
            tag = f"{problem.id} {n_init=} {n_restarts=} metric={metric}"
            LOG.info(f"{i}:{tag}")
            data_file = data_folder / f"{tag}.csv"
            if data_file.exists():
                data = pd.read_csv(data_file, index_col=0)
            else:
                data: pd.DataFrame = solve_problem(
                    problem=problem,
                    n_init=n_init,
                    n_iter=2 * n_init,
                    n_seeds=n_restarts,
                    metric=metric,
                    predictor=None,
                )
                data.to_csv(data_file)
            metrics_by_problem.setdefault(problem.id, {})
            metrics_by_problem[problem.id][metric] = data["f_best"].mean()

with (models_folder / "metrics.json").open("w") as fi:
    json.dump(metrics_by_problem, fi, indent=4)

LOG.info(f"Stage 1 {timer.stop().str()}\n{json.dumps(timers.str(), indent=2)}\n")
########################################################################################


timer = timers.start("Explore sample features")
metrics_data: Dict[METRICS, List[pd.DataFrame]] = {}
for seed in range(15):
    for dim, n_init in [(2, 10), ]:  # settings
        n_iter = 2 * n_init
        sizes = np.linspace(n_init, n_init + n_iter, 20).astype(int)
        for metric in [
            METRICS.R2_CV,
            METRICS.SUIT_EXT,
            METRICS.VM_ANGLES_EXT,
        ]:
            LOG.info(f"{dim=} {seed=} {metric}")
            path_metric_current = samples_folder / f"{metric}" / f"{dim=} {seed=}.csv"
            path_metric_current.parent.mkdir(parents=True, exist_ok=True)
            if path_metric_current.is_file():
                data = pd.read_csv(path_metric_current, index_col=0)
            else:
                data = analyze_sample_features(
                    problem_modules=problems.bbob._all,
                    dim=dim,
                    sizes=sizes,
                    metric=metric,
                    seed=seed,
                )
                data.to_csv(path_metric_current)
            metrics_data.setdefault(metric, []).append(data)
    if seed % 5 == 0:
        path_metric = samples_folder / f"{metric}.csv"
        data_metric = pd.concat(metrics_data[metric], ignore_index=True)
        data_metric.to_csv(path_metric)
for metric in [
    METRICS.R2_CV,
    METRICS.SUIT_EXT,
    METRICS.VM_ANGLES_EXT,
]:
    path_metric = samples_folder / f"{metric}.csv"
    data_metric = pd.concat(metrics_data[metric], ignore_index=True)
    data_metric.to_csv(path_metric)
LOG.info(f"Stage 2 {timer.stop().str()}\n{json.dumps(timers.str(), indent=2)}\n")
########################################################################################


timer = timers.start("Build predicting models")
predictor_sample_path = models_folder / f"predictor_sample.csv"
if predictor_sample_path.is_file():
    predictor_sample = pd.read_csv(predictor_sample_path, index_col=0)
else:
    best_metric_samples: List[pd.DataFrame] = []
    for dim in [2, ]:  # settings
        for i, problem_module in enumerate(problems.bbob._all):
            problem = problem_module.Problem(dim)
            metrics_values = metrics_by_problem[problem.id]
            metrics_values[METRICS.NONE] = np.inf
            metrics_values[METRICS.PREDICTED] = np.inf
            best_metric = sorted(metrics_values, key=lambda m: metrics_values[m])[0]
            sample = pd.read_csv(samples_folder / f"{best_metric}.csv", index_col=0)
            problem_mask = sample["problem_id"] == problem.id
            best_metric_samples.append(sample[problem_mask])
    predictor_sample = pd.concat(best_metric_samples, ignore_index=True)
    predictor_sample.to_csv(predictor_sample_path)
predictor_sample["nu_str"] = predictor_sample["nu"].astype(str)
columns_x = [c for c in predictor_sample.columns if c.startswith("feature.")]
column_f = "nu_str"

for dim in [2, ]:  # settings
    for i, problem_module in enumerate(problems.bbob._all):
        problem = problem_module.Problem(dim)
        LOG.info(f"{i}: {problem.id}")
        path_predictor = models_folder / f"{problem.id}.predictor"
        if path_predictor.is_file():
            continue
        # Here we mask by problem name to exclude all the dimension of
        # the problem presented in sample
        train_mask = predictor_sample["problem"] != problem.NAME
        test_mask = predictor_sample["problem_id"] == problem.id
        predictor, scores = build_predictor(
            predictor_sample[train_mask].copy(),
            columns_x=columns_x,
            column_f=column_f,
        )
        scores["score_test"] = predictor.score(
            predictor_sample[test_mask][columns_x].replace([np.inf, -np.inf], np.nan),
            predictor_sample[test_mask][column_f],
        )
        joblib.dump(predictor, path_predictor)
        with (models_folder / f"{problem.id} scores.json").open("w") as fi:
            json.dump(scores, fi, indent=4)
LOG.info(f"Stage 3 {timer.stop().str()}\n{json.dumps(timers.str(), indent=2)}\n")
########################################################################################


timer = timers.start("Solve problems using predictor")
for dim, n_init, n_restarts in [(2, 10, 10), ]:  # settings
    for i, problem_class in enumerate(problems.bbob._all):
        metric = METRICS.PREDICTED
        problem: ProblemBase = problem_class.Problem(dim)
        tag = f"{problem.id} {n_init=} {n_restarts=} metric={metric}"
        LOG.info(f"{i}:{tag}")
        data_file = data_folder / f"{tag}.csv"
        if data_file.exists():
            continue
        path_predictor = models_folder / f"{problem.id}.predictor"
        data: pd.DataFrame = solve_problem(
            problem=problem,
            n_init=n_init,
            n_iter=2 * n_init,
            n_seeds=n_restarts,
            metric=metric,
            predictor=load_predictor(problem=problem, path=path_predictor),
        )
        data.to_csv(data_file)
LOG.info(f"Stage 4 {timer.stop().str()}\n{json.dumps(timers.str(), indent=2)}\n")
