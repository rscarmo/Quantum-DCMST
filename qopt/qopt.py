import os
from typing import Callable
import os.path as opath
import json
import pandas as pd

from .config import load_config
import numpy as np

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.evolutionary import DE
from opytimizer.optimizers.science import GSA


class QOPT(object):
    def __init__(self, config_file: str):
        self.config = load_config(config_file)
        self.opt_dict = {
            "PSO": PSO,
            "DE": DE,
            "GSA": GSA,
        }
        self._validate_config()
        self.search_space = self._create_search_space()
        self.optimizer = self._create_optimizer()

    def _validate_config(self):
        if "optimizer" not in self.config:
            raise ValueError("Optimizer not defined in the configuration file")

        optimizer_name = self.config["optimizer"]["name"]
        if optimizer_name not in self.opt_dict:
            raise ValueError(f"Optimizer '{optimizer_name}' is not supported")

        required_fields = {"PSO": ["w", "c1", "c2"], "DE": ["CR", "F"], "GSA": ["G"]}

        for field in required_fields[optimizer_name]:
            if field not in self.config[optimizer_name]:
                raise ValueError(
                    f"Field '{field}' is required for optimizer '{optimizer_name}'"
                )
        root_keys = self.config.keys()
        if optimizer_name not in root_keys:
            raise ValueError(
                f"Optimizer '{optimizer_name}' is not in the root keys. Options are: {','.join(self.opt_dict.keys())}"
            )

    def _create_search_space(self):
        params = self.config["search-space"]
        search_space = SearchSpace(**params)
        return search_space

    def _create_optimizer(self):
        optimizer_name = self.config["optimizer"]["name"]
        optimizer_params = self.config[optimizer_name]
        optimizer = self.opt_dict[optimizer_name](params=optimizer_params)
        return optimizer

    def __call__(self, loss_function: Callable):

        def _flatten_fn(x):
            return loss_function(x.flatten())

        loss_fn = Function(_flatten_fn)
        opt = Opytimizer(
            space=self.search_space, optimizer=self.optimizer, function=loss_fn
        )
        results = {"run": [], "best_values": []}

        for i in range(self.config["search-space"]["n_variables"]):
            results[f"params_{i}"] = []

        for run_nr in range(self.config["nr_runs"]):
            opt.start(n_iterations=self.config["optimizer"]["n_iterations"])
            best_position = opt.space.best_agent.position.flatten()
            best_value = loss_function(best_position)
            best_position = [float(z) for z in best_position]
            results["run"].append(run_nr)
            # results["best_positions"].append(list(best_position))
            for i in range(self.config["search-space"]["n_variables"]):
                results[f"params_{i}"].append(float(best_position[i]))

            results["best_values"].append(float(best_value))

        # Salvar resultados em um arquivo CSV
        results_df = pd.DataFrame(results)
        dst_path = opath.join("experiments", "results", self.config["name"])
        os.makedirs(dst_path, exist_ok=True)
        tmstamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        filename = "result_" + tmstamp + ".csv"
        results_df.to_csv(opath.join(dst_path, filename), index=False)

        # Generate descriptive statistics for the parameters
        params_df = results_df[
            [f"params_{i}" for i in range(self.config["search-space"]["n_variables"])]
        ]
        description = params_df.describe()
        description_filename = "summary_" + tmstamp + ".csv"
        description.to_csv(opath.join(dst_path, description_filename), index=True)

        return description
