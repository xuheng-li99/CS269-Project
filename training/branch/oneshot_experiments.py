import copy
import numpy as np
import os
import torch

import datasets.registry
from foundations.step import Step
from training.branch.training_branch import TrainingBranch
import models.registry
from platforms.platform import get_platform
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from utils.tensor_utils import vectorize, unvectorize, shuffle_tensor, shuffle_state_dict, prune

from training.branch.oneshot_experiments_helpers import random, magnitude, synflow, snip, grasp

strategies = [random.RandomPruning, magnitude.MagnitudePruning, synflow.SynFlow, snip.SNIP, grasp.GraSP]


class Branch(TrainingBranch):
    def branch_function(
        self,
        strategy: str,
        prune_fraction: float,
        prune_experiment: str = 'main',
        prune_step: str = '0ep0it',  # The step for states used to prune.
        prune_highest: bool = False,
        prune_iterations: int = 1,
        randomize_layerwise: bool = False,
        state_experiment: str = 'main',
        state_step: str = '0ep0it',  # The step of the state to use alongside the pruning mask.
        start_step: str = '0ep0it',  # The step at which to start the learning rate schedule.
        seed: int = None,
        reinitialize: bool = False
    ):
        # Get the steps for each part of the process.
        iterations_per_epoch = datasets.registry.iterations_per_epoch(self.desc.dataset_hparams)
        prune_step = Step.from_str(prune_step, iterations_per_epoch)
        state_step = Step.from_str(state_step, iterations_per_epoch)
        start_step = Step.from_str(start_step, iterations_per_epoch)
        seed = self.replicate if seed is None else seed

        # Try to load the mask.
        try:
            mask = Mask.load(self.branch_root)
        except:
            mask = None

        if not mask and get_platform().is_primary_process:
            # Gather the weights that will be used for pruning.
            prune_path = self.desc.run_path(self.replicate, prune_experiment)
            prune_model = models.registry.load(prune_path, prune_step, self.desc.model_hparams)

            # Ensure that a valid strategy is available.
            strategy_class = [s for s in strategies if s.valid_name(strategy)]
            if not strategy_class: raise ValueError(f'No such pruning strategy {strategy}')
            if len(strategy_class) > 1: raise ValueError('Multiple matching strategies')
            strategy_instance = strategy_class[0](strategy, self.desc, seed)

            # Run the strategy for each iteration.
            mask = Mask.ones_like(prune_model)
            iteration_fraction = 1 - (1 - prune_fraction) ** (1 / float(prune_iterations))

            if iteration_fraction > 0:
                for it in range(0, prune_iterations):
                    # Make a defensive copy of the model and mask out the pruned weights.
                    prune_model2 = copy.deepcopy(prune_model)
                    with torch.no_grad():
                        for k, v in prune_model2.named_parameters(): v.mul_(mask.get(k, 1))

                    # Compute the scores.
                    scores = strategy_instance.score(prune_model2, mask)

                    # Prune.
                    mask = unvectorize(prune(vectorize(scores), iteration_fraction, not prune_highest, mask=vectorize(mask)), mask)

            # Shuffle randomly per layer.
            if randomize_layerwise: mask = shuffle_state_dict(mask, seed=seed)

            mask = Mask({k: v.clone().detach() for k, v in mask.items()})
            mask.save(self.branch_root)

        # Load the mask.
        get_platform().barrier()
        mask = Mask.load(self.branch_root)
        print('Training')

        # Determine the start step.
        state_path = self.desc.run_path(self.replicate, state_experiment)
        if reinitialize: model = models.registry.get(self.desc.model_hparams)
        else: model = models.registry.load(state_path, state_step, self.desc.model_hparams)
        model = PrunedModel(model, mask)

        train.standard_train(model, self.branch_root, self.desc.dataset_hparams,
                             self.desc.training_hparams, start_step=start_step, verbose=self.verbose,
                             evaluate_every_epoch=self.evaluate_every_epoch)

    @staticmethod
    def description():
        return "Perform various one-shot magnitude pruning experiments."

    @staticmethod
    def name():
        return 'oneshot_experiments'
