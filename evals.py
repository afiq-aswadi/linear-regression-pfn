import functools

from baselines import dmmse_predictor, ridge_predictor
from samplers.tasks import DiscreteTaskDistribution, GaussianTaskDistribution
from samplers.tasks import RegressionSequenceDistribution


def mse(y1, y2, axis=None):
    """
    Loss function: Mean squared error between the elements of two tensors of
    the same shape (summed along all axes or only `axis`).

    * Used as a loss function for least-squares regression
      (e.g., `mse(ys_true, ys_pred)`).
    * Used to compare the difference between two algorithms' regression
      predictions.
      (e.g., `mse(ys_algo1, ys_algo2)`).
    * If `ys1` and `ys2` are (batch, time, dimension) tensors, then we can
      get a vector of per-token losses by averaging over only the first and
      last dimensions (e.g., `mse(ys1, ys2, axis=(0, 2))`).
    """
    return (y1 - y2).square().mean(axis=axis)


class ICLEvaluator:
    """
    Stores fixed evaluation data batches, computed at the start of the
    training run, as well as baseline predictions for these batches.
    """
    def __init__(
        self,
        pretrain_dist,
        true_dist,
        max_examples,
        eval_batch_size,
    ):
        # fixed evaluation batches (computed once at start of training run)
        self.pretrain_xs, self.pretrain_ys = pretrain_dist.get_batch(
            num_examples=max_examples,
            batch_size=eval_batch_size,
        )
        self.true_xs, self.true_ys = true_dist.get_batch(
            num_examples=max_examples,
            batch_size=eval_batch_size,
        )

        # configure baseline predictors
        # dmmse is the bayes-optimal predictor for the pretraining data
        dmmse = functools.partial(
            dmmse_predictor,
            prior=pretrain_dist.task_distribution,
            noise_variance=pretrain_dist.noise_variance,
        )
        # ridge is the bayes-optimal predictor for the true data
        ridge = functools.partial(
            ridge_predictor,
            noise_variance=true_dist.noise_variance,
        )

        # cache baseline predictions (to compare against model predictions)
        self.pretrain_dmmse_preds = dmmse(self.pretrain_xs, self.pretrain_ys)
        self.pretrain_ridge_preds = ridge(self.pretrain_xs, self.pretrain_ys)
        self.true_dmmse_preds = dmmse(self.true_xs, self.true_ys)
        self.true_ridge_preds = ridge(self.true_xs, self.true_ys)

    
    def __call__(self, model):
        """
        Evaluate a model against stored batches, returning a dictionary of
        various metrics.
        """
        # compute model predictions and loss on fixed batch from T_pretrain
        pretrain_model_preds = model(self.pretrain_xs, self.pretrain_ys)
        pretrain_model_losses = mse(
            self.pretrain_ys,
            pretrain_model_preds,
            axis=(0,2),
        )
        # compute model predictions and loss on fixed batch from T_true
        true_model_preds = model(self.true_xs, self.true_ys)
        true_model_losses = mse(
            self.true_ys,
            true_model_preds,
            axis=(0,2),
        )
        # compute and return various metrics based on above
        k = len(pretrain_model_losses)
        return {
            "mse/pretrain": pretrain_model_losses.mean().item(),
            "mse/true": true_model_losses.mean().item(),
            "deltas/pretrain/delta_dmmse": mse(
                pretrain_model_preds,
                self.pretrain_dmmse_preds,
            ),
            "deltas/pretrain/delta_ridge": mse(
                pretrain_model_preds,
                self.pretrain_ridge_preds,
            ),
            "deltas/true/delta_dmmse": mse(
                true_model_preds,
                self.true_dmmse_preds,
            ),
            "deltas/true/delta_ridge": mse(
                true_model_preds,
                self.true_ridge_preds,
            ),
            "pertoken/pretrain/0": pretrain_model_losses[0],
            f"pertoken/pretrain/{k//2}": pretrain_model_losses[k//2],
            f"pertoken/pretrain/{k-1}": pretrain_model_losses[k-1],
            "pertoken/true/0": true_model_losses[0],
            f"pertoken/true/{k//2}": true_model_losses[k//2],
            f"pertoken/true/{k-1}": true_model_losses[k-1],
            # all per-token losses:
            # **{
            #     f"pertoken/pretrain/{i}": l
            #     for i, l in enumerate(pretrain_model_losses)
            # },
            # **{
            #     f"pertoken/true/{i}": l
            #     for i, l in enumerate(true_model_losses)
            # },
        }
