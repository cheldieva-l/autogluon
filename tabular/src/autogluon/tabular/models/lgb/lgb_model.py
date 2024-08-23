import gc
import logging
import os
import random
import re
import time
import warnings

import numpy as np
from pandas import DataFrame, Series

from autogluon.common.features.types import R_BOOL, R_CATEGORY, R_FLOAT, R_INT
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.common.utils.try_import import try_import_lightgbm
from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION, SOFTCLASS
from autogluon.core.models import AbstractModel
from autogluon.core.models._utils import get_early_stopping_rounds

#lgb
#from .callbacks import early_stopping_custom
#autogluon.tabular.models.lgb.callbacks

from . import lgb_utils
from .hyperparameters.parameters import DEFAULT_NUM_BOOST_ROUND, get_lgb_objective, get_param_baseline
from .hyperparameters.searchspaces import get_default_searchspace
from .lgb_utils import construct_dataset, train_lgb_model

############################ brgin
import copy
import logging
import os
import time
import warnings
from operator import gt, lt

from lightgbm.callback import EarlyStopException, _format_eval_result

from autogluon.common.utils.lite import disable_if_lite_mode
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.utils.early_stopping import SimpleES

logger = logging.getLogger(__name__)


# TODO: Add option to stop if current run's metric value is X% lower, such as min 30%, current 40% -> Stop
def early_stopping_custom(
    stopping_rounds,
    first_metric_only=False,
    metrics_to_use=None,
    start_time=None,
    time_limit=None,
    verbose=True,
    max_diff=None,
    ignore_dart_warning=False,
    manual_stop_file=None,
    train_loss_name=None,
    reporter=None,
):
    """Create a callback that activates early stopping.

    Note
    ----
    Activates early stopping.
    The model will train until the validation score stops improving.
    Validation score needs to improve at least every ``early_stopping_rounds`` round(s)
    to continue training.
    Requires at least one validation data and one metric.
    If there's more than one, will check all of them. But the training data is ignored anyway.
    To check only the first metric set ``first_metric_only`` to True.

    Parameters
    ----------
    stopping_rounds : int or tuple
       If int, The possible number of rounds without the trend occurrence.
       If tuple, contains early stopping class as first element and class init kwargs as second element.
    first_metric_only : bool, optional (default=False)
       Whether to use only the first metric for early stopping.
    verbose : bool, optional (default=True)
        Whether to print message with early stopping information.
    train_loss_name : str, optional (default=None):
        Name of metric that contains training loss value.
    reporter : optional (default=None):
        reporter object from AutoGluon scheduler.

    Returns
    -------
    callback : function
        The callback that activates early stopping.
    """
    best_score = []
    best_iter = []
    best_score_list = []
    best_trainloss = []  # stores training losses at corresponding best_iter
    cmp_op = []
    enabled = [True]
    indices_to_check = []
    init_mem_rss = []
    init_mem_avail = []
    es = []

    mem_status = ResourceManager.get_process()

    def _init(env):
        if not ignore_dart_warning:
            enabled[0] = not any((boost_alias in env.params and env.params[boost_alias] == "dart") for boost_alias in ("boosting", "boosting_type", "boost"))
        if not enabled[0]:
            warnings.warn("Early stopping is not available in dart mode")
            return
        if not env.evaluation_result_list:
            raise ValueError("For early stopping, " "at least one dataset and eval metric is required for evaluation")

        if verbose:
            msg = "Training until validation scores don't improve for {} rounds."
            logger.debug(msg.format(stopping_rounds))
            if manual_stop_file:
                logger.debug("Manually stop training by creating file at location: ", manual_stop_file)

        if isinstance(stopping_rounds, int):
            es_template = SimpleES(patience=stopping_rounds)
        else:
            es_template = stopping_rounds[0](**stopping_rounds[1])

        for eval_ret in env.evaluation_result_list:
            best_iter.append(0)
            best_score_list.append(None)
            best_trainloss.append(None)
            es.append(copy.deepcopy(es_template))
            if eval_ret[3]:
                best_score.append(float("-inf"))
                cmp_op.append(gt)
            else:
                best_score.append(float("inf"))
                cmp_op.append(lt)

        if metrics_to_use is None:
            for i in range(len(env.evaluation_result_list)):
                indices_to_check.append(i)
                if first_metric_only:
                    break
        else:
            for i, eval in enumerate(env.evaluation_result_list):
                if (eval[0], eval[1]) in metrics_to_use:
                    indices_to_check.append(i)
                    if first_metric_only:
                        break

        @disable_if_lite_mode()
        def _init_mem():
            init_mem_rss.append(mem_status.memory_info().rss)
            init_mem_avail.append(ResourceManager.get_available_virtual_mem())

        _init_mem()

    @disable_if_lite_mode()
    def _mem_early_stop():
        available = ResourceManager.get_available_virtual_mem()
        cur_rss = mem_status.memory_info().rss

        if cur_rss < init_mem_rss[0]:
            init_mem_rss[0] = cur_rss
        estimated_model_size_mb = (cur_rss - init_mem_rss[0]) >> 20
        available_mb = available >> 20

        model_size_memory_ratio = estimated_model_size_mb / available_mb
        if verbose or (model_size_memory_ratio > 0.25):
            logger.debug("Available Memory: " + str(available_mb) + " MB")
            logger.debug("Estimated Model Size: " + str(estimated_model_size_mb) + " MB")

        early_stop = False
        if model_size_memory_ratio > 1.0:
            logger.warning("Warning: Large GBM model size may cause OOM error if training continues")
            logger.warning("Available Memory: " + str(available_mb) + " MB")
            logger.warning("Estimated GBM model size: " + str(estimated_model_size_mb) + " MB")
            early_stop = True

        # TODO: We will want to track size of model as well, even if we early stop before OOM, we will still crash when saving if the model is large enough
        if available_mb < 512:  # Less than 500 MB
            logger.warning("Warning: Low available memory may cause OOM error if training continues")
            logger.warning("Available Memory: " + str(available_mb) + " MB")
            logger.warning("Estimated GBM model size: " + str(estimated_model_size_mb) + " MB")
            early_stop = True

        if early_stop:
            logger.warning(
                "Warning: Early stopped GBM model prior to optimal result to avoid OOM error. Please increase available memory to avoid subpar model quality."
            )
            logger.log(
                15,
                "Early stopping, best iteration is:\n[%d]\t%s"
                % (best_iter[0] + 1, "\t".join([_format_eval_result(x, show_stdv=False) for x in best_score_list[0]])),
            )
            raise EarlyStopException(best_iter[0], best_score_list[0])

    def _callback(env):
        if not cmp_op:
            _init(env)
        if not enabled[0]:
            return
        if train_loss_name is not None:
            train_loss_evals = [eval for eval in env.evaluation_result_list if eval[0] == "train_set" and eval[1] == train_loss_name]
            train_loss_val = train_loss_evals[0][2]
        else:
            train_loss_val = 0.0
        for i in indices_to_check:
            is_best_iter = False
            eval_result = env.evaluation_result_list[i]
            _, eval_metric, score, greater_is_better = eval_result
            if best_score_list[i] is None or cmp_op[i](score, best_score[i]):
                is_best_iter = True
                best_score[i] = score
                best_iter[i] = env.iteration
                best_score_list[i] = env.evaluation_result_list
                best_trainloss[i] = train_loss_val
            if reporter is not None:  # Report current best scores for iteration, used in HPO
                if i == indices_to_check[0]:  # TODO: documentation needs to note that we assume 0th index is the 'official' validation performance metric.
                    if cmp_op[i] == gt:
                        validation_perf = score
                    else:
                        validation_perf = -score
                    reporter(
                        epoch=env.iteration + 1,
                        validation_performance=validation_perf,
                        train_loss=best_trainloss[i],
                        best_iter_sofar=best_iter[i] + 1,
                        best_valperf_sofar=best_score[i],
                        eval_metric=eval_metric,  # eval_metric here is the stopping_metric from LGBModel
                        greater_is_better=greater_is_better,
                    )
            early_stop = es[i].update(cur_round=env.iteration, is_best=is_best_iter)
            if early_stop:
                if verbose:
                    logger.log(
                        15,
                        "Early stopping, best iteration is:\n[%d]\t%s"
                        % (best_iter[i] + 1, "\t".join([_format_eval_result(x, show_stdv=False) for x in best_score_list[i]])),
                    )
                raise EarlyStopException(best_iter[i], best_score_list[i])
            elif (max_diff is not None) and (abs(score - best_score[i]) > max_diff):
                if verbose:
                    logger.debug("max_diff breached!")
                    logger.debug(abs(score - best_score[i]))
                    logger.log(
                        15,
                        "Early stopping, best iteration is:\n[%d]\t%s"
                        % (best_iter[i] + 1, "\t".join([_format_eval_result(x, show_stdv=False) for x in best_score_list[i]])),
                    )
                raise EarlyStopException(best_iter[i], best_score_list[i])
            if env.iteration == env.end_iteration - 1:
                if verbose:
                    logger.log(
                        15,
                        "Did not meet early stopping criterion. Best iteration is:\n[%d]\t%s"
                        % (best_iter[i] + 1, "\t".join([_format_eval_result(x, show_stdv=False) for x in best_score_list[i]])),
                    )
                raise EarlyStopException(best_iter[i], best_score_list[i])
            if verbose:
                logger.debug((env.iteration - best_iter[i], eval_result))
        if manual_stop_file:
            if os.path.exists(manual_stop_file):
                i = indices_to_check[0]
                logger.log(
                    20,
                    "Found manual stop file, early stopping. Best iteration is:\n[%d]\t%s"
                    % (best_iter[i] + 1, "\t".join([_format_eval_result(x, show_stdv=False) for x in best_score_list[i]])),
                )
                raise EarlyStopException(best_iter[i], best_score_list[i])
        if time_limit:
            time_elapsed = time.time() - start_time
            time_left = time_limit - time_elapsed
            if time_left <= 0:
                i = indices_to_check[0]
                logger.log(
                    20,
                    "\tRan out of time, early stopping on iteration "
                    + str(env.iteration + 1)
                    + ". Best iteration is:\n\t[%d]\t%s" % (best_iter[i] + 1, "\t".join([_format_eval_result(x, show_stdv=False) for x in best_score_list[i]])),
                )
                raise EarlyStopException(best_iter[i], best_score_list[i])

        # TODO: Add toggle parameter to early_stopping to disable this
        # TODO: Identify optimal threshold values for early_stopping based on lack of memory
        if env.iteration % 10 == 0:
            _mem_early_stop()

    _callback.order = 30
    return _callback



##################### end

warnings.filterwarnings("ignore", category=UserWarning, message="Starting from version")  # lightGBM brew libomp warning
logger = logging.getLogger(__name__)


# TODO: Save dataset to binary and reload for HPO. This will avoid the memory spike overhead when training each model and instead it will only occur once upon saving the dataset.
class LGBModel(AbstractModel):
    """
    LightGBM model: https://lightgbm.readthedocs.io/en/latest/

    Hyperparameter options: https://lightgbm.readthedocs.io/en/latest/Parameters.html

    Extra hyperparameter options:
        ag.early_stop : int, specifies the early stopping rounds. Defaults to an adaptive strategy. Recommended to keep default.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._features_internal_map = None
        self._features_internal_list = None
        self._requires_remap = None

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        return get_default_searchspace(problem_type=self.problem_type)

    # Use specialized LightGBM metric if available (fast), otherwise use custom func generator
    def _get_stopping_metric_internal(self):
        stopping_metric = lgb_utils.convert_ag_metric_to_lgbm(ag_metric_name=self.stopping_metric.name, problem_type=self.problem_type)
        if stopping_metric is None:
            stopping_metric = lgb_utils.func_generator(
                metric=self.stopping_metric, is_higher_better=True, needs_pred_proba=not self.stopping_metric.needs_pred, problem_type=self.problem_type
            )
            stopping_metric_name = self.stopping_metric.name
        else:
            stopping_metric_name = stopping_metric
        return stopping_metric, stopping_metric_name

    def _estimate_memory_usage(self, X: DataFrame, **kwargs) -> float:
        """
        Returns the expected peak memory usage in bytes of the LightGBM model during fit.

        The memory usage of LightGBM is primarily made up of two sources:

        1. The size of the data
        2. The size of the histogram cache
            Scales roughly by 5100*num_features*num_leaves bytes
            For 10000 features and 128 num_leaves, the histogram would be 6.5 GB.
        """
        num_classes = self.num_classes if self.num_classes else 1  # self.num_classes could be None after initialization if it's a regression problem
        data_mem_usage = get_approximate_df_mem_usage(X).sum()
        data_mem_usage_bytes = data_mem_usage * 5 + data_mem_usage / 4 * num_classes  # TODO: Extremely crude approximation, can be vastly improved

        params = self._get_model_params(convert_search_spaces_to_default=True)
        max_bins = params.get("max_bins", 255)
        num_leaves = params.get("num_leaves", 31)
        # Memory usage of histogram based on https://github.com/microsoft/LightGBM/issues/562#issuecomment-304524592
        histogram_mem_usage_bytes = 20 * max_bins * len(X.columns) * num_leaves
        histogram_mem_usage_bytes_max = params.get("histogram_pool_size", None)
        if histogram_mem_usage_bytes_max is not None:
            histogram_mem_usage_bytes_max *= 1e6  # Convert megabytes to bytes, `histogram_pool_size` is in MB.
            if histogram_mem_usage_bytes > histogram_mem_usage_bytes_max:
                histogram_mem_usage_bytes = histogram_mem_usage_bytes_max
        histogram_mem_usage_bytes *= 1.2  # Add a 20% buffer

        approx_mem_size_req = data_mem_usage_bytes + histogram_mem_usage_bytes
        return approx_mem_size_req

    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_gpus=0, num_cpus=0, sample_weight=None, sample_weight_val=None, verbosity=2, **kwargs):
        try_import_lightgbm()  # raise helpful error message if LightGBM isn't installed
        start_time = time.time()
        ag_params = self._get_ag_params()
        params = self._get_model_params()

        if verbosity <= 1:
            log_period = False
        elif verbosity == 2:
            log_period = 1000
        elif verbosity == 3:
            log_period = 50
        else:
            log_period = 1

        stopping_metric, stopping_metric_name = self._get_stopping_metric_internal()

        num_boost_round = params.pop("num_boost_round", DEFAULT_NUM_BOOST_ROUND)
        dart_retrain = params.pop("dart_retrain", False)  # Whether to retrain the model to get optimal iteration if model is trained in 'dart' mode.
        if num_gpus != 0:
            if "device" not in params:
                # TODO: lightgbm must have a special install to support GPU: https://github.com/Microsoft/LightGBM/tree/master/python-package#build-gpu-version
                #  Before enabling GPU, we should add code to detect that GPU-enabled version is installed and that a valid GPU exists.
                #  GPU training heavily alters accuracy, often in a negative manner. We will have to be careful about when to use GPU.
                params["device"] = "gpu"
                logger.log(20, f"\tTraining {self.name} with GPU, note that this may negatively impact model quality compared to CPU training.")
        logger.log(15, f"\tFitting {num_boost_round} rounds... Hyperparameters: {params}")

        if "num_threads" not in params:
            params["num_threads"] = num_cpus
        if "objective" not in params:
            params["objective"] = get_lgb_objective(problem_type=self.problem_type)
        if self.problem_type in [MULTICLASS, SOFTCLASS] and "num_classes" not in params:
            params["num_classes"] = self.num_classes
        if "verbose" not in params:
            params["verbose"] = -1

        num_rows_train = len(X)
        dataset_train, dataset_val = self.generate_datasets(
            X=X, y=y, params=params, X_val=X_val, y_val=y_val, sample_weight=sample_weight, sample_weight_val=sample_weight_val
        )
        gc.collect()

        callbacks = []
        valid_names = []
        valid_sets = []
        if dataset_val is not None:
            #from .callbacks import early_stopping_custom

            # TODO: Better solution: Track trend to early stop when score is far worse than best score, or score is trending worse over time
            early_stopping_rounds = ag_params.get("early_stop", "adaptive")
            if isinstance(early_stopping_rounds, (str, tuple, list)):
                early_stopping_rounds = self._get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=early_stopping_rounds)
            if early_stopping_rounds is None:
                early_stopping_rounds = 999999
            reporter = kwargs.get("reporter", None)
            train_loss_name = self._get_train_loss_name() if reporter is not None else None
            if train_loss_name is not None:
                if "metric" not in params or params["metric"] == "":
                    params["metric"] = train_loss_name
                elif train_loss_name not in params["metric"]:
                    params["metric"] = f'{params["metric"]},{train_loss_name}'
            # early stopping callback will be added later by QuantileBooster if problem_type==QUANTILE
            early_stopping_callback_kwargs = dict(
                stopping_rounds=early_stopping_rounds,
                metrics_to_use=[("valid_set", stopping_metric_name)],
                max_diff=None,
                start_time=start_time,
                time_limit=time_limit,
                ignore_dart_warning=True,
                verbose=False,
                manual_stop_file=False,
                reporter=reporter,
                train_loss_name=train_loss_name,
            )
            callbacks += [
                # Note: Don't use self.params_aux['max_memory_usage_ratio'] here as LightGBM handles memory per iteration optimally.  # TODO: Consider using when ratio < 1.
                early_stopping_custom(**early_stopping_callback_kwargs)
            ]
            valid_names = ["valid_set"] + valid_names
            valid_sets = [dataset_val] + valid_sets
        else:
            early_stopping_callback_kwargs = None
        from lightgbm.callback import log_evaluation

        if log_period is not None:
            callbacks.append(log_evaluation(period=log_period))

        seed_val = params.pop("seed_value", 0)
        train_params = {
            "params": params,
            "train_set": dataset_train,
            "num_boost_round": num_boost_round,
            "valid_sets": valid_sets,
            "valid_names": valid_names,
            "callbacks": callbacks,
        }
        if not isinstance(stopping_metric, str):
            train_params["feval"] = stopping_metric
        else:
            if "metric" not in train_params["params"] or train_params["params"]["metric"] == "":
                train_params["params"]["metric"] = stopping_metric
            elif stopping_metric not in train_params["params"]["metric"]:
                train_params["params"]["metric"] = f'{train_params["params"]["metric"]},{stopping_metric}'
        if self.problem_type == SOFTCLASS:
            train_params["fobj"] = lgb_utils.softclass_lgbobj
        elif self.problem_type == QUANTILE:
            train_params["params"]["quantile_levels"] = self.quantile_levels
        if seed_val is not None:
            train_params["params"]["seed"] = seed_val
            random.seed(seed_val)
            np.random.seed(seed_val)

        # Train LightGBM model:
        from lightgbm.basic import LightGBMError

        with warnings.catch_warnings():
            # Filter harmless warnings introduced in lightgbm 3.0, future versions plan to remove: https://github.com/microsoft/LightGBM/issues/3379
            warnings.filterwarnings("ignore", message="Overriding the parameters from Reference Dataset.")
            warnings.filterwarnings("ignore", message="categorical_column in param dict is overridden.")
            try:
                self.model = train_lgb_model(early_stopping_callback_kwargs=early_stopping_callback_kwargs, **train_params)
            except LightGBMError:
                if train_params["params"].get("device", "cpu") != "gpu":
                    raise
                else:
                    logger.warning(
                        "Warning: GPU mode might not be installed for LightGBM, GPU training raised an exception. Falling back to CPU training..."
                        "Refer to LightGBM GPU documentation: https://github.com/Microsoft/LightGBM/tree/master/python-package#build-gpu-version"
                        "One possible method is:"
                        "\tpip uninstall lightgbm -y"
                        "\tpip install lightgbm --install-option=--gpu"
                    )
                    train_params["params"]["device"] = "cpu"
                    self.model = train_lgb_model(early_stopping_callback_kwargs=early_stopping_callback_kwargs, **train_params)
            retrain = False
            if train_params["params"].get("boosting_type", "") == "dart":
                if dataset_val is not None and dart_retrain and (self.model.best_iteration != num_boost_round):
                    retrain = True
                    if time_limit is not None:
                        time_left = time_limit + start_time - time.time()
                        if time_left < 0.5 * time_limit:
                            retrain = False
                    if retrain:
                        logger.log(15, f"Retraining LGB model to optimal iterations ('dart' mode).")
                        train_params.pop("callbacks", None)
                        train_params.pop("valid_sets", None)
                        train_params.pop("valid_names", None)
                        train_params["num_boost_round"] = self.model.best_iteration
                        self.model = train_lgb_model(**train_params)
                    else:
                        logger.log(15, f"Not enough time to retrain LGB model ('dart' mode)...")

        if dataset_val is not None and not retrain:
            self.params_trained["num_boost_round"] = self.model.best_iteration
        else:
            self.params_trained["num_boost_round"] = self.model.current_iteration()

    def _predict_proba(self, X, num_cpus=0, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)

        y_pred_proba = self.model.predict(X, num_threads=num_cpus)
        if self.problem_type == QUANTILE:
            # y_pred_proba is a pd.DataFrame, need to convert
            y_pred_proba = y_pred_proba.to_numpy()
        if self.problem_type in [REGRESSION, QUANTILE, MULTICLASS]:
            return y_pred_proba
        elif self.problem_type == BINARY:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 1:
                return y_pred_proba[:, 1]
            else:
                return y_pred_proba
        elif self.problem_type == SOFTCLASS:  # apply softmax
            y_pred_proba = np.exp(y_pred_proba)
            y_pred_proba = np.multiply(y_pred_proba, 1 / np.sum(y_pred_proba, axis=1)[:, np.newaxis])
            return y_pred_proba
        else:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 2:  # Should this ever happen?
                return y_pred_proba
            else:  # Should this ever happen?
                return y_pred_proba[:, 1]

    def _preprocess_nonadaptive(self, X, is_train=False, **kwargs):
        X = super()._preprocess_nonadaptive(X=X, **kwargs)

        if is_train:
            self._requires_remap = False
            for column in X.columns:
                if isinstance(column, str):
                    new_column = re.sub(r'[",:{}[\]]', "", column)
                    if new_column != column:
                        self._features_internal_map = {feature: i for i, feature in enumerate(list(X.columns))}
                        self._requires_remap = True
                        break
            if self._requires_remap:
                self._features_internal_list = np.array([self._features_internal_map[feature] for feature in list(X.columns)])
            else:
                self._features_internal_list = self._features_internal

        if self._requires_remap:
            X_new = X.copy(deep=False)
            X_new.columns = self._features_internal_list
            return X_new
        else:
            return X

    def generate_datasets(self, X: DataFrame, y: Series, params, X_val=None, y_val=None, sample_weight=None, sample_weight_val=None, save=False):
        lgb_dataset_params_keys = ["two_round"]  # Keys that are specific to lightGBM Dataset object construction.
        data_params = {key: params[key] for key in lgb_dataset_params_keys if key in params}.copy()

        X = self.preprocess(X, is_train=True)
        if X_val is not None:
            X_val = self.preprocess(X_val)
        # TODO: Try creating multiple Datasets for subsets of features, then combining with Dataset.add_features_from(), this might avoid memory spike

        y_og = None
        y_val_og = None
        if self.problem_type == SOFTCLASS:
            y_og = np.array(y)
            y = None
            if X_val is not None:
                y_val_og = np.array(y_val)
                y_val = None

        # X, W_train = self.convert_to_weight(X=X)
        dataset_train = construct_dataset(
            x=X, y=y, location=os.path.join("self.path", "datasets", "train"), params=data_params, save=save, weight=sample_weight
        )
        # dataset_train = construct_dataset_lowest_memory(X=X, y=y, location=self.path + 'datasets/train', params=data_params)
        if X_val is not None:
            # X_val, W_val = self.convert_to_weight(X=X_val)
            dataset_val = construct_dataset(
                x=X_val,
                y=y_val,
                location=os.path.join(self.path, "datasets", "val"),
                reference=dataset_train,
                params=data_params,
                save=save,
                weight=sample_weight_val,
            )
            # dataset_val = construct_dataset_lowest_memory(X=X_val, y=y_val, location=self.path + 'datasets/val', reference=dataset_train, params=data_params)
        else:
            dataset_val = None
        if self.problem_type == SOFTCLASS:
            if y_og is not None:
                dataset_train.softlabels = y_og
            if y_val_og is not None:
                dataset_val.softlabels = y_val_og
        return dataset_train, dataset_val

    def _get_train_loss_name(self):
        if self.problem_type == BINARY:
            train_loss_name = "binary_logloss"
        elif self.problem_type == MULTICLASS:
            train_loss_name = "multi_logloss"
        elif self.problem_type == REGRESSION:
            train_loss_name = "l2"
        else:
            raise ValueError(f"unknown problem_type for LGBModel: {self.problem_type}")
        return train_loss_name

    def _get_early_stopping_rounds(self, num_rows_train, strategy="auto"):
        return get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=strategy)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_BOOL, R_INT, R_FLOAT, R_CATEGORY],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _is_gpu_lgbm_installed(self):
        # Taken from https://github.com/microsoft/LightGBM/issues/3939
        try_import_lightgbm()
        import lightgbm

        try:
            data = np.random.rand(50, 2)
            label = np.random.randint(2, size=50)
            train_data = lightgbm.Dataset(data, label=label)
            params = {"device": "gpu"}
            gbm = lightgbm.train(params, train_set=train_data, verbose=-1)
            return True
        except Exception as e:
            return False

    def get_minimum_resources(self, is_gpu_available=False):
        minimum_resources = {
            "num_cpus": 1,
        }
        if is_gpu_available and self._is_gpu_lgbm_installed():
            minimum_resources["num_gpus"] = 0.5
        return minimum_resources

    def _get_default_resources(self):
        # logical=False is faster in training
        num_cpus = ResourceManager.get_cpu_count_psutil(logical=False)
        num_gpus = 0
        return num_cpus, num_gpus

    @property
    def _features(self):
        return self._features_internal_list

    def _ag_params(self) -> set:
        return {"early_stop"}

    def _more_tags(self):
        # `can_refit_full=True` because num_boost_round is communicated at end of `_fit`
        return {"can_refit_full": True}
