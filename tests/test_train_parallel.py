import json
import logging

import pytest

import train_parallel


class DummyModel:
    def __init__(self, device="cpu"):
        self._device = device

    def cpu(self):
        self._device = "cpu"
        return self


@pytest.fixture()
def base_training_cfg():
    return train_parallel.TrainConfig(
        device="cpu",
        task_size=1,
        num_tasks=1,
        noise_var=0.1,
        num_examples=2,
        learning_rate=1e-3,
        training_steps=1,
        batch_size=1,
        eval_batch_size=1,
        print_loss_interval=1,
        print_metrics_interval=1,
        n_checkpoints=None,
        logarithmic_checkpoints=False,
    )


def test_run_sweep_logs_eval_metrics(tmp_path, monkeypatch, caplog, base_training_cfg):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    def fake_train_once(config, training_config, print_model_dimensionality=False, plot_checkpoints=False):
        assert isinstance(training_config, train_parallel.TrainConfig)
        run_id = f"run_{training_config.num_tasks}"
        return run_id, DummyModel(), str(checkpoint_dir)

    def fake_evaluate_run(run_id, **kwargs):
        idx = int(run_id.split("_")[-1])
        return {
            "run_id": run_id,
            "mse/pretrain": 0.1 * idx,
            "mse/true": 0.2 * idx,
            "deltas/pretrain/delta_dmmse": 0.01 * idx,
            "deltas/pretrain/delta_ridge": 0.02 * idx,
            "deltas/true/delta_dmmse": 0.03 * idx,
            "deltas/true/delta_ridge": 0.04 * idx,
            "baseline/pretrain/mse_dmmse": 0.001 * idx,
            "baseline/pretrain/mse_ridge": 0.002 * idx,
            "baseline/true/mse_dmmse": 0.003 * idx,
            "baseline/true/mse_ridge": 0.004 * idx,
        }

    monkeypatch.setattr(train_parallel, "train_once", fake_train_once)
    monkeypatch.setattr(train_parallel, "evaluate_run", fake_evaluate_run)

    caplog.set_level(logging.INFO, logger=train_parallel.__name__)

    model_cfg = train_parallel.ModelConfig()

    num_tasks, results = train_parallel.run_sweep(
        task_sizes=[2],
        num_tasks_list=[3, 5],
        model_cfg=model_cfg,
        base_training_cfg=base_training_cfg,
        device="cpu",
    )

    assert num_tasks == [3, 5]
    assert [r["num_tasks"] for r in results] == [3, 5]
    assert base_training_cfg.num_tasks == 1  # ensure original config is unchanged

    log_file = tmp_path / "metrics.jsonl"
    train_parallel.log_results(results, log_file)

    # Ensure logging emitted entries for both datasets per run
    logged_datasets = [record.message for record in caplog.records if "eval_result" in record.message]
    assert any("dataset=pretrain" in msg for msg in logged_datasets)
    assert any("dataset=true" in msg for msg in logged_datasets)

    entries = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert len(entries) == 4  # two runs * two datasets
    assert {entry["dataset"] for entry in entries} == {"pretrain", "true"}


def test_run_sweep_trains_multiple_num_tasks(monkeypatch, base_training_cfg):
    captured_calls = []

    def fake_train_once(config, training_config, print_model_dimensionality=False, plot_checkpoints=False):
        captured_calls.append(training_config.num_tasks)
        return f"run_{training_config.num_tasks}", DummyModel(), "./checkpoints"

    def fake_evaluate_run(run_id, **kwargs):
        idx = int(run_id.split("_")[-1])
        return {
            "run_id": run_id,
            "mse/pretrain": 0.1 * idx,
            "mse/true": 0.2 * idx,
            "deltas/pretrain/delta_dmmse": 0.01 * idx,
            "deltas/pretrain/delta_ridge": 0.02 * idx,
            "deltas/true/delta_dmmse": 0.03 * idx,
            "deltas/true/delta_ridge": 0.04 * idx,
            "baseline/pretrain/mse_dmmse": 0.001 * idx,
            "baseline/pretrain/mse_ridge": 0.002 * idx,
            "baseline/true/mse_dmmse": 0.003 * idx,
            "baseline/true/mse_ridge": 0.004 * idx,
        }

    monkeypatch.setattr(train_parallel, "train_once", fake_train_once)
    monkeypatch.setattr(train_parallel, "evaluate_run", fake_evaluate_run)

    model_cfg = train_parallel.ModelConfig()
    num_tasks_list = [2, 5, 9]

    _, results = train_parallel.run_sweep(
        task_sizes=[base_training_cfg.task_size],
        num_tasks_list=num_tasks_list,
        model_cfg=model_cfg,
        base_training_cfg=base_training_cfg,
        device="cpu",
    )

    assert captured_calls == num_tasks_list
    assert [result["num_tasks"] for result in results] == num_tasks_list
    assert base_training_cfg.num_tasks == 1  # original config untouched


def test_run_sweep_parallel_balances_devices(monkeypatch, base_training_cfg):
    def fake_train_single_config(args):
        num_tasks, task_size, _model_cfg, _training_cfg, device_id = args
        return {
            "num_tasks": num_tasks,
            "task_size": task_size,
            "run_id": f"run_{num_tasks}",
            "device_id": device_id,
        }

    class DummyPool:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, configs):
            return [func(cfg) for cfg in configs]

    class DummyContext:
        def __init__(self):
            self.processes = None

        def Pool(self, processes=None):
            self.processes = processes
            return DummyPool()

    dummy_context = DummyContext()

    monkeypatch.setattr(train_parallel, "train_single_config", fake_train_single_config)
    monkeypatch.setattr(train_parallel.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(train_parallel.mp, "get_context", lambda method: dummy_context)

    results = train_parallel.run_sweep_parallel(
        task_sizes=[1],
        num_tasks_list=[2, 4, 6, 8],
        model_cfg=train_parallel.ModelConfig(),
        base_training_cfg=base_training_cfg,
        num_gpus=None,
    )[1]

    assert dummy_context.processes == 2
    # There should be one config per num_tasks, each assigned to alternating device ids 0/1
    device_ids = [entry["device_id"] for entry in results]
    assert device_ids == [0, 1, 0, 1]

    # Ensure results can be logged even without evaluator metrics (should fill NaNs)
    train_parallel.log_results(results, None)


def test_train_config_indexing_for_backward_compatibility(base_training_cfg):
    cfg = base_training_cfg.copy_with(num_tasks=7, training_steps=13)
    # dict-style access still works for downstream utilities expecting legacy dicts
    assert cfg["num_tasks"] == 7
    assert cfg.get("training_steps") == 13
    # Non-existent keys should fall back to None just like dict.get
    assert cfg.get("unknown_key") is None
