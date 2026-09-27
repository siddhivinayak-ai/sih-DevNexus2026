"""RL Training: launch / stop PPO training in a background thread; real logged values only."""

from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from smartscan.config import MODELS_DIR, PROJECT_ROOT
from smartscan.evaluation.experiment import run_experiment
from smartscan.evaluation.metrics import HEADLINE
from smartscan.rl.callbacks import TrainingProgress
from smartscan.rl.train import default_out_dir, train_ppo
from smartscan.ui.common import active_model, fmt, get_config, kpis, list_models, note, panel, plot
from smartscan.visualization.reward import training_losses, training_reward_curve


def _progress() -> TrainingProgress | None:
    return st.session_state.get("training")


def _running() -> bool:
    p = _progress()
    return p is not None and p.status in ("running", "stopping")


def render() -> None:
    cfg = get_config()
    p = cfg.ppo

    with panel("PPO configuration (Stable-Baselines3)"):
        disabled = _running()
        c = st.columns(6)
        p.total_timesteps = int(c[0].number_input("Total timesteps", 1_000, 50_000_000, p.total_timesteps, step=10_000, disabled=disabled))
        p.learning_rate = float(c[1].number_input("Learning rate", 1e-6, 1e-2, p.learning_rate, format="%.5f", disabled=disabled))
        p.gamma = float(c[2].number_input("Gamma γ", 0.5, 0.9999, p.gamma, step=0.01, format="%.4f", disabled=disabled))
        p.gae_lambda = float(c[3].number_input("GAE λ", 0.5, 1.0, p.gae_lambda, step=0.01, disabled=disabled))
        p.ent_coef = float(c[4].number_input("Entropy coef", 0.0, 0.5, p.ent_coef, step=0.005, format="%.4f", disabled=disabled))
        p.clip_range = float(c[5].number_input("Clip range", 0.05, 0.5, p.clip_range, step=0.05, disabled=disabled))
        c = st.columns(6)
        p.n_steps = int(c[0].number_input("n_steps (per env)", 16, 16384, p.n_steps, step=64, disabled=disabled))
        p.batch_size = int(c[1].number_input("Batch size", 16, 16384, p.batch_size, step=64, disabled=disabled))
        p.n_epochs = int(c[2].number_input("Epochs / update", 1, 50, p.n_epochs, disabled=disabled))
        p.n_envs = int(c[3].number_input("Parallel envs", 1, 32, p.n_envs, disabled=disabled))
        p.seed = int(c[4].number_input("Random seed", 0, 10**6, p.seed, disabled=disabled))
        arch = c[5].text_input("Hidden layers", ",".join(map(str, p.net_arch)), disabled=disabled)
        try:
            p.net_arch = [int(x) for x in arch.split(",") if x.strip()]
        except ValueError:
            note("Hidden layers must be comma-separated integers, e.g. 128,128", "err")
        c = st.columns(3)
        p.eval_freq = int(c[0].number_input("Validation every (timesteps)", 1_000, 10_000_000, p.eval_freq, step=5_000, disabled=disabled))
        p.checkpoint_freq = int(c[1].number_input("Checkpoint every (timesteps)", 1_000, 10_000_000, p.checkpoint_freq, step=10_000, disabled=disabled))
        out_dir = c[2].text_input("Output directory", str(default_out_dir(cfg).relative_to(PROJECT_ROOT).as_posix()), disabled=disabled)
        st.caption(f"Training seeds {cfg.seeds.train[0]}–{cfg.seeds.train[1]} · model selection on validation seeds "
                   f"{cfg.seeds.validation[0]}–{cfg.seeds.validation[1]} · test seeds {cfg.seeds.test[0]}–{cfg.seeds.test[1]} "
                   "are never used during training.")

        b = st.columns(5)
        if b[0].button("START TRAINING", type="primary", disabled=_running(), width="stretch"):
            _start(cfg, PROJECT_ROOT / out_dir)
            st.rerun()
        if b[1].button("STOP", disabled=not _running(), width="stretch"):
            _progress().request_stop()
            st.rerun()
        save_clicked = b[2].button("SAVE MODEL", disabled=_running(), width="stretch",
                                   help="Copy the selected model (+config, metadata) to a timestamped snapshot folder")
        load_clicked = b[3].button("LOAD MODEL", disabled=_running(), width="stretch")
        eval_clicked = b[4].button("EVALUATE", disabled=_running(), width="stretch",
                                   help="Quick evaluation of the loaded model on 20 unseen test seeds")

    _model_manager(cfg, save_clicked, load_clicked, eval_clicked)

    @st.fragment(run_every=2.0 if _running() else None)
    def monitor():
        _monitor()

    monitor()


def _start(cfg, out: Path) -> None:
    progress = TrainingProgress()
    progress.status = "running"
    progress.message = "starting"
    progress.total_timesteps = cfg.ppo.total_timesteps
    run_cfg = cfg.copy()

    def job():
        try:
            train_ppo(run_cfg, out, progress)
        except Exception:  # status/message already set by train_ppo
            pass

    th = threading.Thread(target=job, daemon=True, name="smartscan-ppo")
    th.start()
    st.session_state.training = progress
    st.session_state.training_thread = th


def _monitor() -> None:
    p = _progress()
    with panel("Training monitor"):
        if p is None:
            p = _load_finished_progress()
            if p is None:
                note("No training run in this session. Press START TRAINING, or train from the command line: "
                     "<code>python -m smartscan.rl.train --config configs/default.yaml</code>")
                return
            st.caption(f"Showing logged progress of the loaded model ({p.out_dir}).")
        frac = min(1.0, p.timesteps / max(1, p.total_timesteps))
        st.progress(frac, text=f"{p.status.upper()} — {p.timesteps:,} / {p.total_timesteps:,} timesteps ({frac:.1%})"
                    + (f" · {p.message}" if p.message and p.status not in ("running",) else ""))
        tone = {"finished": "good", "error": "bad", "stopped": "warn"}.get(p.status, "")
        kpis([
            ("Status", p.status, "", tone),
            ("Total timesteps", f"{p.timesteps:,}", f"of {p.total_timesteps:,}"),
            ("Episodes", p.episodes),
            ("Current reward", fmt(p.last_episode_reward, "{:.2f}", "—"), "last episode"),
            ("Mean reward", fmt(p.mean_reward(), "{:.2f}", "—"), "last 20 episodes"),
            ("Episode length", fmt(p.mean_length(), "{:.0f}", "—"), "decisions"),
            ("Throughput", f"{p.fps:.0f}" if p.fps else "—", "timesteps / s"),
            ("Best validation", fmt(p.best_validation, "{:.2f}", "—"), "mean episode reward"),
        ])
        c1, c2 = st.columns([1.1, 1])
        with c1:
            plot(training_reward_curve(p.episode_rewards, p.validation, height=330), key="train_curve")
        with c2:
            plot(training_losses(p.train_logs, height=330), key="train_losses")
        if p.validation:
            v = pd.DataFrame(p.validation)
            st.dataframe(v.rename(columns={"timesteps": "timesteps", "mean_reward": "val. mean reward",
                                           "std_reward": "val. std", "interception_rate": "val. interception rate"}),
                         hide_index=True, width="stretch", height=180)
        if p.out_dir and (Path(p.out_dir) / "tb").exists():
            st.caption(f"TensorBoard: `tensorboard --logdir \"{Path(p.out_dir) / 'tb'}\"`")


def _load_finished_progress() -> TrainingProgress | None:
    model = active_model(get_config().scenario.n_bands)
    if model is None:
        return None
    f = model.path.parent / "progress.json"
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    p = TrainingProgress(total_timesteps=d["total_timesteps"], timesteps=d["timesteps"], episodes=d["episodes"],
                         status=d["status"], message=d.get("message", ""), out_dir=str(model.path.parent), fps=d.get("fps", 0))
    p.episode_rewards = [tuple(x) for x in d.get("episode_rewards", [])]
    p.last_episode_reward = d.get("last_episode_reward")
    p.validation = d.get("validation", [])
    p.train_logs = d.get("train_logs", [])
    p.best_validation = d.get("best_validation_reward")
    p.episode_lengths = [int(d["mean_episode_length_20"])] if d.get("mean_episode_length_20") else []
    return p


def _model_manager(cfg, save_clicked: bool, load_clicked: bool, eval_clicked: bool) -> None:
    models = list_models()
    with panel("Models"):
        if not models:
            note("PPO model not trained.", "warn")
            return
        n = cfg.scenario.n_bands
        labels = [m.label + ("" if m.compatible(n) else "  — incompatible with current scenario") for m in models]
        cur = active_model(n)
        idx = next((i for i, m in enumerate(models) if cur and m.path == cur.path), 0)
        choice = st.selectbox("Model", range(len(models)), index=idx, format_func=lambda i: labels[i])
        sel = models[choice]
        if load_clicked:
            if not sel.compatible(n):
                note(f"Model expects a different number of bands ({sel.meta.get('n_bands')}) than the scenario ({n}).", "err")
            else:
                st.session_state.model_path = str(sel.path)
                st.session_state.pop("live", None)
                st.success(f"Loaded {sel.path.parent.name}/{sel.path.name} as the active PPO model.")
        if save_clicked:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dst = MODELS_DIR / f"{sel.path.parent.name}_snapshot_{stamp}"
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(sel.path, dst / "best_model.zip")
            for extra in ("config.json", "metadata.json", "progress.json"):
                if (sel.path.parent / extra).exists():
                    shutil.copy(sel.path.parent / extra, dst / extra)
            st.success(f"Saved snapshot to {dst.relative_to(PROJECT_ROOT).as_posix()}")
        if sel.meta:
            m = sel.meta
            kpis([
                ("Scenario", m.get("scenario", "?")),
                ("Timesteps", f"{m.get('timesteps_trained', 0):,}", "stopped early" if m.get("stopped_early") else ""),
                ("Episodes", m.get("episodes", "?")),
                ("Wall time", f"{m.get('wall_time_s', 0):.0f} s"),
                ("Best val. reward", fmt(m.get("best_validation_reward"), "{:.2f}", "—")),
                ("Obs / actions", f"{m.get('obs_dim')} / {m.get('n_bands')}"),
            ])
        if eval_clicked:
            if not sel.compatible(n):
                note("Selected model is incompatible with the current scenario.", "err")
                return
            seeds = cfg.seeds.seeds("test")[:20]
            with st.spinner(f"Evaluating on {len(seeds)} unseen test seeds..."):
                res = run_experiment(cfg, ["ppo"], seeds, split="test", model_path=sel.path)
            if res.skipped:
                note(f"PPO skipped: {res.skipped.get('ppo')}", "err")
            else:
                st.markdown(f"**Unseen test seeds {seeds[0]}–{seeds[-1]} (n = {len(seeds)}), mean ± std**")
                st.dataframe(res.table(HEADLINE), width="stretch")
                st.session_state.run_history.append((f"PPO quick eval {datetime.now():%H:%M:%S}", res))
