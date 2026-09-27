"""SmartScan: adaptive RF spectrum scanning research platform (simulation prototype).

Package layout
--------------
config          Typed experiment configuration (scenario, receiver, reward, PPO) + YAML I/O.
environment     Emitter models, scenario realisation, Turing dataset source, Gymnasium env.
receiver        Limited-bandwidth receiver / detector and receiver-side observation history.
schedulers      Open-loop, random, POMDP (belief-state) and PPO scan schedulers.
rl              PPO training, evaluation and callbacks (Stable-Baselines3).
evaluation      Metrics, multi-seed experiment runner, algorithm comparison.
visualization   Plotly figures in a MATLAB-like engineering style.
ui              Streamlit pages of the engineering console (entry point: app.py).
"""

__version__ = "2.0.0"
