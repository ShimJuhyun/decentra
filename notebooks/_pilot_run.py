"""Runs executor on both GMSC and HC and emits one progress line per major step
so Monitor can stream status. Uses the existing outputs/N01/datasets.pkl.
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, time, pickle, json
import numpy as np, pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from executor import run_case

OUT_ROOT = Path('outputs/pilot_v12'); OUT_ROOT.mkdir(parents=True, exist_ok=True)

with open('outputs/N01/datasets.pkl', 'rb') as f:
    datasets = pickle.load(f)
print(f'LOADED datasets: {list(datasets.keys())}', flush=True)


def fit_teacher(X_tr, y_tr, seed=42):
    X_tr_i, X_val, y_tr_i, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=seed)
    clf = lgb.LGBMClassifier(
        max_depth=-1, num_leaves=63, n_estimators=1000, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
        random_state=seed, n_jobs=-1, verbose=-1)
    clf.fit(X_tr_i, y_tr_i, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    return clf


for ds_name in ['GMSC', 'HC']:
    t0 = time.time()
    d = datasets[ds_name]
    print(f'START {ds_name}: train={len(d["X_train"])} test={len(d["X_test"])} '
          f'features={d["X_train"].shape[1]}', flush=True)

    teacher = fit_teacher(d['X_train'], d['y_train'])
    print(f'TEACHER_FIT {ds_name} in {time.time()-t0:.0f}s', flush=True)

    res = run_case(
        teacher=teacher,
        X_train=d['X_train'], y_train=d['y_train'],
        X_test=d['X_test'],   y_test=d['y_test'],
        out_dir=OUT_ROOT, tag=f'{ds_name}_pilot',
        compute_interventional=True,
        compute_calibration=True,
        compute_scorecard=True,
        n_jobs_ebm=8,
    )

    # Print key numbers for monitoring
    bench = pd.DataFrame(res.bench_rows)
    keys = ['surrogate','R2','AdvTop1','AdvTop4','AdvFull_R','fit_seconds']
    for _, row in bench[keys].iterrows():
        print(f'BENCH {ds_name} {row["surrogate"]:<14s} '
              f'R2={row["R2"]:.3f} AT1={row["AdvTop1"]:.3f} AT4={row["AdvTop4"]:.3f} '
              f'AF_R={row["AdvFull_R"]:.3f} ({row["fit_seconds"]:.0f}s)', flush=True)
    for row in res.calibration_rows:
        if 'error' in row: continue
        print(f'CAL {ds_name} {row["surrogate"]:<10s} {row["method"]:<15s} '
              f'R2={row["R2"]:.3f} AT1={row["AdvTop1"]:.3f} AT4={row["AdvTop4"]:.3f}', flush=True)
    for row in res.scorecard_rows:
        if 'error' in row: continue
        print(f'SCORECARD {ds_name} {row["surrogate"]} R2={row["R2"]:.3f} '
              f'AT1={row["AdvTop1"]:.3f} AT4={row["AdvTop4"]:.3f}', flush=True)
    for row in res.interventional_rows:
        if 'error' in row:
            print(f'INT {ds_name} ERROR: {row["error"]}', flush=True)
        else:
            print(f'INT {ds_name} {row["surrogate"]} DA={row["DA"]:.3f} '
                  f'rho={row["Spearman_rho"]:.3f} IR={row["IR"]:.3f} '
                  f'n_pairs={row["n_pairs"]}', flush=True)
    print(f'DONE {ds_name} in {time.time()-t0:.0f}s', flush=True)

print('ALL_DONE', flush=True)
