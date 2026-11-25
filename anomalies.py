"""
anomalies.py — Profile anomalies: people whose declared age bracket diverges strongly from the
age their health / lifestyle / socio-economic profile suggests.

How it works
  1. 5-fold CV gives out-of-fold (OOF) probabilities for every training person: each prediction
     comes from a model that never saw that person.
  2. "Apparent bracket" = expected bracket under the blended probabilities, sum_k k * p_k
     (continuous, more stable than the argmax).
  3. gap = apparent - real.  gap >= +T -> profile looks OLDER than declared age;
                             gap <= -T -> profile looks YOUNGER.
  4. Characterization compares anomalies with "normal" people of the SAME real bracket, so the
     differences are not just an effect of age itself.

  The anomaly model does NOT use cfg.ROUTING_COLS (questions only asked from bracket 9 up): they
  reveal age through the questionnaire design, which would make anomalies across the 8/9 boundary
  impossible to detect. Use --keep-routing to compare with the competition-style model.

Usage
    python anomalies.py                 # threshold 2.5 brackets (~top 2%), reuses cached OOF if available
    python anomalies.py --threshold 3   # stricter: fewer, more extreme cases
    python anomalies.py --recompute     # rerun the CV (~9 min) e.g. after changing the model
    python anomalies.py --fast          # LightGBM only, lr=0.1 (~1 min), for quick tests
    python anomalies.py --keep-routing  # include the age-routed questions (for comparison)

Outputs (folder outputs/):
    anomalies.csv          one row per anomaly, sorted by |gap|
    anomaly_profile.csv    mean indicator values: anomalies vs same-bracket normals
    gap_distribution.png   histogram of gap with thresholds
    real_vs_apparent.png   heatmap real bracket vs apparent bracket
"""
import argparse
import hashlib

import numpy as np
import pandas as pd

from config import cfg
from cv import git_commit
from data.loader import load_train_data
from evaluation.cv import N_CLASSES, cross_validate, tune_blend
from features.pipeline import FeaturePipeline

OUT_DIR = cfg.BASE_DIR / "outputs"
OOF_CACHE = OUT_DIR / "oof_cache.npz"

# Interpretable indicators used to characterize anomalies (built in profile_indicators)
INDICATORS = {
    "CHRONIC_DISEASE_COUNT": "Nº de doenças crónicas (CD1A-CD1P)",
    "HAS_CHRONIC_CONDITION": "Tem problema de saúde prolongado (HS2)",
    "HS1_ORDERED": "Saúde autoavaliada (0=muito má … 4=muito boa)",
    "SEVERELY_LIMITED": "Severamente limitado nas atividades (HS3)",
    "BMI": "IMC",
    "IS_RETIRED": "Reformado",
    "IS_STUDENT": "Estudante",
    "IS_EMPLOYED": "Tem emprego",
    "IS_WIDOWED": "Viúvo",
    "IS_SINGLE": "Solteiro",
    "HATLEVEL_ORDERED": "Escolaridade (0=nenhuma … 8=doutoramento)",
    "HHINCOME_ORDERED": "Quintil de rendimento (0=1º … 4=5º)",
    "LIVES_ALONE": "Vive sozinho",
    "DRINKS_DAILY": "Bebe álcool todos/quase todos os dias (AL1)",
    "IS_FEMALE": "Mulher",
}


def profile_indicators(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Readable per-person indicators (raw survey + FeaturePipeline features)."""
    feats = FeaturePipeline().fit_transform(X, y)
    ind = pd.DataFrame(index=X.index)
    for col in ["CHRONIC_DISEASE_COUNT", "HAS_CHRONIC_CONDITION", "BMI", "IS_RETIRED", "IS_STUDENT",
                "IS_WIDOWED", "IS_SINGLE"]:
        ind[col] = feats[col]
    # FeaturePipeline uses -1 for "unknown" in ordered columns -> NaN so means are not distorted
    for col in ["HS1_ORDERED", "HATLEVEL_ORDERED", "HHINCOME_ORDERED"]:
        ind[col] = feats[col].where(feats[col] >= 0)
    ind["SEVERELY_LIMITED"] = (X["HS3"] == "Severamente limitado").astype(int)
    ind["IS_EMPLOYED"] = X["MAINSTAT"].str.startswith("Tem um emprego", na=False).astype(int)
    ind["LIVES_ALONE"] = (X["HHTYPE"] == "Agregado unipessoal").astype(int)
    ind["DRINKS_DAILY"] = (X["AL1"] == "Todos os dias ou quase todos os dias").astype(int)
    ind["IS_FEMALE"] = (X["SEX"] == "Mulher").astype(int)
    return ind[list(INDICATORS)]


def model_fingerprint(fast: bool, keep_routing: bool) -> str:
    """Hash of the code that defines the model, so the OOF cache is reused only when it is still valid."""
    h = hashlib.sha1(f"fast={fast};keep_routing={keep_routing}".encode())
    files = [cfg.BASE_DIR / "config.py", cfg.BASE_DIR / "evaluation" / "cv.py"]
    for folder in ["data", "features", "encoders", "models"]:
        files += sorted((cfg.BASE_DIR / folder).glob("*.py"))
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def get_oof(X, y, recompute: bool, fast: bool, keep_routing: bool):
    """OOF blended probabilities, cached so re-analysis (other thresholds, plots) is instant."""
    key = model_fingerprint(fast, keep_routing)
    if not keep_routing:
        X = X.drop(columns=[c for c in cfg.ROUTING_COLS if c in X.columns])
    if OOF_CACHE.exists() and not recompute:
        cache = np.load(OOF_CACHE, allow_pickle=True)
        if str(cache["key"]) == key:
            print("[OOF] Using cached OOF probabilities (--recompute to rerun the CV)")
            return cache["proba"], float(cache["oof_f1"])
        print("[OOF] Model code changed since the cache was made: recomputing")

    models = {"lgb": {"learning_rate": 0.1}} if fast else {"xgb": {}, "lgb": {}}
    print(f">>> Running {cfg.N_SPLITS}-fold CV {list(models)} for OOF probabilities"
          f"{' (fast)' if fast else ' (~9 min)'}...")
    res = cross_validate(X, y, models)
    weights, _, oof_f1 = tune_blend(res.oof, y)
    proba = sum(weights[m] * res.oof[m] for m in res.oof)
    OUT_DIR.mkdir(exist_ok=True)
    np.savez(OOF_CACHE, proba=proba, oof_f1=oof_f1, key=key, commit=git_commit())
    return proba, oof_f1


def characterize(ind: pd.DataFrame, real: pd.Series, mask_anom: pd.Series, mask_normal: pd.Series) -> pd.Series:
    """
    Age-controlled difference: for every real bracket with anomalies, (mean anomalies - mean normals)
    in that bracket, averaged with weights = number of anomalies in the bracket.
    """
    diffs, weights = [], []
    for b in sorted(real[mask_anom].unique()):
        a = ind[mask_anom & (real == b)]
        n = ind[mask_normal & (real == b)]
        if len(n) < 10:
            continue
        diffs.append(a.mean() - n.mean())
        weights.append(len(a))
    if not diffs:
        return pd.Series(np.nan, index=ind.columns)
    return pd.concat(diffs, axis=1).mul(weights, axis=1).sum(axis=1) / sum(weights)


def save_plots(real, apparent, gap, threshold):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plots] matplotlib not installed, skipping plots")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    lim = max(6, int(np.ceil(gap.abs().max())))
    ax.hist(gap, bins=np.arange(-lim, lim + 0.5, 0.5), color="#4C78A8")
    for t in (-threshold, threshold):
        ax.axvline(t, color="#E45756", linestyle="--")
    ax.set_xlabel("Gap = faixa aparente − faixa real (nº de faixas)")
    ax.set_ylabel("Nº de pessoas")
    ax.set_title("Distribuição do desvio entre idade aparente e idade real")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gap_distribution.png", dpi=150)
    plt.close(fig)

    table = pd.crosstab(real, np.clip(np.rint(apparent), 0, N_CLASSES - 1).astype(int), normalize="index")
    table = table.reindex(index=range(N_CLASSES), columns=range(N_CLASSES), fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(table.values, cmap="Blues", origin="lower")
    ax.set_xlabel("Faixa aparente (prevista)")
    ax.set_ylabel("Faixa real")
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_title("Faixa real vs aparente (proporção por linha)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "real_vs_apparent.png", dpi=150)
    plt.close(fig)
    print(f"[Plots] Saved gap_distribution.png and real_vs_apparent.png in {OUT_DIR.name}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=2.5, help="min |gap| in brackets (default 2.5 = top ~2%% of people)")
    parser.add_argument("--recompute", action="store_true", help="ignore cached OOF and rerun the CV")
    parser.add_argument("--fast", action="store_true", help="LightGBM only, lr=0.1 (~1 min) — for quick tests")
    parser.add_argument("--keep-routing", action="store_true",
                        help="keep the questions only asked from bracket 9 up (competition-style model)")
    args = parser.parse_args()
    T = args.threshold

    X, y, raw = load_train_data()
    proba, oof_f1 = get_oof(X, y, args.recompute, args.fast, args.keep_routing)
    print(f"[Model] age-routed questions {'INCLUDED' if args.keep_routing else 'excluded'}")
    print(f"[OOF] Blend Macro F1: {oof_f1:.4f}")

    real = y.reset_index(drop=True)
    apparent = pd.Series(proba @ np.arange(N_CLASSES), name="apparent")
    gap = apparent - real
    older = gap >= T
    younger = gap <= -T
    normal = gap.abs() < 1.5  # clearly consistent people, used as reference group

    n = len(real)
    print(f"\n=== Anomalies (|gap| >= {T:g} brackets) ===")
    print(f"  Profile looks OLDER than declared age  : {older.sum():5d} ({older.mean() * 100:.1f}%)")
    print(f"  Profile looks YOUNGER than declared age: {younger.sum():5d} ({younger.mean() * 100:.1f}%)")
    print(f"  Reference group (|gap| < 1.5)          : {normal.sum():5d} ({normal.mean() * 100:.1f}%)")

    print("\n  Anomalies per real bracket (older / younger):")
    per_b = pd.DataFrame({"older": older.groupby(real).sum(), "younger": younger.groupby(real).sum(),
                          "total": real.value_counts().sort_index()})
    for b, r in per_b.iterrows():
        print(f"    bracket {b:2d}: {int(r.older):4d} / {int(r.younger):4d}   (of {int(r.total)})")

    # Characterization (age-controlled)
    ind = profile_indicators(X, y).reset_index(drop=True)
    profile = pd.DataFrame({
        "indicador": [INDICATORS[c] for c in ind.columns],
        "media_geral": ind.mean().round(3),
        "dif_parece_mais_velho": characterize(ind, real, older, normal).round(3),
        "dif_parece_mais_novo": characterize(ind, real, younger, normal).round(3),
    })
    OUT_DIR.mkdir(exist_ok=True)
    profile.to_csv(OUT_DIR / "anomaly_profile.csv", encoding="utf-8")

    print("\n=== Characterization: anomalies minus same-bracket normals (age-controlled) ===")
    with pd.option_context("display.width", 140, "display.max_colwidth", 50):
        print(profile.to_string())

    # Per-person list
    anom_idx = gap.index[older | younger]
    out = pd.DataFrame({
        "PID": raw["PID"].iloc[anom_idx].values,
        "faixa_real": real[anom_idx].values,
        "faixa_prevista": proba[anom_idx].argmax(axis=1),
        "faixa_aparente": apparent[anom_idx].round(2).values,
        "gap": gap[anom_idx].round(2).values,
        "tipo": np.where(older[anom_idx], "parece_mais_velho", "parece_mais_novo"),
        "prob_faixa_real": proba[anom_idx, real[anom_idx].values].round(3),
    })
    out = pd.concat([out, ind.iloc[anom_idx].reset_index(drop=True),
                     X[["SEX", "MAINSTAT", "MARSTALEGAL", "HS1", "HS3", "HATLEVEL", "HHTYPE"]]
                     .iloc[anom_idx].reset_index(drop=True)], axis=1)
    out = out.reindex(out["gap"].abs().sort_values(ascending=False).index)
    out.to_csv(OUT_DIR / "anomalies.csv", index=False, encoding="utf-8")
    print(f"\n[Saved] {len(out)} anomalies -> {OUT_DIR.name}/anomalies.csv ; profile -> {OUT_DIR.name}/anomaly_profile.csv")

    save_plots(real, apparent, gap, T)


if __name__ == "__main__":
    main()
