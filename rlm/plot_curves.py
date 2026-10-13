from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(path: str) -> list[dict]:
    """Acepta un trainer_state.json, o una carpeta (busca el checkpoint más reciente)."""
    p = Path(path)
    if p.is_dir():
        direct = p / "trainer_state.json"
        if direct.exists():
            p = direct
        else:
            ckpts = sorted(p.glob("checkpoint-*/trainer_state.json"),
                           key=lambda x: int(x.parent.name.split("-")[1]))
            if not ckpts:
                raise FileNotFoundError(f"No hay trainer_state.json en {path}")
            p = ckpts[-1]
    return json.loads(p.read_text())["log_history"]


def series(history: list[dict], key: str) -> tuple[list[int], list[float]]:
    rows = [r for r in history if key in r]
    return [r["step"] for r in rows], [r[key] for r in rows]


def smooth(ys: list[float], window: int) -> list[float]:
    """Media móvil; las curvas de GRPO son muy ruidosas con logging_steps=1."""
    if window <= 1:
        return ys
    out = []
    for i in range(len(ys)):
        chunk = ys[max(0, i - window + 1): i + 1]
        out.append(sum(chunk) / len(chunk))
    return out

REWARD_KEYS = [
    ("rewards/format_reward/mean", "format"),
    ("rewards/accuracy_reward/mean", "accuracy"),
    ("rewards/domain_reward/mean", "domain"),
]


def plot_reward_summary(histories, labels, window: int, output: str) -> None:
    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4),
                             sharey=True, squeeze=False)
    for ax, hist, label in zip(axes[0], histories, labels):
        # Solo los pasos en los que están las tres recompensas
        rows = [r for r in hist if all(k in r for k, _ in REWARD_KEYS)]
        steps = [r["step"] for r in rows]
        if not rows:
            ax.set_title(f"{label} (sin datos)")
            continue

        per_reward = {name: [r[k] for r in rows] for k, name in REWARD_KEYS}
        mean_all = [sum(vals) / len(vals) for vals in zip(*per_reward.values())]

        for name, ys in per_reward.items():
            ax.plot(steps, smooth(ys, window), label=name, alpha=0.85)
        
        mean_smooth = smooth(mean_all, window)
        ax.plot(steps, mean_smooth, "k--", linewidth=2.2, label="media")

        # Punto rojo en el máximo de la media (suavizada)
        i_max = max(range(len(mean_smooth)), key=mean_smooth.__getitem__)
        ax.scatter(steps[i_max], mean_smooth[i_max], color="red", s=70, zorder=5,
                   label=f"máx = {mean_smooth[i_max]:.3f} (step {steps[i_max]})")
        ax.annotate(f"{mean_smooth[i_max]:.3f}", (steps[i_max], mean_smooth[i_max]),
                    textcoords="offset points", xytext=(0, 10), ha="center", color="red")

        ax.set_title(label)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0][0].set_ylabel("recompensa media")

    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"guardado en {output}")


# (clave en log_history, título del panel)
PANELS = [
    ("rewards/format_reward/mean", "Format reward"),
    ("rewards/accuracy_reward/mean", "Accuracy reward (≈ pass@1 en train)"),
    ("rewards/domain_reward/mean", "Domain reward"),
    ("completions/mean_length", "Longitud media de la completion (tokens)"),
    ("reward_std", "Desviación de la recompensa dentro del grupo"),
    ("kl", "KL"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+",
                        help="uno o más trainer_state.json o carpetas de run")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="nombre de cada run en la leyenda")
    parser.add_argument("--smooth", type=int, default=10, help="ventana de la media móvil")
    parser.add_argument("--output", default="rlm/eval/curves.png")
    args = parser.parse_args()

    labels = args.labels or [Path(r).name for r in args.runs]
    histories = [load_history(r) for r in args.runs]

    # Solo pintamos los paneles que existan en algún run (p. ej. kl solo si beta > 0)
    panels = [(k, t) for k, t in PANELS if any(series(h, k)[0] for h in histories)]
    ncols = 2
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows), squeeze=False)

    for ax, (key, title) in zip(axes.flat, panels):
        for hist, label in zip(histories, labels):
            xs, ys = series(hist, key)
            if not xs:
                continue
            ax.plot(xs, ys, alpha=0.25)                      # curva cruda, tenue
            ax.plot(xs, smooth(ys, args.smooth), label=label)  # curva suavizada
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend()

    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"guardado en {args.output}")

    summary_path = str(Path(args.output).with_name(Path(args.output).stem + "_rewards.png"))
    plot_reward_summary(histories, labels, args.smooth, summary_path)


if __name__ == "__main__":
    main()