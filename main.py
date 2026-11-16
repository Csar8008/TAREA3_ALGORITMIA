import math
import random
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt


def tour_length(points, tour) -> float:
    return sum(math.dist(points[tour[i]], points[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))


def two_opt_move(route: list[int], i: int, k: int) -> list[int]:
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]


def simulated_annealing(points, tour, T0: float, Tf: float, iterations: int):
    n, current = len(tour), tour_length(points, tour)
    best = current
    for t in range(iterations):
        T = T0 * (Tf / T0) ** (t / iterations)
        i, k = sorted(random.sample(range(1, n), 2))
        a, b = points[tour[i - 1]], points[tour[i]]
        c, d = points[tour[k]], points[tour[(k + 1) % n]]
        delta = math.dist(a, c) + math.dist(b, d) - math.dist(a, b) - math.dist(c, d)
        if delta < 0 or random.random() < math.exp(-delta / T):
            tour, current = two_opt_move(tour, i, k), current + delta
        best = min(best, current)
    return tour, best


def bootstrap_mediana(datos: list[float], n_boot: int = 10_000, alpha: float = 0.05):
    rng = np.random.default_rng(42)
    medianas = np.median(rng.choice(datos, size=(n_boot, len(datos)), replace=True), axis=1)
    return float(np.median(medianas)), *np.percentile(medianas, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def guardar_o_mostrar(filename: str) -> None:
    plt.savefig(filename, dpi=150)
    print(f"Gráfica guardada automáticamente en: {filename}")
    if input(f"¿Mostrar {filename} en pantalla? (s/n): ").strip().lower() == "s":
        plt.show()
    plt.close()


def plot_efectos_principales(df: pd.DataFrame, etiquetas_temp: dict, filename: str = "ex1_efectos_principales.png"):
    niveles = ["Bajo", "Medio", "Alto"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    medias_t = df.groupby("nivel_temp")["costo"].mean().reindex(niveles)
    ax1.plot([etiquetas_temp[nv] for nv in niveles], medias_t.values, "o-", linewidth=2, color="tab:blue")
    ax1.set(title="Efecto Principal: Temperatura Inicial ($T_0$)", xlabel="Nivel de Temperatura ($T_0$)",
            ylabel="Costo medio del tour (menor es mejor)")

    medias_n = df.groupby("tamano")["costo"].mean()
    ax2.plot([str(n) for n in medias_n.index], medias_n.values, "s-", linewidth=2, color="tab:green")
    ax2.set(title="Efecto Principal: Tamaño del Problema ($n$)", xlabel="Número de ciudades ($n$)",
            ylabel="Costo medio del tour")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.suptitle("DOE Opción A: Gráfica de Efectos Principales", fontsize=13)
    fig.tight_layout()
    guardar_o_mostrar(filename)


def plot_interaccion(df: pd.DataFrame, etiquetas_temp: dict, filename: str = "ex1_interaccion.png"):
    niveles = ["Bajo", "Medio", "Alto"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for n, marcador in zip([5, 10, 15], ["o", "s", "^"]):
        medias = df[df["tamano"] == n].groupby("nivel_temp")["costo"].mean().reindex(niveles)
        ax.plot([etiquetas_temp[nv] for nv in niveles], medias.values, marker=marcador, linewidth=2,
                label=f"Tamaño n = {n}")
    ax.set(title="DOE Opción A: Gráfica de Interacción (Temperatura × Tamaño)",
           xlabel="Nivel de Temperatura Inicial ($T_0$)", ylabel="Costo medio del tour (menor es mejor)")
    ax.legend(title="Bloque (Instancia)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    guardar_o_mostrar(filename)


if __name__ == "__main__":
    MATRICULA, ALPHA = 2069375, 0.05
    random.seed(MATRICULA)
    r = random.random()

    T_min, T_max, Tf, iterations = 0.5, 25.0, 0.01, 800
    tamanos, replicas = [5, 10, 15], 10

    fracciones = {"Bajo": 0.3 * r, "Medio": 0.3 * r + 0.3, "Alto": 0.3 * r + 0.6}
    temperaturas = {nivel: round(T_min + f * (T_max - T_min), 4) for nivel, f in fracciones.items()}
    etiquetas_temp = {nv: f"{nv}\n(T0={val:.2f})" for nv, val in temperaturas.items()}
    print(f"Matrícula: {MATRICULA} | r = {r:.6f}")
    print(f"Rango: [{T_min}, {T_max}] -> Niveles T0: {temperaturas}\n")

    instancias = {}
    for n in tamanos:
        random.seed(1000 + n)
        instancias[n] = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(n)]

    registros = []
    for n in tamanos:
        for nivel, T0 in temperaturas.items():
            for rep in range(replicas):
                random.seed(f"{MATRICULA}-{n}-{nivel}-{rep}")
                _, costo = simulated_annealing(instancias[n], random.sample(range(n), n), T0, Tf, iterations)
                registros.append({"tamano": n, "nivel_temp": nivel, "T0": T0, "replica": rep + 1,
                                  "costo": round(costo, 9)})
    df = pd.DataFrame(registros)
    df.to_csv("ex1_resultados_doe.csv", index=False)

    print(f"TABLA ANOVA DE DOS FACTORES ( alpha = {ALPHA})")
    tabla_anova = sm.stats.anova_lm(ols("costo ~ C(nivel_temp) * C(tamano)", data=df).fit(), typ=2)
    print(tabla_anova)
    p_temp, p_inter = tabla_anova["PR(>F)"].iloc[[0, 2]]
    print(f"\nTemperatura: p = {p_temp:.4f} | Interacción: p = {p_inter:.4f}\n")

    todos_puntos = []
    resumen = []
    for n in tamanos:
        candidatos = {}
        for nv in temperaturas:
            datos = df[(df["tamano"] == n) & (df["nivel_temp"] == nv)]["costo"].tolist()
            med, inf, sup = bootstrap_mediana(datos)
            media_val = float(np.mean(datos))
            std_val = float(np.std(datos, ddof=1))

            todos_puntos.append({
                "n": n,
                "nivel_temp": nv,
                "T0": temperaturas[nv],
                "mediana_boot": med,
                "ic_inf": inf,
                "ic_sup": sup,
                "media": media_val,
                "std": std_val
            })
            candidatos[nv] = (round(med, 6), round(media_val, 6), round(std_val, 6), med, inf, sup)

        mejor = min(candidatos, key=lambda nv: (candidatos[nv][0], candidatos[nv][1], candidatos[nv][2]))
        _, media_best, std_best, med, inf, sup = candidatos[mejor]
        resumen.append({
            "n": n,
            "nivel_optimo": mejor,
            "T0_optima": temperaturas[mejor],
            "mediana_boot": med,
            "ic_inf": inf,
            "ic_sup": sup,
            "media": media_best,
            "std": std_best
        })

    print("ESTADÍSTICOS BOOTSTRAP DE LOS 9 PUNTOS DE DISEÑO:")
    print(pd.DataFrame(todos_puntos).round(4).to_string(index=False))
    print()

    print("TABLA RESUMEN POR TAMAÑO:")
    print(pd.DataFrame(resumen).round(4).to_string(index=False))
    print()

    plot_efectos_principales(df, etiquetas_temp)
    plot_interaccion(df, etiquetas_temp)