# %% [markdown]
# # Hackatón IIA · T01 · ¿Este inicio de sesión es una intrusión?
# Pista: Neurona (una sola neurona con pesos y sesgo). Solo NumPy + Pandas + Matplotlib.
# Ejecutable en VS Code (celdas `# %%`) o Jupyter.

# %% 1. CARGA Y VALIDACIÓN
# Qué hace: lee T01.csv y respeta la división de la ficha (conjunto = entrenamiento / prueba).
# Por qué: la división NO se puede cambiar (Bases, punto 4) y evita filtración entre train y test.
# Resultado esperado: 14 filas de entrenamiento, 6 de prueba; 3 intrusiones en total.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("T01.csv")
df = df.rename(columns={
    "Intentos fallidos previos (en la última hora)": "intentos",
    "Distancia al último acceso (km)": "distancia_km",
    "Datos descargados (MB en la sesión)": "mb",
    "resultado_real": "y",
})
assert len(df) == 20 and df.isna().sum().sum() == 0
assert (df[df.caso <= 14].conjunto == "entrenamiento").all()
assert (df[df.caso >= 15].conjunto == "prueba").all()

train = df[df.conjunto == "entrenamiento"].reset_index(drop=True)
test = df[df.conjunto == "prueba"].reset_index(drop=True)
print("Entrenamiento:", len(train), "| intrusiones:", train.y.sum())
print("Prueba:       ", len(test), "| intrusiones:", test.y.sum())
print(df.groupby(["conjunto", "y"])[["intentos", "distancia_km", "mb"]].agg(["min", "max"]))

# %% 2. EXPLORACIÓN VISUAL
# Qué hace: dibuja intentos vs distancia, color = clase real, forma = conjunto.
# Por qué: si una recta separa las clases, UNA neurona alcanza (no hace falta red).
# Resultado esperado: las 3 intrusiones quedan arriba a la derecha (muchos intentos Y mucha distancia).
fig, ax = plt.subplots(figsize=(7, 5))
for conj, marca in [("entrenamiento", "o"), ("prueba", "s")]:
    for clase, color in [(0, "tab:blue"), (1, "tab:red")]:
        s = df[(df.conjunto == conj) & (df.y == clase)]
        ax.scatter(s.intentos, s.distancia_km, c=color, marker=marca, s=80,
                   label=f"{conj} · {'intrusión' if clase else 'normal'}", edgecolor="k")
for _, r in df.iterrows():
    ax.annotate(int(r.caso), (r.intentos + 0.15, r.distancia_km + 80), fontsize=8)
ax.set_xlabel("Intentos fallidos previos (última hora)")
ax.set_ylabel("Distancia al último acceso (km)")
ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("dispersion.png", dpi=150); plt.show()

# %% 3. FUNCIONES DE MEDICIÓN (matriz de confusión y métricas, a mano)
# Qué hace: calcula VP, FN, FP, VN y las 4 métricas que pide el README.
# Por qué: el README exige matrices que cierren con los datos y las 4 métricas por conjunto.
# Resultado esperado: VP+FN+FP+VN = cantidad de casos del conjunto.
def matriz(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    vp = int(((y_real == 1) & (y_pred == 1)).sum())
    fn = int(((y_real == 1) & (y_pred == 0)).sum())
    fp = int(((y_real == 0) & (y_pred == 1)).sum())
    vn = int(((y_real == 0) & (y_pred == 0)).sum())
    return vp, fn, fp, vn

def metricas(vp, fn, fp, vn):
    div = lambda a, b: a / b if b else float("nan")   # nan = indefinido (denominador 0)
    exactitud = div(vp + vn, vp + fn + fp + vn)
    precision = div(vp, vp + fp)
    sensibilidad = div(vp, vp + fn)       # = recall
    especificidad = div(vn, vn + fp)
    f1 = div(2 * precision * sensibilidad, precision + sensibilidad) if vp else 0.0
    return dict(exactitud=exactitud, precision=precision, sensibilidad=sensibilidad,
                especificidad=especificidad, f1=f1)

def reporte(nombre, y_real, y_pred):
    vp, fn, fp, vn = matriz(y_real, y_pred)
    assert vp + fn + fp + vn == len(y_real)
    print(f"\n{nombre}:  VP={vp}  FN={fn}  FP={fp}  VN={vn}")
    for k, v in metricas(vp, fn, fp, vn).items():
        print(f"   {k:14s} {v:.3f}")
    return vp, fn, fp, vn

# %% 4. LÍNEA BASE TRIVIAL Y REGLAS DE UNA SOLA VARIABLE
# Qué hace: mide "responder siempre NO" y reglas con una sola variable.
# Por qué: rúbrica E y tope 7 si la métrica principal es exactitud sin línea base.
# Resultado esperado: "siempre NO" saca ~86% / ~83% de exactitud con sensibilidad 0.
print("=== LÍNEA BASE: siempre NO ===")
reporte("Entrenamiento", train.y, np.zeros(len(train), int))
reporte("Prueba", test.y, np.zeros(len(test), int))

print("\n=== SOLO INTENTOS (umbral >= 13, el mejor en entrenamiento) ===")
reporte("Entrenamiento", train.y, (train.intentos >= 13).astype(int))
reporte("Prueba", test.y, (test.intentos >= 13).astype(int))

# %% 5. LA NEURONA
# Qué hace: z = w1*x1 + w2*x2 + b ; a = sigmoide(z) ; decide SÍ si a >= umbral.
# Por qué sigmoide: devuelve un número entre 0 y 1 (probabilidad de intrusión) y es derivable,
#   lo que permite aprender por descenso del gradiente. Con umbral 0.5 equivale a "z >= 0".
# Entradas: intentos y distancia_km, estandarizadas con media/desvío SOLO del entrenamiento.
# Pérdida: entropía cruzada ponderada (pocas intrusiones: 2 de 14) + L2 pequeña
#   (con datos separables, sin L2 los pesos crecen sin parar).
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

class Neurona:
    def __init__(self, columnas, lr=0.1, epocas=5000, l2=0.01):
        self.cols, self.lr, self.epocas, self.l2 = columnas, lr, epocas, l2

    def fit(self, X_df, y):
        X = X_df[self.cols].to_numpy(float)
        self.mu, self.sd = X.mean(0), X.std(0)          # solo entrenamiento
        Xs = (X - self.mu) / self.sd
        y = np.asarray(y, float)
        peso_clase = np.where(y == 1, (y == 0).sum() / (y == 1).sum(), 1.0)  # balancea clases
        self.w, self.b = np.zeros(Xs.shape[1]), 0.0
        self.hist = []
        for _ in range(self.epocas):
            a = sigmoide(Xs @ self.w + self.b)
            err = (a - y) * peso_clase
            self.w -= self.lr * ((Xs.T @ err) / len(y) + self.l2 * self.w)
            self.b -= self.lr * err.mean()
            eps = 1e-12
            self.hist.append(-np.mean(peso_clase * (y*np.log(a+eps) + (1-y)*np.log(1-a+eps))))
        return self

    def z(self, X_df):
        Xs = (X_df[self.cols].to_numpy(float) - self.mu) / self.sd
        return Xs @ self.w + self.b

    def proba(self, X_df):
        return sigmoide(self.z(X_df))

    def predict(self, X_df, umbral=0.5):
        return (self.proba(X_df) >= umbral).astype(int)

    def pesos_en_unidades_originales(self):
        w_raw = self.w / self.sd
        b_raw = self.b - np.sum(self.w * self.mu / self.sd)
        return w_raw, b_raw

neurona = Neurona(["intentos", "distancia_km"]).fit(train, train.y)
w_raw, b_raw = neurona.pesos_en_unidades_originales()
print("\nPesos (estandarizados):", dict(zip(neurona.cols, neurona.w.round(3))), "| b =", round(neurona.b, 3))
print("Pesos en unidades originales:", dict(zip(neurona.cols, w_raw.round(5))), "| b =", round(b_raw, 3))

plt.figure(figsize=(5, 3)); plt.plot(neurona.hist); plt.xlabel("época"); plt.ylabel("pérdida")
plt.title("Curva de aprendizaje"); plt.tight_layout(); plt.savefig("perdida.png", dpi=150); plt.show()

# %% 6. EVALUACIÓN (umbral 0.5)
print("\n=== NEURONA (intentos + distancia), umbral 0.5 ===")
reporte("Entrenamiento", train.y, neurona.predict(train))
reporte("Prueba", test.y, neurona.predict(test))

# %% 7. ¿SIRVE LA TERCERA VARIABLE (MB)? Comparación honesta
# Qué hace: entrena otra neurona con las 3 variables.
# Por qué: justificar con evidencia por qué se usa o se descarta "MB".
n3 = Neurona(["intentos", "distancia_km", "mb"]).fit(train, train.y)
print("\nPesos 3 variables (estandarizados):", dict(zip(n3.cols, n3.w.round(3))))
reporte("3 vars · Entrenamiento", train.y, n3.predict(train))
reporte("3 vars · Prueba", test.y, n3.predict(test))

# %% 8. UMBRAL: se elige SOLO con entrenamiento
# Qué hace: barre umbrales y muestra precisión/sensibilidad en entrenamiento.
# Por qué: elegir el umbral mirando la prueba sería filtración de datos.
print("\numbral | VP FN FP VN | precisión | sensibilidad  (ENTRENAMIENTO)")
for u in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    vp, fn, fp, vn = matriz(train.y, neurona.predict(train, u))
    m = metricas(vp, fn, fp, vn)
    print(f"{u:5.1f}  | {vp:2d} {fn:2d} {fp:2d} {vn:2d} |   {m['precision']:.2f}    |   {m['sensibilidad']:.2f}")

# %% 9. EJEMPLOS CON CASOS REALES, CUENTA A MANO
# Qué hace: muestra z y la salida de dos casos del CSV con los pesos en unidades originales.
# Por qué: la defensa en la mesa es SIN IA; tienen que poder hacer esta cuenta a mano.
def explicar(caso):
    r = df[df.caso == caso].iloc[0]
    z = w_raw[0] * r.intentos + w_raw[1] * r.distancia_km + b_raw
    a = sigmoide(z)
    print(f"Caso {caso}: intentos={r.intentos}, km={r.distancia_km} -> z={z:.3f}, "
          f"a={a:.3f} -> {'SÍ (intrusión)' if a >= .5 else 'NO (normal)'} | real={r.y}")
print()
for c in [20, 16, 19]:
    explicar(c)

# %% 10. EN VIVO: caso nuevo
def predecir(intentos, distancia_km, umbral=0.5):
    z = w_raw[0] * intentos + w_raw[1] * distancia_km + b_raw
    a = sigmoide(z)
    return ("SÍ" if a >= umbral else "NO"), round(float(z), 3), round(float(a), 3)

print("\nEn vivo (13 intentos, 100 km):", predecir(13, 100))
print("En vivo (15 intentos, 7000 km):", predecir(15, 7000))
