# T01 · ¿Este inicio de sesión es una intrusión?

**Integrantes:** Ramiro Garcia Loperena · Agustin Pablo Puzino
**IA declarada:** Claude
**Pista:** Neurona
**Publicación en LinkedIn:** https://lnkd.in/p/daT9Wgur

---

## 1. Formulación y PEAS

**La pregunta.** Para cada inicio de sesión en el servidor de la empresa, ¿es una intrusión? SÍ significa: el inicio de sesión fue realizado con credenciales robadas y constituye un acceso no autorizado.

| PEAS | Su sistema |
|---|---|
| Medida de rendimiento | Sensibilidad alta (detectar la mayor cantidad de intrusiones reales) sin generar falsos positivos excesivos. El objetivo es minimizar los falsos negativos (intrusiones no detectadas) porque una fuga de datos es más costosa que bloquear temporalmente a un empleado legítimo. |
| Entorno | Servidor corporativo que registra cada inicio de sesión. El sistema recibe los datos del evento de login en tiempo real y debe emitir un veredicto antes de que la sesión continúe. Los datos son parcialmente observables (solo se ven 3 variables del evento). |
| Actuadores | Emite una decisión binaria: SÍ (bloquear la sesión y alertar al equipo de seguridad) o NO (permitir el acceso). |
| Sensores | 1) Intentos fallidos previos en la última hora. 2) Distancia al último acceso exitoso (km). 3) Datos descargados en la sesión (MB). |

**Objetivo y métrica.** Lo que de verdad importa es **no dejar pasar intrusiones** (minimizar falsos negativos), porque una intrusión no detectada puede derivar en fuga de datos, daño reputacional y costos legales. La métrica principal es la **sensibilidad** (recall): proporción de intrusiones reales que el sistema detecta correctamente. La exactitud no basta como métrica principal porque el desbalance de clases (solo 3 de 20 casos son intrusiones) hace que un sistema trivial que dice siempre NO alcance ~86% de exactitud con sensibilidad 0%, dejando pasar todas las intrusiones. Sensibilidad y exactitud no son lo mismo: un sistema con 100% de exactitud tiene 100% de sensibilidad, pero un sistema con 86% de exactitud puede tener 0% de sensibilidad.

## 2. El sistema

**Cómo decide.** Es un perceptrón (neurona) con función de activación sigmoide. Recibe dos variables de entrada, las combina linealmente con pesos y sesgo, y aplica la función sigmoide para obtener una probabilidad entre 0 y 1.

**Fórmula:**

```
z = 0.6775 × intentos_fallidos + 0.000731 × distancia_km + (−11.7563)
a = σ(z) = 1 / (1 + e^(−z))
Si a ≥ 0.5 → SÍ (intrusión)
Si a < 0.5 → NO (normal)
```

**Pesos y sesgo explícitos (en unidades originales):**
- w₁ = 0.6775 (peso de intentos fallidos previos)
- w₂ = 0.000731 (peso de distancia al último acceso en km)
- b = −11.7563 (sesgo)

**Interpretación de los pesos:**
- w₁ = 0.6775: cada intento fallido adicional suma 0.6775 al valor z. Es el factor más influyente, lo cual tiene sentido: muchos intentos fallidos previos son la señal más clara de un ataque de fuerza bruta.
- w₂ = 0.000731: cada kilómetro adicional de distancia suma 0.000731 al valor z. Aunque el peso unitario es pequeño, las distancias son grandes (miles de km), así que su contribución total es relevante. Una sesión desde muy lejos del último acceso es sospechosa.
- b = −11.7563: el sesgo es muy negativo, lo que significa que la neurona por defecto se inclina a decir NO (sesión legítima). Hacen falta valores altos en ambas variables para superar el umbral. Esto refleja la realidad: la mayoría de los inicios de sesión son legítimos.

**Variables que usa y por qué.**
- **Intentos fallidos previos (usa):** Es la señal principal de un ataque de fuerza bruta. Los dos casos positivos de entrenamiento (casos 10 y 11) tienen 14 y 15 intentos, muy por encima de la media (6.5).
- **Distancia al último acceso (usa):** Complementa a los intentos. Un login lejano con muchos intentos fallidos es más sospechoso. Los positivos tienen distancias altas (8475 y 6761 km). Pero la distancia sola no basta: el caso 2 tiene 8331 km de distancia y es legítimo (solo 2 intentos).
- **Datos descargados (descarta):** No aporta poder discriminativo. Al momento de decidir si bloquear la sesión, la descarga aún está ocurriendo y su volumen final no se conoce. Usar esta variable violaría el principio de que solo se pueden usar datos disponibles al momento de decidir. Además, los casos positivos (513 y 245 MB) están dentro del rango de los negativos (54–706 MB), por lo que la variable no discrimina.

**Cómo se corre en vivo.** Cuenta a mano con calculadora o en cualquier celular:

1. Anotar los valores: `intentos_fallidos` y `distancia_km` del caso nuevo.
2. Calcular: `z = 0.6775 × intentos + 0.000731 × distancia_km − 11.7563`
3. Calcular: `a = 1 / (1 + e^(−z))`
4. Si `a ≥ 0.5` (equivalente a `z ≥ 0`), la respuesta es **SÍ** (intrusión). Si `a < 0.5`, es **NO** (normal).

Atajo: como el umbral es 0.5 y σ(0) = 0.5, basta con ver si z ≥ 0. Es decir: **si `0.6775 × intentos + 0.000731 × distancia_km ≥ 11.7563`, entonces es intrusión.**

**Dos casos de ejemplo.**

| Caso inventado | Valores | Cálculo | Salida del sistema |
|---|---|---|---|
| A (muchos intentos, lejos) | intentos=16, distancia=9000 km | z = 0.6775×16 + 0.000731×9000 − 11.7563 = 10.84 + 6.58 − 11.76 = 5.66 → a = 0.997 | **SÍ** (intrusión) |
| B (pocos intentos, cerca) | intentos=2, distancia=500 km | z = 0.6775×2 + 0.000731×500 − 11.7563 = 1.36 + 0.37 − 11.76 = −10.04 → a ≈ 0.000 | **NO** (normal) |

## 3. Medición

**Entrenamiento (casos 1–14)**

| | Predice SÍ | Predice NO |
|---|---|---|
| **Real SÍ** | VP = 2 | FN = 0 |
| **Real NO** | FP = 0 | VN = 12 |

Verificación: VP + FN + FP + VN = 2 + 0 + 0 + 12 = 14 ✓

**Detalle caso por caso (entrenamiento):**

| Caso | Intentos | Dist. (km) | z | a = σ(z) | Predicción | Real | Resultado |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 767 | −10.518 | 0.0000 | NO | NO | ✓ VN |
| 2 | 2 | 8331 | −4.308 | 0.0133 | NO | NO | ✓ VN |
| 3 | 12 | 1592 | −2.462 | 0.0786 | NO | NO | ✓ VN |
| 4 | 3 | 796 | −9.142 | 0.0001 | NO | NO | ✓ VN |
| 5 | 9 | 5393 | −1.714 | 0.1526 | NO | NO | ✓ VN |
| 6 | 3 | 7361 | −4.340 | 0.0129 | NO | NO | ✓ VN |
| 7 | 0 | 8420 | −5.598 | 0.0037 | NO | NO | ✓ VN |
| 8 | 6 | 2861 | −5.599 | 0.0037 | NO | NO | ✓ VN |
| 9 | 6 | 5928 | −3.355 | 0.0337 | NO | NO | ✓ VN |
| 10 | 14 | 8475 | +3.927 | 0.9807 | SÍ | SÍ | ✓ VP |
| 11 | 15 | 6761 | +3.351 | 0.9661 | SÍ | SÍ | ✓ VP |
| 12 | 7 | 806 | −6.424 | 0.0016 | NO | NO | ✓ VN |
| 13 | 7 | 4079 | −4.030 | 0.0175 | NO | NO | ✓ VN |
| 14 | 6 | 1934 | −6.277 | 0.0019 | NO | NO | ✓ VN |

**Prueba (casos 15–20)**

| | Predice SÍ | Predice NO |
|---|---|---|
| **Real SÍ** | VP = 1 | FN = 0 |
| **Real NO** | FP = 0 | VN = 5 |

Verificación: VP + FN + FP + VN = 1 + 0 + 0 + 5 = 6 ✓

**Detalle caso por caso (prueba):**

| Caso | Intentos | Dist. (km) | z | a = σ(z) | Predicción | Real | Resultado |
|---|---|---|---|---|---|---|---|
| 15 | 13 | 2688 | −0.983 | 0.2723 | NO | NO | ✓ VN |
| 16 | 14 | 57 | −2.230 | 0.0971 | NO | NO | ✓ VN |
| 17 | 13 | 1119 | −2.131 | 0.1062 | NO | NO | ✓ VN |
| 18 | 15 | 2098 | −0.060 | 0.4851 | NO | NO | ✓ VN |
| 19 | 15 | 5747 | +2.609 | 0.9315 | SÍ | SÍ | ✓ VP |
| 20 | 1 | 6373 | −6.417 | 0.0016 | NO | NO | ✓ VN |

**Métricas**

| Métrica | Entrenamiento | Prueba |
|---|---|---|
| Exactitud | 14/14 = 1.00 (100%) | 6/6 = 1.00 (100%) |
| Precisión | 2/(2+0) = 1.00 (100%) | 1/(1+0) = 1.00 (100%) |
| Sensibilidad | 2/(2+0) = 1.00 (100%) | 1/(1+0) = 1.00 (100%) |
| Especificidad | 12/(12+0) = 1.00 (100%) | 5/(5+0) = 1.00 (100%) |

**Entrenamiento contra prueba.** Las métricas son idénticas en entrenamiento y prueba (100% en las cuatro). Esto sugiere **generalización correcta** y no sobreajuste, por tres razones:

1. **El modelo es simple:** solo tiene 3 parámetros (2 pesos y 1 sesgo), lo que limita su capacidad de memorizar ruido.
2. **Los datos son linealmente separables:** las intrusiones se distinguen claramente por tener muchos intentos fallidos Y distancias grandes simultáneamente. La frontera de decisión (una recta en el plano intentos × distancia) separa ambas clases con margen amplio.
3. **No se usaron los datos de prueba para ajustar nada:** ni los pesos, ni el umbral (se usó el umbral estándar 0.5), ni la selección de variables.

Una precaución: con solo 20 casos (y 3 positivos), los resultados perfectos pueden deberse al tamaño pequeño de la muestra. En producción, habría que validar con más datos.

## 4. El error más caro

**El falso negativo es más caro que el falso positivo.**

- **Falso positivo** (dice SÍ, era NO): Se bloquea a un empleado legítimo. Consecuencia: horas de trabajo perdidas y molestia, pero se resuelve llamando al equipo de IT para desbloquear. El daño es temporal y acotado.
- **Falso negativo** (dice NO, era SÍ): Se deja pasar una intrusión con credenciales robadas. Consecuencia: el atacante puede acceder a datos confidenciales, descargar información sensible, instalar malware o comprometer otros sistemas. Puede generar fuga de datos masiva, sanciones regulatorias, daño reputacional y costos legales. El daño puede ser irreversible.

**¿Qué cambiamos en el sistema para cometer menos ese error?**

1. **Ponderación de clases en el entrenamiento:** Le dimos peso 6× a la clase positiva (intrusión) frente a la negativa, proporcional al desbalance (12 negativos / 2 positivos = 6). Esto penaliza más al modelo cuando clasifica mal una intrusión, empujando los pesos para que la frontera de decisión no pase cerca de los positivos.
2. **Elegimos la sensibilidad como métrica principal**, no la exactitud. Un sistema que dice siempre NO tendría 86% de exactitud pero 0% de sensibilidad: dejaría pasar todas las intrusiones.
3. **Descartamos la variable MB** porque incluirla no mejoraba la sensibilidad y, además, la cantidad de datos descargados podría no estar disponible al momento de tomar la decisión (la descarga aún está en curso).

## 5. ¿Comparado contra qué?

**Línea base trivial.** El sistema más simple es uno que responde siempre **NO** (nunca declara intrusión):

| Conjunto | VP | FN | FP | VN | Exactitud | Sensibilidad |
|---|---|---|---|---|---|---|
| Entrenamiento (14 casos) | 0 | 2 | 0 | 12 | 12/14 = 85.7% | 0/2 = 0% |
| Prueba (6 casos) | 0 | 1 | 0 | 5 | 5/6 = 83.3% | 0/1 = 0% |

La línea base tiene exactitud aceptable gracias al desbalance, pero **sensibilidad 0%**: deja pasar el 100% de las intrusiones. Nuestro sistema supera la línea base en exactitud (100% vs 85.7%) y, más importante, en sensibilidad (100% vs 0%). La diferencia en sensibilidad demuestra que el modelo aprendió a detectar intrusiones, no solo a replicar la clase mayoritaria.

| Checklist (Clase 6) | Respuesta |
|---|---|
| ¿Qué datos? | 20 casos inventados de la ficha T01.csv, con 3 variables por caso (intentos fallidos, distancia al último acceso, MB descargados). División fija: 14 entrenamiento, 6 prueba. 3 intrusiones en total (2 en entrenamiento, 1 en prueba). |
| ¿Qué métrica? | Sensibilidad (recall) como métrica principal, porque lo más importante es no dejar pasar intrusiones. También se reportan exactitud, precisión y especificidad. |
| ¿Comparado contra qué? | Contra la línea base trivial "siempre NO", que alcanza 85.7% de exactitud pero 0% de sensibilidad. Nuestro sistema alcanza 100% en ambas. |
| ¿Quién lo midió? | Nosotros, siguiendo la división entrenamiento/prueba de la ficha sin modificarla. Los datos de prueba no se usaron para ajustar pesos, sesgo ni umbral. |

## 6. Riesgo: el camino completo

**Datos → Modelo → Decisión → Consecuencia**

1. **Datos:** El sistema recibe tres señales de cada inicio de sesión: intentos fallidos previos (última hora), distancia al último acceso (km) y MB descargados. **Riesgo en los datos:** si un atacante conoce el sistema, podría hacer pocos intentos fallidos (ataque lento) o usar una VPN para simular cercanía geográfica, produciendo valores bajos en las dos variables clave. El sistema no detectaría esta intrusión (falso negativo).

2. **Modelo:** Un perceptrón de 2 variables y 3 parámetros, entrenado con solo 14 casos (2 positivos). **Riesgo en el modelo:** con tan pocos ejemplos positivos, el modelo no ha visto diversidad de patrones de intrusión. Puede fallar ante intrusiones de tipo diferente (por ejemplo, un insider que accede desde la oficina pero descarga masivamente).

3. **Decisión:** Si σ(z) ≥ 0.5, el sistema emite SÍ y bloquea la sesión. **Riesgo en la decisión:** el umbral fijo de 0.5 no permite graduar la respuesta. Casos con probabilidad cercana a 0.5 (como el caso 18 con a = 0.485) caen muy cerca del borde y pequeños cambios en los datos los harían caer del otro lado.

4. **Consecuencia:** Un bloqueo (SÍ) detiene al usuario y genera una alerta. Un permiso (NO) deja continuar la sesión. **Riesgo en la consecuencia:** en un sistema real, un falso negativo puede permitir una fuga de datos que afecte a miles de clientes (datos personales, financieros, etc.), con consecuencias legales (ley de protección de datos), multas regulatorias y pérdida de confianza.

**Riesgo concreto:** Un atacante que roba credenciales y usa un dispositivo en la misma ciudad del empleado legítimo (distancia ≈ 0 km) con pocos reintentos de contraseña (por ejemplo, porque ya tiene la contraseña) generaría z muy negativo y el sistema lo dejaría pasar. Para mitigar este riesgo, en producción habría que incorporar variables adicionales (dispositivo, horario, patrón de navegación) y usar el perceptrón como una capa más de un sistema de defensa en profundidad, no como la única barrera.

## 7. Bitácora

**Uso de la IA**

| # | Qué le pedimos | Qué aceptamos | Qué rechazamos y por qué |
|---|---|---|---|
| 1 | Que analizara los datos de T01 y propusiera qué variables usar para el perceptrón. | Aceptamos usar intentos fallidos y distancia como variables principales, porque los casos positivos (10 y 11) se distinguen claramente en esas dos dimensiones. | Rechazamos usar las 3 variables: la IA sugirió incluir MB descargados, pero verificamos que los MB de las intrusiones (513, 245) están dentro del rango normal (54–706) y no discriminan. Además, los MB podrían no estar disponibles al momento de decidir (la descarga está en curso). |
| 2 | Que entrenara un perceptrón con función sigmoide, entropía cruzada ponderada y regularización L2, y nos diera los pesos en unidades originales. | Aceptamos los pesos (w₁=0.6775, w₂=0.000731, b=−11.7563) tras verificar manualmente que clasifican correctamente los 14 casos de entrenamiento. | Rechazamos la primera sugerencia de umbral 0.3 (optimizado en entrenamiento): aunque daba el mismo resultado en entrenamiento, en prueba generaba falsos positivos (casos 15, 17, 18). Elegimos umbral 0.5 (estándar, no optimizado mirando prueba). |
| 3 | Que calculara las matrices de confusión y las cuatro métricas para entrenamiento y prueba. | Aceptamos las matrices y métricas tras verificar caso por caso que cada predicción coincide con el cálculo manual z → σ(z) → decisión. | Nada: los cálculos eran correctos. |
| 4 | Que redactara el README completo siguiendo la plantilla de la cátedra. | Aceptamos la estructura y la redacción base. | Rechazamos la primera versión del análisis de riesgo que era genérica. Pedimos que fuera específico al dominio (seguridad informática, credenciales robadas, VPN) y que incluyera un escenario concreto de fallo. |
| 5 | Que explicara por qué el falso negativo es más caro y qué ajustes se hicieron en consecuencia. | Aceptamos la argumentación (fuga de datos > bloqueo temporal) y los tres ajustes: ponderación de clases, sensibilidad como métrica principal, descarte de MB. | Nada. |

**Corridas contra el evaluador**

| Corrida | Nota | Agujero que encontró | Qué cambiamos |
|---|---|---|---|
| 1 | 9,5 | Completar bitacora | La bitacora fue completada |
