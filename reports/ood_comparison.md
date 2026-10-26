# Degradación en el test fuera de distribución (OOD)

Caída = pass@1 en test − pass@1 en OOD (positiva: el modelo empeora con problemas de una familia nueva). Intervalos al 95 %: Wilson para cada pass@1 y Newcombe para la caída.

| Modelo | Test | n | OOD | n | Caída (puntos) |
|---|---|---:|---|---:|---|
| sft | 80,5 % [74,5; 85,4] | 200 | 24,0 % [16,7; 33,2] | 100 | +56,5 [45,5; 65,3] |
| grpo_sin_kl | 89,5 % [84,5; 93,0] | 200 | 30,0 % [21,9; 39,6] | 100 | +59,5 [48,7; 68,3] |

## Efecto de volver a puntuar con el verificador actual

Las respuestas guardadas se han puntuado de nuevo, sin volver a generar nada.

| Modelo | Conjunto | Aciertos antes | Aciertos después | Pasan a correctas | Pasan a incorrectas |
|---|---|---:|---:|---:|---:|
| sft | test | 161/200 | 161/200 | 0 | 0 |
| sft | ood | 20/100 | 24/100 | 4 | 0 |
| grpo_sin_kl | test | 179/200 | 179/200 | 0 | 0 |
| grpo_sin_kl | ood | 27/100 | 30/100 | 3 | 0 |

## Aciertos por familia en OOD

| Familia | n | sft | grpo_sin_kl |
|---|---:|---:|---:|
| infusion | 50 | 20/50 (40,0 %) | 27/50 (54,0 %) |
| infusion_inverse | 50 | 4/50 (8,0 %) | 3/50 (6,0 %) |

## Tipo de resultado en OOD

| Modelo | ok | valor_ok_unidad_mal | valor_mal | sin_answer | formato válido |
|---|---:|---:|---:|---:|---:|
| sft | 24 | 3 | 33 | 40 | 60,0 % |
| grpo_sin_kl | 30 | 4 | 31 | 35 | 65,0 % |

## Comparación pareada (mismos problemas)

| Pareja | Conjunto | Solo A acierta | Solo B acierta | Ambos | Ninguno | p |
|---|---|---:|---:|---:|---:|---:|
| A = sft, B = grpo_sin_kl | test | 6 | 24 | 155 | 15 | 0,001 |
| A = sft, B = grpo_sin_kl | ood | 9 | 15 | 15 | 61 | 0,307 |

p: test de signos exacto bilateral sobre los problemas que solo uno de los dos resuelve; un valor por debajo de 0,05 indica una diferencia que el azar explica mal.

## Cautelas

- Cada conjunto tiene pocos problemas: un intervalo que contiene el 0 significa que la caída no se distingue del ruido.
- El conjunto OOD cambia de familia y puede cambiar también de dificultad (nivel); si es así, la caída mezcla ambos efectos. Comprobadlo en `rlm/data/test_ood.jsonl`.
- Es una única generación greedy por problema: no mide la variabilidad entre ejecuciones ni entre semillas de entrenamiento.
