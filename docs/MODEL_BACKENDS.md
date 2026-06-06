### Backends del proyecto en v0.6

La plataforma usa una arquitectura **multibackend**: la interfaz de uso permanece estable, mientras que el mecanismo de corrección puede variar según el artefacto.

#### 1. `lexical`

Backend base en Python puro, orientado a recuperación aproximada sobre vocabulario observado.

##### Ventajas
- rápido,
- interpretable,
- sin dependencias pesadas,
- ideal para baseline, pruebas rápidas y demos inmediatas.

##### Límites
- depende fuertemente de cobertura léxica,
- maneja peor vocabulario abierto,
- no reranquea candidatos usando supervisión emparejada.

##### Uso recomendado
- baseline inicial,
- comparación rápida,
- despliegue ligero,
- primeros benchmarks por idioma.

#### 2. `subword`

Backend agregado y consolidado en v0.6. Usa subunidades derivadas de n-gramas de caracteres para reducir dependencia de vocabulario cerrado.

#### Ventajas
- mejor comportamiento ante OOV,
- útil para palabras largas o morfológicamente complejas,
- mejor primer paso hacia open-vocabulary corrección,
- no requiere framework pesado.

#### Límites
- no sustituye todavía un reranker contextual completo,
- su ganancia depende de la calidad del vocabulario/subunidades observadas.

#### Uso recomendado
- robustez con palabras nuevas,
- demos donde importa menos la dependencia a vocabulario exacto,
- evaluación de sobrecorrección y open vocabulary.

#### 3. `torch-hybrid-reranker`

Backend moderno en PyTorch.

##### Idea
1. genera candidatos con una base léxica,
2. representa palabra ruidosa y candidato a nivel de caracteres,
3. reranquea candidatos con una red en PyTorch.

##### Ventajas
- aprovecha pares `Input,Output`,
- ruta moderna de evolución del proyecto,
- no depende de TensorFlow,
- buena base para futuros modelos más potentes.

##### Límites
- requiere PyTorch,
- sigue siendo más costoso que `lexical`,
- no es todavía un modelo contextual grande.

#### Uso recomendado
- entrenamiento moderno con CSV/TSV emparejados,
- demos más fuertes que el baseline léxico,
- experimentos orientados a mejora incremental supervisada.

#### 4. `legacy-seq2seq`

Backend heredado en TensorFlow/Keras.

#### Ventajas
- preserva compatibilidad con parte del trabajo anterior,
- permite conectar con pesos `.h5` y experimentos históricos.

#### Límites
- pipeline más frágil,
- dependencia de TensorFlow/Keras,
- algunos experimentos antiguos no son portables de forma limpia.

#### Uso recomendado
- preservación del legado,
- comparación histórica,
- recuperación de resultados previos cuando aplica.

#### Recomendación práctica

- usa `lexical` para baseline y arranque rápido,
- usa `subword` cuando quieras una ruta más robusta frente a OOV,
- usa `torch-hybrid-reranker` cuando ya tengas pares supervisados y quieras una base moderna,
- usa `legacy-seq2seq` solo cuando necesites conectar con investigación heredada.

#### Perspectiva de evolución

La arquitectura futura del proyecto apunta a combinaciones híbridas entre:

- candidatos léxicos,
- modelado carácter/subpalabra,
- reranking contextual,
- políticas explícitas de abstención.
