# Backends del proyecto

## 1. lexical

Backend base en Python puro.

**Ventajas**
- rápido
- simple
- sin dependencias pesadas
- útil para baseline y demo inmediata

**Límite**
- depende mucho del vocabulario observado
- no aprende a reranquear candidatos con pares supervisados

## 2. torch-hybrid-reranker

Backend moderno agregado en v5.

**Idea**
1. genera candidatos con el backend léxico
2. representa palabra ruidosa y candidato a nivel de caracteres
3. reranquea candidatos con una red en PyTorch

**Ventajas**
- usa pares `Input,Output`
- mantenible
- no depende de TensorFlow
- buena base para seguir hacia modelos más potentes

**Uso recomendado**
- entrenamiento moderno con CSV/TSV emparejados
- demos más fuertes que el baseline léxico
- ruta de evolución futura del proyecto

## 3. legacy-seq2seq

Backend heredado en Keras/TensorFlow.

**Ventajas**
- preserva compatibilidad con parte del trabajo anterior
- permite empaquetar pesos `.h5` heredados cuando aplica

**Límite**
- pipeline más frágil
- depende de TensorFlow
- no siempre reproduce de forma limpia todos los experimentos viejos

## Recomendación práctica

- usa `lexical` para baseline y arranque rápido
- usa `torch-hybrid-reranker` para la versión moderna del proyecto
- usa `legacy-seq2seq` solo cuando necesites conectar resultados históricos
