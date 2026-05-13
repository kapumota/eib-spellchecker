# Cómo presentar la v5

## Mensaje recomendado

**eib-spellchecker v5** es una plataforma de corrección ortográfica para lenguas EIB con arquitectura modular, benchmark reproducible, conservación de investigación heredada y una nueva ruta moderna de entrenamiento en PyTorch.

## Qué mostrar primero

1. demo visual con `artifacts/torch/demo` o un artefacto propio
2. benchmark de un CSV `Input,Output`
3. comparación entre backend `lexical` y `torch-hybrid-reranker`
4. inventario de datasets y corridas heredadas

## Qué resalta bien la v5

- software instalable y testeable
- API y demo visual
- baseline léxico listo
- backend moderno PyTorch
- compatibilidad controlada con el legado TensorFlow/Keras
- reportes reproducibles

## Qué no sobredimensionar

- no vender todos los `.h5` heredados como inferencia garantizada en cualquier entorno moderno
- no llamar “LLM” o “SLM contextual” al sistema si lo que se está mostrando es todavía corrección carácter-a-carácter e híbrida

## Narrativa técnica fuerte

- baseline léxico operativo
- datasets emparejados reales/sintéticos del legado
- variantes de error (`common`, `keyboard`, `syllable`)
- model zoo de investigación preservado
- nueva base moderna `torch-hybrid-reranker`
