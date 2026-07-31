# CNN en CUDA/C++ para MNIST

Implementación desde cero (sin PyTorch/TensorFlow/cuDNN/cuBLAS) de una red
neuronal convolucional en C++/CUDA, entrenada sobre MNIST. Kernels propios de
GEMM, im2col/col2im, max-pooling, ReLU, dropout, softmax+entropía cruzada y
AdamW. Ver `paper/cnn_paper.tex` para el informe completo con marco teórico,
metodología y resultados.

## Requisitos

- CUDA Toolkit (probado con 13.3) + GPU NVIDIA (probado en RTX 4060, CC 8.9)
- Visual Studio 2022 con integración de CUDA, o cualquier host compiler soportado
- CMake >= 3.20
- Python 3 + numpy + matplotlib (solo para generar las figuras)

## Compilar

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## Entrenar

Desde la raíz del proyecto (usa rutas relativas `data/` y `results/`):

```bash
./build/Release/cnn_train.exe
```

Descarga previa del dataset (si `data/` está vacío):

```bash
curl -fsSL -o data/train-images-idx3-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
curl -fsSL -o data/train-labels-idx1-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
curl -fsSL -o data/t10k-images-idx3-ubyte.gz  https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
curl -fsSL -o data/t10k-labels-idx1-ubyte.gz  https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
gunzip data/*.gz
```

## Generar figuras del paper

```bash
python scripts/plot_training.py
python scripts/plot_confusion.py
python scripts/plot_filters.py
python scripts/plot_predictions.py
```

Las figuras se guardan en `paper/` y son referenciadas por `paper/cnn_paper.tex`.

## Resultados obtenidos (20 épocas, batch 128, RTX 4060 Laptop)

- Parámetros totales: 206,922
- Exactitud de prueba final: 99.26% (máxima 99.28% en época 18)
- Pérdida de prueba final: 0.0260
- Memoria GPU: ~1.17 GiB

## Estructura

```
include/    Headers (kernels, capas, modelo, loader MNIST)
src/        Implementación (.cu/.cpp)
data/       Dataset MNIST (idx-ubyte)
results/    CSVs generados por el entrenamiento (log, matriz de confusión, filtros, activaciones)
scripts/    Scripts Python para generar las figuras
paper/      Informe LaTeX (IEEEtran) y figuras PNG
```
