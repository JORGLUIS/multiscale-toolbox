# multiscale-toolbox

Toolbox local para procesamiento multiescala de imagenes 2D, orientado a las partes 1 y 2 de la tarea de Procesamiento Multiescala de Imagenes.

Incluye:

- filtros promedio, gaussianos, ideales y binomiales;
- construccion de piramides gaussianas y laplacianas;
- reconstruccion laplaciana;
- upsampling por insercion de ceros + convolucion;
- metricas de reconstruccion;
- reponderacion de capas;
- hard threshold y soft threshold;
- utilidades simples de visualizacion.

## Instalacion local

```bash
pip install -e .
```

## Uso rapido

```python
from pathlib import Path
from multiscale_toolbox import (
    load_grayscale_image,
    create_default_filters,
    build_laplacian_predictive,
    reconstruct_laplacian,
    psnr,
)

img = load_grayscale_image(Path("example1.png"))
filters = create_default_filters()
lap, residual, gauss = build_laplacian_predictive(img, filters["Gauss sigma=1"], levels=5)
rec = reconstruct_laplacian(lap, residual)
print(psnr(img, rec))
```
