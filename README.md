# multiscale-toolbox

Toolbox local para procesamiento multiescala de imagenes 2D, extraido y ordenado a partir de `Tarea1_P1y2_Jorge_Medina.ipynb`.

Incluye:

- filtros promedio, gaussianos, ideales y binomiales;
- construccion de piramides gaussianas y laplacianas;
- reconstruccion laplaciana;
- upsampling por interpolacion lineal o por insercion de ceros + convolucion;
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

## Estado respecto a GitHub

El repositorio quedo listo localmente. En esta sesion no tengo `gh` ni una autenticacion activa contra GitHub, asi que no pude crear ni subir el repo remoto automaticamente.
