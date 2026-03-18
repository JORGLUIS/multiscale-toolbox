# multiscale-toolbox

Toolbox reusable para procesamiento multiescala de imagenes 2D y volumenes 3D, orientado a la Tarea 1 de Procesamiento Multiescala de Imagenes.

Incluye:

- filtros promedio, gaussianos, ideales y binomiales;
- filtros esfericos para volumenes 3D;
- construccion de piramides gaussianas y laplacianas en 2D y 3D;
- reconstruccion laplaciana;
- upsampling por insercion de ceros + convolucion;
- carga de volumenes NIfTI;
- metricas de reconstruccion y metricas con mascara;
- reponderacion de capas;
- hard threshold y soft threshold;
- utilidades simples de visualizacion 2D y 3D.

## Instalacion local

```bash
pip install -e .
```

## Uso rapido

```python
from pathlib import Path
from multiscale_toolbox import (
    build_laplacian_predictive,
    create_default_filters,
    load_grayscale_image,
    load_nifti_volume,
    masked_correlation,
    psnr,
    reconstruct_laplacian,
)

img = load_grayscale_image(Path("example1.png"))
filters = create_default_filters()
lap, residual, _ = build_laplacian_predictive(img, filters["Gauss sigma=1"], levels=5)
rec = reconstruct_laplacian(lap, residual)
print(psnr(img, rec))

volume, affine, header = load_nifti_volume(Path("unwrapped_seguetotalphase.nii"))
filters_3d = create_default_filters(ndim=3)
lap3d, residual3d, _ = build_laplacian_predictive(volume, filters_3d["Esferico r=2"], levels=5)
background_like = reconstruct_laplacian([], residual3d)
print(background_like.shape, masked_correlation(volume, background_like))
```
