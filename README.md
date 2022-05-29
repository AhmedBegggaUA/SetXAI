<img width="600" alt="image" src="https://user-images.githubusercontent.com/60975511/170879500-dd060150-5ea2-47b2-aa17-d1fbf3cd4366.png">
# SetXAI

Este trabajo nace del reto que supone enfrentarse a las  redes neuronales invariantes a permutación y tamaño de entrada, que hasta la fecha es algo novedoso y poco explorado debido a su complejidad computacional. Sin embargo, estas redes neuronales son muy expresivas y hacen que sea posible trabajar a nivel de conjuntos (p.e. nubes de puntos 2D y 3D, tramas de vídeo, grafos, etc).

Nuestra labor en este proyecto será la aportación de un framework que nos permita entender el funcionamiento de estas arquitecturas, desde la codificación los datos, hasta qué operaciones invariantes son las más efectivas en la actualidad. Por ello, se incluirá, aparte del código fuente,  una serie de tutoriales notebooks con las diferentes variantes de estas arquitecturas y distintos conjuntos de datos. Cabe mencionar que en los tutoriales abordaremos tanto tareas de clasificación como de regresión con cada una de las variantes de las arquitecturas. Por último, aportaremos un método explicativo especifico para estas arquitecturas: SetXAI, cuyo funcionamiento podrá ser tanto estático como dinámico (por épocas).SetXAI obtiene aquellos puntos de mayor criticidad, es decir, aquellos que nos aporten mayor información para la clasificación o regresión. Nuestros experimentos aportarán evidencia acerca de la capacidad de los puntos críticos para resumir los conjuntos de entrada y realizar por sí solos la tarea de clasificación o regresión.
<img width="732" alt="image" src="https://user-images.githubusercontent.com/60975511/170883475-6c6f0a90-37f2-472d-b840-a404c77c672c.png">


## Dependencias

Creación de un entorno con los paquetes necesarios.
```
conda env create -f tfg.yml
conda activate tfg
```
## Organización del repositorio
Documentación : Documentos esenciales para la teoría de Deep Set
Tutoriales : Recopilación de tutoriales para el entendimiento de los DeepSet, además de los puntos críticos
data : Directorio que contiene los datos, para poder hacer uso de ModelNet10, se ha de descargar el fichero http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
runs : Directorio que contiene los logs de los experimentos
scripts : Directorio que contiene los modelos entrenados, además de los scripts del servidor
src : Directorio con todo el código fuente de los DeepSet
visualisation : Directorio con notebooks para previsualizar los datasets usados
   
## Ejecutar experimentos
```
cd src/
./experimentos.sh
```
## Visualizar experimentos con TensorBoard
```
tensorboard --logdir runs
```

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-latex](https://img.shields.io/badge/Made%20with-LaTeX-1f425f.svg)](https://www.latex-project.org/)
