```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier # modelo knn
from sklearn.model_selection import train_test_split # devidir datos training, testeo
from sklearn.linear_model import LinearRegression # regresion lineal
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format}) # No mostrar en notacion
pd.set_option('display.float_format', lambda x: f'{x:.2f}') # No mostrar en notacion
```

# Clasificacion

**Aprendizaje No Supervisado**: es el proceso de descubrir patrones y estructuras ocultas a partir de datos no etiquetados, ejemplo:

una empresa puede desear agrupar a sus clientes en categorias dsitintas en funcion del comportamiento de compra sin saber de antemano cuales son las categorias. esto se conoce como (clustering).

**Aprendizaje Supervisado**: en este tipo de aprendizaje los valores que se vana predecir ya son conocidos, y se construye un modelo con el objetivo de predecir con precision valores de datos nunca antes visto. Utiliza caracteristicas (predictores, feature, caracteristica, variable independiente) para predecir el valor de una variable objetivo (variable dependiente, variable objetivo, variable respuesta). Hay dos tipos de aprendizaje supervizado:

    - Clasificacion: se utiliza para predecir la etiqueta o categoria de una observacion, por ejemplo predecir para clasificar si una transaccion bancaria es fraudulenta o no. El resultado seria:

            - Transaccion fraudulenta o
            - Transaccion no fraudulenta

      Esto se conoce como (clasificion binaria)

     - Regresion: La regresion se utiliza para predecir valores continuos, por aejemplo un modelo que pueda utlizar caracteristicas (predictores) como el numero de habitantes y el tamano de la propiedad, para predecir el precio de la vivienda (variable dependiente). 

Antes de realizar un modelo de apredizaje supervisado, NO debe haber:

    - Valores faltantes: hacer EDA para conocer data
    - Variables deben estar en formato numerico
    - El formato de almacenamiento debe ser: DataFrame o Series de pandas, o matrices numpy.
scikit-learn syntax:

from sklearn.module import Model
model = Model()
model.fit(X, y) 

El modelo se ajusta a: 

    'X' una matriz de nuestras caracteristicas
    'y' una matriz de nuestros valores de variables dependiente

predictions = model,predict(X_new)

Por ejemplo, si se intreducen caracteristicas para 6 correos electronicos en un modelo de clasificacion de spam , se devuelve una matriz de 6 valores:

array([0,0,0,0,1,0])
1 -> si es spam
0 -> no es spam

Quedando el modelo asi:

from sklearn.module import Model
model = Model()
model.fit(X, y)
predictions = model,predict(X_new)
## Clasificar etiquetas 

1. Definimos un modelo de clasificacion.
2. El modelo aprendera de datos etiquetados que le pasamos "que ya tiene etiquetas".
3. Pasamos datos sin etiquetar al modelo como entrada.
4. El modelo predice las etiquetas.

### Knn - K nearest neighbors
knn es popular para problemas de clasificacion, knn predice la etiqueta de cualquier punto de datos observador en las observaciones k. Ejemplo:

Como se clasifica la observacion "negra":

![nombreImagen](http://localhost:8888/files/Downloads/Python/MLS/1_Supervised_Learning_with_scikit_learn/1.jpg?_xsrf=2%7C2d083ae1%7C3457333de4e74ce098bbb308c43d362c%7C1721828162)

si k=3 buscara los tres vecinos mas cercanos y del que haya mas observaciones lo clasifica con esa etiqueta: en este caso "rojo"

![nombreImagen](http://localhost:8888/files/Downloads/Python/MLS/1_Supervised_Learning_with_scikit_learn/2.jpg?_xsrf=2%7C2d083ae1%7C3457333de4e74ce098bbb308c43d362c%7C1721828162)

si k=5 el resultado de la etiqueta es "azul".

![nombreImagen](http://localhost:8888/files/Downloads/Python/MLS/1_Supervised_Learning_with_scikit_learn/3.jpg?_xsrf=2%7C2d083ae1%7C3457333de4e74ce098bbb308c43d362c%7C1721828162)

### Ejercicio:

"account_length": indica lealtad del cliente.

"customer_service_calls": las llamadas frecuentes al servicio de atención al cliente pueden indicar insatisfacción.


```python
# Datos:
telecom = pd.read_csv("C:/Users/alejo/Downloads/Python/MLS/1_Supervised_Learning_with_scikit_learn/telecom_churn_clean.csv")
telecom.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>account_length</th>
      <th>area_code</th>
      <th>international_plan</th>
      <th>voice_mail_plan</th>
      <th>number_vmail_messages</th>
      <th>total_day_minutes</th>
      <th>total_day_calls</th>
      <th>total_day_charge</th>
      <th>total_eve_minutes</th>
      <th>total_eve_calls</th>
      <th>total_eve_charge</th>
      <th>total_night_minutes</th>
      <th>total_night_calls</th>
      <th>total_night_charge</th>
      <th>total_intl_minutes</th>
      <th>total_intl_calls</th>
      <th>total_intl_charge</th>
      <th>customer_service_calls</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>128</td>
      <td>415</td>
      <td>0</td>
      <td>1</td>
      <td>25</td>
      <td>265.10</td>
      <td>110</td>
      <td>45.07</td>
      <td>197.40</td>
      <td>99</td>
      <td>16.78</td>
      <td>244.70</td>
      <td>91</td>
      <td>11.01</td>
      <td>10.00</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>107</td>
      <td>415</td>
      <td>0</td>
      <td>1</td>
      <td>26</td>
      <td>161.60</td>
      <td>123</td>
      <td>27.47</td>
      <td>195.50</td>
      <td>103</td>
      <td>16.62</td>
      <td>254.40</td>
      <td>103</td>
      <td>11.45</td>
      <td>13.70</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>137</td>
      <td>415</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>243.40</td>
      <td>114</td>
      <td>41.38</td>
      <td>121.20</td>
      <td>110</td>
      <td>10.30</td>
      <td>162.60</td>
      <td>104</td>
      <td>7.32</td>
      <td>12.20</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>84</td>
      <td>408</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>299.40</td>
      <td>71</td>
      <td>50.90</td>
      <td>61.90</td>
      <td>88</td>
      <td>5.26</td>
      <td>196.90</td>
      <td>89</td>
      <td>8.86</td>
      <td>6.60</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>75</td>
      <td>415</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>166.70</td>
      <td>113</td>
      <td>28.34</td>
      <td>148.30</td>
      <td>122</td>
      <td>12.61</td>
      <td>186.90</td>
      <td>121</td>
      <td>8.41</td>
      <td>10.10</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Definimos variables:
from sklearn.neighbors import KNeighborsClassifier
X = telecom[['account_length', 'customer_service_calls']].values
y = np.ravel(telecom[['churn']].values)

# X: es una matriz 2D de nuestros predictores
# y: es una matriz 1D de nuestra variable dependiente
# .values: convertir las variables en matrices numpy
# np.ravel(): y está en forma de un vector columna, pero scikit-learn espera un array unidimensional. 
#             Para solucionar esto, puedes aplanar y usando el método ravel()
print(X.shape, y.shape)
```

    (3333, 2) (3333,)
    


```python
# modelo de clasificacion con scikit-learn:
telecom_knn = KNeighborsClassifier(n_neighbors=6) # Definimos el modelo
telecom_knn.fit(X, y)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KNeighborsClassifier(n_neighbors=6)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;KNeighborsClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">?<span>Documentation for KNeighborsClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>KNeighborsClassifier(n_neighbors=6)</pre></div> </div></div></div></div>




```python
# Data para hacer predicciones:
X_new = np.array([[30.0, 17.5],
                 [107.0, 24.1],
                 [213.0, 10.9]])
print(X_new.shape)
```

    (3, 2)
    


```python
# Hacer predicciones de etiquetas de nuevos datapoints:
telecom_knn_pred = telecom_knn.predict(X_new)
```

    1 -> Si abandono
    0 -> No abandono


```python
# Imprimier prediccion:
print('Predicciones: {}'.format(telecom_knn_pred))
```

    Predicciones: [0 1 0]
    

El modelo ha predicho que el primer y el tercer cliente no abandonarán la nueva matriz. Pero, ¿cómo sabemos cuán precisas son estas predicciones?

### Medición del rendimiento del modelo

Accuracy: predicciones correctas / totalobservaciones

Para poder medir la precision del modelo debemos realizar split de los datos en: training y test:

    - training: ajuste del modelo de clasificacion
    - test: Calculamos la precision del modelo con la data de test

![nombreImagen](http://localhost:8888/files/Downloads/Python/MLS/1_Supervised_Learning_with_scikit_learn/4.jpg?_xsrf=2%7C2d083ae1%7C3457333de4e74ce098bbb308c43d362c%7C1721828162)


Para realizar los split debemos descargar:
from sklearn.model_selection import train_test_split

- modelo:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3,
                                                    random_state= 21, stratify=y)

- explicacion:
# random_state: crea una semilla para un generador de numeros aleatorios que divide los datos
# stratify= y: asegura que la división de los datos en conjuntos de entrenamiento y prueba mantenga la misma proporción de clases que en el conjunto de datos original. Esto es especialmente útil cuando tienes un conjunto de datos desequilibrado, ya que garantiza que ambas divisiones (entrenamiento y prueba) tengan una representación proporcional de cada clase.

Por ejemplo, si tu variable objetivo `y` tiene un 70% de clase A y un 30% de clase B, al usar stratify= y, tanto el conjunto de entrenamiento como el de prueba mantendrán aproximadamente estas proporciones.

train_test_split devuelve cuatro metricas:

    - datos de entrenamiento - traning -> X_train
    - datos de prueba - test -> X_test
    - etiquetas de entrenamiento -> y_train
    - etiquetas de prueba -> y_test

Que se decomponen en: X_train, X_test, y_train, y_test

- modelo:
knn = KNeighborsClassifier(n_neighbors= 6) # se crea instancia del modelo
knn.fit(X_train, y_train) # Se ajusta el modelo de entrenamiento
print(knn.score(X_test, y_test)) # Comprobar precision del modelo

- explicacion:
0.8800303494305894985

la precision del modelo es del 88%, lo cual es una proporcion 9 a 1, de 10 etiquetas generadas 9 son buenas y 1 esta mal clasificada.
Como interpretal un knn

![xxx](http://localhost:8888/files/Downloads/Python/MLS/1_Supervised_Learning_with_scikit_learn/5.jpg?_xsrf=2%7C2d083ae1%7C3457333de4e74ce098bbb308c43d362c%7C1721828162)

**limite de decision**: 
umbral que determina que etiqueta aigna un modelo a una observacion, es la linea que separa los dos grupos en este caso.

Como se puede observar en la imagen, a medida que aumenta k, el limite de decision es menos efectivo por observaciones individuales, lo que refleja que un modelo mas simple es mejo.

Modelos mas simples -> los modelos mas simples son MENOS capaces de detectar relaciones en el conjunto de datos, lo que se conoce como **subajuste** o **underfitting**.

Modelos mas comlejos -> por el contratio los modelos mas complejos pueden ser sencibles al ruido en los datos de entrenamiento, en lugar de reflejar tendencias generales, lo que se conoce como **sobreajuste** o **overfitting**
Curva de complejidad del modelo y over/underfitting:

- 1modelo:
train_accuracies = {}
test_accuracies = {}

- explicacion:
Crear direcciones vacias para almacenar precision de datos de entrenamiento y testeo

- 2modelo:
neighbord = np.arange(1, 26)

- explicacion:
matriz que contiene el rango de valores k, utilizamos un blucle for para repetir nuestro flujo de trabajo anterior, utilizando difenrentes valores k.

```python
neighbords = np.arange(1, 26)
print(neighbords)
```

    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25]
    
- 3modelo
for neighbord in neighbords:
    knn = KNeighborsClassifier(n_neighbors= neighbord) # se toma el array de 2modelo
    knn.fit(X_train, y_train) # se ajusta la iteracion a los datos en entrenamiento
    train_accuracies[neighbord] = knn.score(X_train, y_train) # precision de datos de entrenamiento 1modelo
    test_accuracies[neighbord] = knn.score(X_test, y_test) # precision de datos de teste 1modelo
- explicacion:
Recorremos la matriz de vecinos y, dentro del bucle, instanciamos un modelo knn = a los datos de entrenamiento 'X_train' y 'y_train'.Graficando los resultados:

plt.figure(figsize=(8,6))
plt.title('KNN: Varying Number of Neighbors')
plt.plot(neighbord, train_accuracies.values(), label= 'Training Accuracy')
plt.plot(neighbord, test_accuracies.values(), label= 'Testing Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
![xxx](http://localhost:8888/files/Downloads/Python/MLS/1_Supervised_Learning_with_scikit_learn/6.jpg?_xsrf=2%7C2d083ae1%7C3457333de4e74ce098bbb308c43d362c%7C1721828162)

A medida que aumenta k mas de 15, vemos un subajuste en el rendimiento del modelo, tanto en el conjunto de entrenamiento como en el conjunto de testeo. La maxima precision se alcanza con k= 13, que es hasta donde sllega  a su maximo la linea de la grafica de Testing Accuracy.

### Ejercicio:


```python
# Datos:
telecom = pd.read_csv("C:/Users/alejo/Downloads/Python/MLS/1_Supervised_Learning_with_scikit_learn/telecom_churn_clean.csv")
telecom.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>account_length</th>
      <th>area_code</th>
      <th>international_plan</th>
      <th>voice_mail_plan</th>
      <th>number_vmail_messages</th>
      <th>total_day_minutes</th>
      <th>total_day_calls</th>
      <th>total_day_charge</th>
      <th>total_eve_minutes</th>
      <th>total_eve_calls</th>
      <th>total_eve_charge</th>
      <th>total_night_minutes</th>
      <th>total_night_calls</th>
      <th>total_night_charge</th>
      <th>total_intl_minutes</th>
      <th>total_intl_calls</th>
      <th>total_intl_charge</th>
      <th>customer_service_calls</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>128</td>
      <td>415</td>
      <td>0</td>
      <td>1</td>
      <td>25</td>
      <td>265.10</td>
      <td>110</td>
      <td>45.07</td>
      <td>197.40</td>
      <td>99</td>
      <td>16.78</td>
      <td>244.70</td>
      <td>91</td>
      <td>11.01</td>
      <td>10.00</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>107</td>
      <td>415</td>
      <td>0</td>
      <td>1</td>
      <td>26</td>
      <td>161.60</td>
      <td>123</td>
      <td>27.47</td>
      <td>195.50</td>
      <td>103</td>
      <td>16.62</td>
      <td>254.40</td>
      <td>103</td>
      <td>11.45</td>
      <td>13.70</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>137</td>
      <td>415</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>243.40</td>
      <td>114</td>
      <td>41.38</td>
      <td>121.20</td>
      <td>110</td>
      <td>10.30</td>
      <td>162.60</td>
      <td>104</td>
      <td>7.32</td>
      <td>12.20</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>84</td>
      <td>408</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>299.40</td>
      <td>71</td>
      <td>50.90</td>
      <td>61.90</td>
      <td>88</td>
      <td>5.26</td>
      <td>196.90</td>
      <td>89</td>
      <td>8.86</td>
      <td>6.60</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>75</td>
      <td>415</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>166.70</td>
      <td>113</td>
      <td>28.34</td>
      <td>148.30</td>
      <td>122</td>
      <td>12.61</td>
      <td>186.90</td>
      <td>121</td>
      <td>8.41</td>
      <td>10.10</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### Modelo: 

Divida "X" e "y" en conjuntos de entrenamiento y prueba, estableciendo test_size en 20 %, random_state en 42 y garantizando que las proporciones de la etiqueta de destino reflejen las del conjunto de datos original.

Ajuste un modelo y compruebe su precision.


```python
# Ajustado data para el modelo:

X= telecom.drop('churn', axis=1).values
# telecom.drop('churn', axis=1): Elimina la columna llamada 'churn' del DataFrame telecom. 
# El parámetro axis=1 indica que se debe eliminar una columna (si fuera axis=0, se eliminaría una fila)
# Crea la matriz con todas los predictores, menos la variable respuesta churn

y= telecom['churn'].values
# Crea la variable respuesta solo tomando los valores de dicha variable
```


```python
print(X)
```

    [[0.000000 128.000000 415.000000 ... 3.000000 2.700000 1.000000]
     [1.000000 107.000000 415.000000 ... 3.000000 3.700000 1.000000]
     [2.000000 137.000000 415.000000 ... 5.000000 3.290000 0.000000]
     ...
     [3330.000000 28.000000 510.000000 ... 6.000000 3.810000 2.000000]
     [3331.000000 184.000000 510.000000 ... 10.000000 1.350000 2.000000]
     [3332.000000 74.000000 415.000000 ... 4.000000 3.700000 0.000000]]
    


```python
print(y)
```

    [0 0 0 ... 0 0 0]
    


```python
# Realizar split y clasificar datos de entrenamiento y testeo:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2,
                                                    random_state= 42, stratify=y)
```


```python
# Ajusta modelo
telecom_knn = KNeighborsClassifier(n_neighbors=5)
```


```python
# Ajusta ttelecom_knn a datos de entrenamiento:
telecom_knn.fit(X_train,y_train)
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KNeighborsClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;KNeighborsClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">?<span>Documentation for KNeighborsClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>KNeighborsClassifier()</pre></div> </div></div></div></div>




```python
# Comprobar la precision del modelo:
print(telecom_knn.score(X_test, y_test))
```

    0.8545727136431784
    

#### Overfitting y Subfitting:

Interpretar la complejidad del modelo es una excelente manera de evaluar el rendimiento del aprendizaje supervisado. El objetivo es producir un modelo que pueda interpretar la relación entre las características y la variable objetivo, así como generalizar bien cuando se exponga a nuevas observaciones.


```python
# Creando validacion de neighbords
vecinos  = np.arange(1, 20)
print(vecinos)
```

    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    


```python
# Creando diccionarios para almacenar precision de training y testing
train_accuracies= {}
test_accuracies= {}
```


```python
# Creando bucle para hacer validacion de modelos con k = vecinos
for vecino in vecinos:
    telecom_knn= KNeighborsClassifier(n_neighbors= vecino) # modelo de clasificaicon
    telecom_knn.fit(X_train, y_train) # Ajuste del modelo a datos de training
    train_accuracies[vecino]= telecom_knn.score(X_train, y_train) # precision train
    test_accuracies[vecino]= telecom_knn.score(X_test, y_test) # precision test
print(vecinos, '\n', train_accuracies, '\n', test_accuracies)
```

    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19] 
     {1: 1.0, 2: 0.8885971492873218, 3: 0.8994748687171793, 4: 0.8750937734433608, 5: 0.878469617404351, 6: 0.8660915228807202, 7: 0.8705926481620405, 8: 0.8615903975993998, 9: 0.86384096024006, 10: 0.858589647411853, 11: 0.8604651162790697, 12: 0.8574643660915229, 13: 0.858589647411853, 14: 0.8567141785446362, 15: 0.858589647411853, 16: 0.8574643660915229, 17: 0.8582145536384096, 18: 0.8567141785446362, 19: 0.8570892723180795} 
     {1: 0.7856071964017991, 2: 0.8470764617691154, 3: 0.8320839580209896, 4: 0.856071964017991, 5: 0.8545727136431784, 6: 0.8590704647676162, 7: 0.8605697151424287, 8: 0.8620689655172413, 9: 0.863568215892054, 10: 0.8605697151424287, 11: 0.8605697151424287, 12: 0.8605697151424287, 13: 0.8605697151424287, 14: 0.8620689655172413, 15: 0.8620689655172413, 16: 0.8590704647676162, 17: 0.8605697151424287, 18: 0.856071964017991, 19: 0.8575712143928036}
    
# Bucle almacenando los datos en un dataframe

# Lista para almacenar resultados
telecom_knn_result = []

# Bucle
for vecino in vecinos:
    telecom_knn= KNeighborsClassifier(n_neighbors= vecino) # modelo de clasificaicon
    telecom_knn.fit(X_train, y_train) # Ajuste del modelo a datos de training
    train_accuracies[vecino]= telecom_knn.score(X_train, y_train) # precision train
    test_accuracies[vecino]= telecom_knn.score(X_test, y_test) # precision test

    # Almacenando resultados en un diccionario:
    telecom_knn_result.append({
        'vecino': vecino,
        'train_accuracy': train_accuracies,
        'test_accuracy': test_accuracies
    })

# Conviertiendo lista de diccionario en dataframe
df_telecom_knn_result = pd.DataFrame(telecom_knn_result)
print(df_telecom_knn_result.head())
#### Resultado:

¿Observa cómo la precisión del entrenamiento disminuye a medida que aumenta el número de vecinos inicialmente y viceversa para la precisión de la prueba? Estos puntajes serían mucho más fáciles de interpretar en un gráfico de líneas, así que generemos una curva de complejidad del modelo a partir de estos resultados.


```python
#Graficando los resultados:
plt.title('KNN: # optimo de vecinos')
plt.plot(vecinos, train_accuracies.values(), label= 'Training Accuracy')
plt.plot(vecinos, test_accuracies.values(), label= 'Testing Accuracy')
plt.legend()
plt.xlabel('Numero de vecinos')
plt.xlabel('Accuracy')
plt.show()
```


    
![png](output_40_0.png)
    


Observa cómo la precisión del entrenamiento disminuye y la precisión de la prueba aumenta a medida que aumenta la cantidad de vecinos. Para el conjunto de prueba, la precisión alcanza su punto máximo con 7 vecinos, lo que sugiere que es el valor óptimo para nuestro modelo.

# Regresion
Por lo general las variables respuesta de la regresion son variables continuas, se valida un modelo de niveles de diabetes en mujeres:

pregnancies: numero de embarazos
glucose: nivel de glucosa en sangre
triceps: pliegues cutaneos del triceps
insulin: nivel de insulina
bmi: indice de masa corporal
dpf:
diabetes: 1- Si diabetes, 0- No diabetes



```python
diab = pd.read_csv("C:/Users/alejo/Downloads/Python/MLS/1_Supervised_Learning_with_scikit_learn/diabetes_clean.csv")
print(diab.head())
```

       pregnancies  glucose  diastolic  triceps  insulin   bmi  dpf  age  diabetes
    0            6      148         72       35        0 33.60 0.63   50         1
    1            1       85         66       29        0 26.60 0.35   31         0
    2            8      183         64        0        0 23.30 0.67   32         1
    3            1       89         66       23       94 28.10 0.17   21         0
    4            0      137         40       35      168 43.10 2.29   33         1
    


```python
# seleccionamos las variables para X, y
X= diab.drop('glucose', axis=1).values # Todas las variables menos glucosa
y= diab['glucose'].values # Solo glucosa
print(type(X), type(y)) # verificar que ambas variabels sean matrices numpy
```

    <class 'numpy.ndarray'> <class 'numpy.ndarray'>
    


```python
# Intentar precedir los niveles de glucosa en sangre
# a partir de una unica caracteristica: bmi
X_bmi= X[:, 4] # Seleccionar columna
print(y.shape, X_bmi.shape) # Verificar matrices
```

    (768,) (768,)
    


```python
# Para y  esta bien que sea una matriz unidimensional, pero para X_bmi la matriz debe estar formateada
# como una matriz bidimensional, sino, scikit-learn no la acepta
# en el paso 1 X_bmi esta "[]", con este ajuste queda "[[]]"
X_bmi= X_bmi.reshape(-1, 1)
print(X_bmi.shape)
```

    (768, 1)
    


```python
# graficamos para observar relacion de variables
plt.scatter(X_bmi, y)
plt.ylabel('Glucosa en sangre')
plt.xlabel('bmi')
plt.show()
```


    
![png](output_47_0.png)
    



```python
# Modelo
from sklearn.linear_model import LinearRegression # regresion lineal
reg= LinearRegression() # modelo
reg.fit(X_bmi, y) # ajuste de variables al modelo
predictions= reg.predict(X_bmi) # realizar predicciones

# Grafico
plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions)
plt.ylabel('Glucosa en sangre')
plt.xlabel('bmi')
plt.show()
```


    
![png](output_48_0.png)
    


La linea azul, representa el ajuste del modelo de regresion lineal de los valores de y= Glucosa en sangre dado el bmi, que parece tener una correlacion positiva debil.

## Ejercicio
 sales_d:, que contiene información sobre los gastos de las campañas publicitarias en diferentes tipos de medios y la cantidad de dólares generados en ventas para la campaña respectiva:
sales_df = pd.read_csv("C:/Users/alejo/Downloads/Python/MLS/1_Supervised_Learning_with_scikit_learn/advertising_and_sales_clean.csv")
print(sales_df.head())
Utilizará los gastos de publicidad como predictores para predecir los valores de ventas, trabajando inicialmente con la columna "radio". Sin embargo, antes de realizar predicciones, deberá crear las matrices de características y objetivos, y modificarlas para que tengan el formato correcto para scikit-learn.


```python
# Crando varianles
X= sales_df['radio'].values
y= sales_df['sales'].values

# Volviendo matriz bidimensional X
X= X.reshape(-1, 1)
print(X.shape, y.shape)
```

    (4546, 1) (4546,)
    


```python
# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X, y)

# Make predictions
predictions = reg.predict(X)

print(predictions[:5])
```

    [95491.171191 117829.510384 173423.380715 291603.114442 111137.281671]
    
Vea cómo los valores de venta de las primeras cinco predicciones varían de $95,000 a más de $290,000. Visualicemos cómo se ajusta el modelo.Ahora que ha creado su modelo de regresión lineal y lo ha entrenado utilizando todas las observaciones disponibles, puede visualizar qué tan bien se ajusta el modelo a los datos. Esto le permite interpretar la relación entre el gasto en publicidad radial y los valores de ventas.

Las variables X, una matriz de valores de radio, y, una matriz de valores de ventas, y predicciones, una matriz de valores predichos del modelo para y dado X

```python
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Create scatter plot
plt.scatter(X, y, color="blue")

# Create line plot
plt.plot(X, predictions, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()
```


    
![png](output_57_0.png)
    

El modelo captura muy bien una correlación lineal casi perfecta entre el gasto en publicidad en radio y las ventas. Ahora veamos qué sucede en segundo plano para calcular esta relación.
### Como funciona la regresion lineal:

Maquinas de regresion:

                                    y= ax + b

    y= target, variable obsjetivo, variable respuesta
    x= feature, caracteristica, predictor, variable independiente
    a, b= paramentros/coeficientes del modelo, slope, intercept

Como elegimos la precision de los valores a y b?
    definiendo una funcion de error para cualquier liena dada 
    luego elegir la liena que minimiza la funcion

La funcion de error, tambien se denomina funcion de perdida o de coste.

![xxx](http://localhost:8888/files/Downloads/Python/MLS/1_Supervised_Learning_with_scikit_learn/7.jpg?_xsrf=2%7C2d083ae1%7C3457333de4e74ce098bbb308c43d362c%7C1721828162)

La distancia vertical entre la observacion y la recta es el residual, podemos intertar minimizar la suma de los residuales, para evitar que los residuales positivos se canceles con los negativos, elevanmos al cuadrado los residuales. RSS es el calculo de todos los residuales al cuadrado.

Este tipo de minimizacio se conoce como MCO o minimos cuadrados ordinarios donde el objetivo es minimizar el RSS.

Regresion lineal con varias dimensiones:

                                y= a1x1+ a2x2 + b

Encesitamos especificar tres parametros: las dos slope "a1" y "a2", y el intercepto "b".

                        y= a1x1 + a2x2 + a3x3 +...+ anxn + b

Para ajustar un modelo de regresion lineal multiple se debe especificar un coeficiente "an" para "n" numero de predictores y un intercepto "b"
                

### Ejemplo


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# hacemos split de data
X_train, X_test, y_trai, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

# Crear modelo
reg_all= LinearRegression()

# Ajustamos predictores al modelo
reg_all.fit(X_train, y_trai)

# Hacer predicciones
y_pred= reg_all.predict(X_test)

# ver
print(y_pred)
```

    [69614.401966 104976.823548 210026.944318 ... 91090.256600 191229.961159
     220082.285406]
    

#### R-squared

Cuantifica la cantidad de variacion de la variable objetivo que se explica por los predictores, los valores van desde 0 a 1, donde 1 significa que los predictores explican completamente la variabcion de la variable objetivo:

![xxx](http://localhost:8888/files/Downloads/Python/MLS/1_Supervised_Learning_with_scikit_learn/8.jpg?_xsrf=2%7C2d083ae1%7C3457333de4e74ce098bbb308c43d362c%7C1721828162)


```python
# Validar metrica de r-squared
reg_all.score(X_test, y_test)
```




    0.7609020445216754


En este caso los predictores explican el 76% de la variacion del nivel de grucosa en sangre.
#### MSE
Es la media de la suma de los residuos al cuadrado, esto se conoce como error cuadratico medio MSE, el MSE se mide en unidades de la variable objetivo al cuadrado. Por ejemplo, si modelo predice el valor en dolares el MSE estara en dolares al cuadrado. Para pordel interpretar el valor sacamos la raiz cuadrada, lo que es RMSE

#### RMSE
RMSE es el raiz error cuadrativo medio.
para calcular el error cuadratico medio tomamos:

                    from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred, squared= False)

El squared lo dejamos en False lo que devuelve la raiz cuadrada del MSE, osea el RMSE. Si para el ejemplo de glucosa el RMSE es 24.0282948659438563 significa que el modelo tiene un error cuadratico medio para los niveles de glucosa en sangre al rededor de 24 miligramos por decilitro.

### Ejercicio
Crear un modelo de regresión lineal múltiple utilizando todas las características del conjunto de datos sales_df


```python
# data:
print(sales_df.head())
```

            tv    radio  social_media influencer     sales
    0 16000.00  6566.23       2907.98       Mega  54732.76
    1 13000.00  9237.76       2409.57       Mega  46677.90
    2 41000.00 15886.45       2913.41       Mega 150177.83
    3 83000.00 30020.03       6922.30       Mega 298246.34
    4 15000.00  8437.41       1406.00      Micro  56594.18
    


```python
# Crear predictores
X= sales_df.drop(['sales', 'influencer'], axis=1).values
X
```




    array([[16000.000000, 6566.230000, 2907.980000],
           [13000.000000, 9237.760000, 2409.570000],
           [41000.000000, 15886.450000, 2913.410000],
           ...,
           [44000.000000, 19800.070000, 5096.190000],
           [71000.000000, 17534.640000, 1940.870000],
           [42000.000000, 15966.690000, 5046.550000]])




```python
# Crear target
y= sales_df['sales'].values
y
```




    array([54732.760000, 46677.900000, 150177.830000, ..., 163631.460000,
           253610.410000, 148202.410000])




```python
# Split de datos de train y test
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.3, random_state= 42)
```


```python
# Asignar modelo
reg= LinearRegression()
```


```python
# Ajustar variables a modelo
reg.fit(X_train, y_train)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;LinearRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LinearRegression.html">?<span>Documentation for LinearRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>LinearRegression()</pre></div> </div></div></div></div>




```python
# Hacer prediccione
y_pred= reg.predict(X_test)
y_pred
```




    array([53176.661542, 70996.198732, 267032.641321, ..., 53186.974178,
           124484.966924, 138713.214779])


Las dos primeras predicciones parecen estar dentro del 5% de los valores reales del conjunto de prueba
Su tarea es averiguar qué tan bien las características pueden explicar la varianza en los valores objetivo, junto con evaluar la capacidad del modelo para realizar predicciones sobre datos no vistos.


```python
# Rendimiento del modelo:

# Import mean_squared_error
from sklearn.metrics import mean_squared_error

# Compute R-squared
r_squared = reg.score(X_test, y_test)

# Compute RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))
```

    R^2: 0.9990152104759368
    RMSE: 2944.433199600101
    

    C:\ProgramData\anaconda3\Lib\site-packages\sklearn\metrics\_regression.py:483: FutureWarning: 'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function'root_mean_squared_error'.
      warnings.warn(
    
Las características explican el 99,9 % de la variación en los valores de venta. En cuanto al RMSE significa que de las predicciones realizadas, estas se desvian en promedio +-2944.433199600101

```python

```
