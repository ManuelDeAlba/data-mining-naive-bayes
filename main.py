from fastapi import FastAPI, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Endpoint que retorna el template de la página principal
# Se pueden utilizar query params para pasar los valores iniciales y no perderlos al recargar la página
#? http://localhost:8000/?iteraciones=2&modelo=naive-bayes&train_size=70&clase=clase
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, modelo="naive-bayes", iteraciones=1, clase="", train_size=70):
    return templates.TemplateResponse(
        name="index.html",
        request=request,
        context={"modelo": modelo.lower(), "iteraciones": iteraciones, "clase": clase, "train_size": train_size}
    )

# Endpoint para subir un archivo CSV y realizar el proceso de Naive Bayes
# Se pueden pasar los valores del modelo, iteraciones, clase y train_size como el body de la petición como form-data
# Retorna un JSON con los resultados de las iteraciones y los parámetros utilizados en el proceso
@app.post("/upload-file")
async def uploadFile(file: UploadFile, modelo: str = Form(...), iteraciones: int = Form(...), clase: str = Form(...), train_size: int = Form(...)):
    # Leer el archivo CSV pasado por el usuario
    content = await file.read()
    decoded_content = content.decode("utf-8")
    data = StringIO(decoded_content)
    dataframe = pd.read_csv(data)

    # Separar el dataframe en atributos normales y clase para la división de los conjuntos de entrenamiento y prueba
    try:
        X = dataframe.drop(columns=[clase])
        y = dataframe[clase]
    except KeyError:
        return {"error": True, "message": "La clase especificada no existe en el archivo"}

    resultados = [] # Resultados de las iteraciones para devolver al usuario

    # Se repetirá el proceso dependiendo de las iteraciones
    for i in range(iteraciones):
        # Se divide el conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size/100, shuffle=True)

        # Unir los conjuntos de entrenamiento y prueba para mostrarlos en la tabla de resultados
        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        # Aplicar el modelo correspondiente
        if modelo == "naive-bayes":
            # Se obtienen las columnas categóricas y numéricas, primero se seleccionan las columnas por tipo
            # Después se obtienen los nombres de las columnas
            categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
            numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

            # Creamos el preprocesador para transformar las columnas a numéricas y que el modelo pueda trabajar
            # Aquí solamente le indicamos las columnas que deberá transformar y no hace el proceso en este momento
            preprocesador = ColumnTransformer(
                transformers=[
                    # Las columnas numéricas se dejan igual
                    ("num", "passthrough", numerical_cols),
                    # Las columnas categóricas se transforman a OneHot (variables dummy)
                    ("cat", OneHotEncoder(), categorical_cols)
                ]
            )

            # Se crea el pipeline por donde van a pasar los datos y pasaran por el preprocesador y por el modelo
            model = Pipeline(steps=[
                ('preprocesador', preprocesador),
                ('clasificador', GaussianNB())
            ])

            # Entrenar el modelo
            model.fit(X_train, y_train)

            try:
                # Predecir los valores de prueba
                y_pred = model.predict(X_test)
            except ValueError:
                return {"error": True, "message": "Después de la división de conjuntos, existen valores que no se encuentran en ambos. Posiblemente el dataset es muy pequeño. Intenta con un dataset más grande o repite esta operación."}

            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="macro")
            recall = recall_score(y_test, y_pred, average="macro")
            f1 = f1_score(y_test, y_pred, average="macro")

            resultados.append({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicciones": y_pred.tolist(),
                "columnas_train": train.columns.to_list(),
                "valores_train": train.values.tolist(),
                "columnas_test": test.columns.to_list(),
                "valores_test": test.values.tolist()
            })
    
    # Resultados
    return {
        "resultados": resultados,
        "modelo": modelo,
        "iteraciones": iteraciones,
        "clase": clase,
        "train_size": train_size
    }