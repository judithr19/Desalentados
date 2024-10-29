# Modelos de Machine Learning: Desalentados en México
# ______ Librerías ______#
library(dplyr);library(tidyr);library(ggplot2);library(WriteXLS);
library(readxl);library(corrplot);library(gam);library(ISLR);
library(splines);library(MASS);library(scatterplot3d);
library(patchwork);library(cluster);library(factoextra);
library(tibble);library(NbClust);library(dendextend);
library(VIM);library(car);library(rpart);library(randomForest);
library(e1071);library(forecast);library(KFAS);library(zoo);
library(keras);library(ssmodels);library(readxl);
library(tidyr);library("mlr3");library(mlr3verse);
library(mlr3learners);library(WriteXLS);library(tidyverse);
library(readr);library(caret);library(nnet);

#_________________________________________________________#
#_________________________________________________________#
#_________________________________________________________#

# Fijar directorio donde se encuentre la subcarpeta BasesDatos
# setwd("~/...")
# setwd("C:/...")

# Elegir las bases de datos de acuerdo al año analizado (4 trimestres)
# En el 2020 no hubo datos del segundo trimestre
datost1<-read.csv("t105.csv",sep=",",header = T)
datost2<-read.csv("t205.csv",sep=",",header = T)
datost3<-read.csv("t305.csv",sep=",",header = T)
datost4<-read.csv("t405.csv",sep=",",header = T)

# Se unen todos los trimestres
datos<-rbind(datost1,datost2,datost3,datost4)
datos <- datos %>%
  mutate_all(as.factor)


#_________________________________________________________#
#_________________________________________________________#
#_________________________________________________________#
# Se dividen los datos  en 80%-20% (entrenamiento y prueba)
set.seed(123) # Establecer semilla para reproducibilidad
indice_entrenamiento <- createDataPartition(datos$respuesta, p = 0.8, list = FALSE)
datos_entrenamiento <- datos[indice_entrenamiento, ]
datos_prueba <- datos[-indice_entrenamiento, ]

#________________ Modelos _______________#
# Entrenar modelo de regresión logística multinomial
modelo_rlm <- multinom(respuesta ~ ., data = datos_entrenamiento)
predicciones_rlm <- predict(modelo_rlm, newdata = datos_prueba)
mc_rlm<- table(datos_prueba$respuesta, predicciones_rlm)
precision_rlm <- sum(diag(mc_rlm)) / sum(mc_rlm)

# Entrenar modelo de árbol de decisión
library(rpart)
modelo_arboles <- rpart(respuesta ~ ., data = datos_entrenamiento)
predicciones_arboles <- predict(modelo_arboles, newdata = datos_prueba, type = "class")
mc_ad<- table(datos_prueba$respuesta, predicciones_arboles)
precision_ad <- sum(diag(mc_ad)) / sum(mc_ad)

# Random Forest
library(randomForest)
# Entrenar modelo de random forest
modelo_rf <- randomForest(respuesta ~ ., data = datos_entrenamiento)
predicciones_rf <- predict(modelo_rf, newdata = datos_prueba)
mc_rf<- table(datos_prueba$respuesta, predicciones_rf)
precision_rf <- sum(diag(mc_rf)) / sum(mc_rf)
importance(modelo_rf)
#write.csv(importance(modelo_rf),"Peso_ModeloRF_20.csv")


# Clasificador Naive Bayes
library(e1071)
modelo_nb <- naiveBayes(respuesta ~ ., data = datos_entrenamiento)
predicciones_nb <- predict(modelo_nb, newdata = datos_prueba)
mc_nb<- table(datos_prueba$respuesta, predicciones_nb)
precision_nb <- sum(diag(mc_nb)) / sum(mc_nb)

# Modelo k-NN
library(kknn)
modelo_knn <- kknn(respuesta ~ ., train = datos_entrenamiento, test = datos_prueba, k = 5)
predicciones_kknn <- fitted(modelo_knn)
mc_kknn<- table(datos_prueba$respuesta, predicciones_kknn)
precision_kknn <- sum(diag(mc_kknn)) / sum(mc_kknn)

# Modelo de red neuronal
library(nnet)
modelo_ann <- nnet(respuesta ~ ., data = datos_entrenamiento, size = 10, maxit = 100)
predicciones_ann <- predict(modelo_ann, newdata = datos_prueba, type = "class")
mc_ann<- table(datos_prueba$respuesta, predicciones_ann)
precision_ann <- sum(diag(mc_ann)) / sum(mc_ann)

# Modelo SVM
library(e1071)
modelo_svm <- svm(respuesta ~ ., data = datos_entrenamiento, kernel = "radial")
predicciones_svm <- predict(modelo_svm, newdata = datos_prueba)
mc_svm<- table(datos_prueba$respuesta, predicciones_svm)
precision_svm <- sum(diag(mc_svm)) / sum(mc_svm)

# Tabla de Precisiones
tablaPreci<-data.frame(Método=c("Regresión logística","Árbol de decisión", "Bosque Aleatorio",
                    "Naive Bayes", "k-NN", "Red neuronal", "SVM"),
           Precisión=c(precision_rlm,precision_ad,precision_rf,precision_nb,precision_kknn,
                       precision_ann,precision_svm))
tablaPreci
data_reordenado <- arrange(tablaPreci, desc(Precisión))# reordenamos
data_reordenado
#write.csv(tablaPreci,"Precisiones_2020.csv")

#__________________________________________#
#_________________ Gráficas _______________#
#__________________________________________#

g1<-ggplot(datos_prueba, aes(x = factor(respuesta), fill = factor(predicciones_rlm))) +
  geom_bar(position = "fill", show.legend = FALSE) +
  labs(title = " Regresión logística",x = "",y = "Tasa ") + 
  scale_x_discrete(labels = c("Desocupada", "Disponible"))
g2<-ggplot(datos_prueba, aes(x = factor(respuesta), fill = factor(predicciones_arboles))) +
  geom_bar(position = "fill", show.legend = FALSE) +
  labs(title = " Árboles de decisión",x = "",y = "Tasa ") + 
  scale_x_discrete(labels = c("Desocupada", "Disponible"))
g3<-ggplot(datos_prueba, aes(x = factor(respuesta), fill = factor(predicciones_rf))) +
  geom_bar(position = "fill" ,show.legend = FALSE) +
  labs(title = " Bosque aleatorio", x = "",y = "Tasa ") + 
  scale_x_discrete(labels = c("Desocupada", "Disponible"))
g4<-ggplot(datos_prueba, aes(x = factor(respuesta), fill = factor(predicciones_nb))) +
  geom_bar(position = "fill", show.legend = FALSE) +
  labs(title = " Naive Bayes",x = "",y = "Tasa ") + 
  scale_x_discrete(labels = c("Desocupada", "Disponible"))
g5<-ggplot(datos_prueba, aes(x = factor(respuesta), fill = factor(predicciones_kknn))) +
  geom_bar(position = "fill", show.legend = FALSE) +
  labs(title = " k-nn",x = "",y = "Tasa ") + 
  scale_x_discrete(labels = c("Desocupada", "Disponible"))
g6<-ggplot(datos_prueba, aes(x = factor(respuesta), fill = factor(predicciones_ann))) +
  geom_bar(position = "fill" ,show.legend = FALSE) +
  labs(title = " Red Neuronal", x = "",y = "Tasa ") + 
  scale_x_discrete(labels = c("Desocupada", "Disponible"))
g7<-ggplot(datos_prueba, aes(x = factor(respuesta), fill = factor(predicciones_svm))) +
  geom_bar(position = "fill" ,show.legend = FALSE) +
  labs(title = " SVM", x = "",y = "Tasa ") + 
  scale_x_discrete(labels = c("Desocupada", "Disponible"))

dev.new(4,4)
grafica_precisiones<-(g1+g2+g3+g4)/(g5+g6+g7)
grafica_precisiones

ggsave(
  paste("Grafica_Precisiones_2020", ".pdf"),
  plot = grafica_precisiones,
  device = "pdf",
  width = 10,
  height = 8
)

#_____________________________________________________________#
# Una vez definido cuál fue el mejor modelo 
library(neuralnet)
library(caret)
library(e1071)
library(MASS)

# Probaremos el modelo para el trimestre 4
datost4 <- datost4 %>%
  mutate_all(as.factor)

predicciones_ann_t4 <- predict(modelo_ann, newdata = datost4, type = "class")
mc_ann_t4<- table(datost4$respuesta, predicciones_ann_t4)
precision_ann_t4 <- sum(diag(mc_ann_t4)) / sum(mc_ann_t4)
#write.csv(precision_ann_t4,"Precisiont420.csv")

#____________________________________________________#
# Dividir los datos en conjuntos de entrenamiento y prueba
 set.seed(123) # Para reproducibilidad
 indices <- createDataPartition(datost4$respuesta, p = 0.8, list = FALSE)
 datos_entrenamiento <- datost4[indices, ]
 datos_prueba <- datost4[-indices, ]

# 1. Optimización de hiperparámetros para redes neuronales
# Define una función de control para la optimización de hiperparámetros
ctrl <- trainControl(method = "cv", number = 5)
# Define la cuadrícula de parámetros a explorar
param_grid <- expand.grid(size = c(5, 10, 15), decay = c(0.1, 0.01, 0.001))
# Entrena el modelo de redes neuronales con optimización de hiperparámetros
modelo_nn <- train(respuesta ~ ., data = datos_entrenamiento, method = "nnet",
                   trControl = ctrl, tuneGrid = param_grid)
summary(modelo_nn)
modelo_nn$results

# 2. Validación cruzada
# Realiza validación cruzada para evaluar el rendimiento del modelo
cv_resultados <- train(respuesta ~ ., data = datos_entrenamiento, method = "nnet",
                       trControl = trainControl(method = "cv", number = 5))
cv_resultados

# 3. Selección de características
# Realiza selección de características utilizando un modelo de regresión logística
modelo_reglog <- glm(respuesta ~ ., data = datos_entrenamiento, family = "binomial")
# Selecciona características importantes basadas en los coeficientes del modelo
caracteristicas_importantes <- caret::varImp(modelo_reglog)
caracteristicas_importantes
#write.csv(caracteristicas_importantes, "caracteristicas_importantes_t4_2020.csv")

# 4. Análisis de errores
# Evalúa los errores del modelo de redes neuronales
predicciones_entrenamiento <- predict(modelo_nn, newdata = datos_entrenamiento)
errores <- confusionMatrix(predicciones_entrenamiento, datos_entrenamiento$respuesta)
errores

#___________________________________________________________#
# Explicación de las variables para las redes neuronales
library(lime)
# Crear una explicación de LIME para una observación de prueba
# Crear una función de predicción para la red neuronal
prediccion_func <- function(datost4) predict(modelo_ann, datost4)
explicacion <- lime(datos_prueba, modelo_ann, prediccion_func)
# Obtener características importantes
caracteristicas_importantes <- explicacion$feature_distribution
print(caracteristicas_importantes)

# Crear un data frame para almacenar los pesos de las características
df_caracteristicas <- data.frame(
  Feature = character(),  # Inicializar un vector vacío para las características
  Level = character(),    # Inicializar un vector vacío para los niveles de las características
  Weight = numeric()      # Inicializar un vector vacío para los pesos de las características
)

# Iterar sobre las características importantes
for (feature_name in names(caracteristicas_importantes)) {
  # Obtener los niveles y pesos de la característica actual
  niveles <- names(caracteristicas_importantes[[feature_name]])
  pesos <- unlist(caracteristicas_importantes[[feature_name]])
  
  # Crear un data frame temporal para almacenar los niveles y pesos de la característica actual
  df_temp <- data.frame(
    Feature = rep(feature_name, length(niveles)),
    Level = niveles,
    Weight = pesos
  )
  
  # Agregar el data frame temporal al data frame principal
  df_caracteristicas <- rbind(df_caracteristicas, df_temp)
}
df_caracteristicas

# Convertir la variable de las características en un factor con el orden deseado
df_caracteristicas$Feature <- factor(df_caracteristicas$Feature, levels = unique(df_caracteristicas$Feature))

# Crear el gráfico de barras para todas las variables
dev.new(4,4)
grafico <- ggplot(df_caracteristicas, aes(x = Level, y = Weight.Freq)) +
  geom_bar(stat = "identity") +
  labs(x = "Nivel de Característica", y = "Peso") +
  ggtitle("Pesos de las características") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  facet_wrap(~ Feature, scales = "free_y", ncol = 1)

# Mostrar el gráfico
print(grafico)

ggsave(
  paste("Caracteristicas_2020", ".pdf"),
  plot = grafico,
  device = "pdf",
  width = 10,
  height = 8
)
