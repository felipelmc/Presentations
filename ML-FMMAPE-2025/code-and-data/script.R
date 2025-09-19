# --------------------------------------------------------------
# Machine Learning
# regressão linear (votos) e regressão logistica (resultado_bin)
# --------------------------------------------------------------

library(tidyverse)
library(rpart)
library(rpart.plot)

# 1) Ler dados
dados <- read_csv("data.csv")

# 2) Split treino/teste (70/30)
set.seed(123)
n <- nrow(dados)
idx_treino <- sample.int(n, size = round(0.7*n))
treino <- dados[idx_treino, ]
teste  <- dados[-idx_treino, ]

# 3) Funções de métricas (curtas)
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2, na.rm = TRUE))
mae  <- function(y, yhat) mean(abs(y - yhat), na.rm = TRUE)

accuracy  <- function(y, yhat) mean(y == yhat, na.rm = TRUE)
precision <- function(y, yhat) {
  tp <- sum(y==1 & yhat==1, na.rm = TRUE)
  fp <- sum(y==0 & yhat==1, na.rm = TRUE)
  if (tp+fp == 0) return(NA) else tp/(tp+fp)
}
recall <- function(y, yhat) {
  tp <- sum(y==1 & yhat==1, na.rm = TRUE)
  fn <- sum(y==1 & yhat==0, na.rm = TRUE)
  if (tp+fn == 0) return(NA) else tp/(tp+fn)
}
f1 <- function(p, r) if (is.na(p) || is.na(r) || p+r==0) NA else 2*p*r/(p+r)


# ------------------------------------------------------------
# REGRESSÃO (alvo = votos): SEM e COM incumbente
# ------------------------------------------------------------

# (A) sem incumbente
m_lm_A <- lm(votos ~ genero + despesa_total, data = treino)
pA <- predict(m_lm_A, newdata = teste)

# (B) com incumbente
m_lm_B <- lm(votos ~ incumbente + genero + despesa_total, data = treino)
pB <- predict(m_lm_B, newdata = teste)

res_reg <- rbind(
  c("sem_incumbente", rmse(teste$votos, pA), mae(teste$votos, pA), summary(m_lm_A)$r.squared),
  c("com_incumbente", rmse(teste$votos, pB), mae(teste$votos, pB), summary(m_lm_B)$r.squared)
)
colnames(res_reg) <- c("especificacao","RMSE_teste","MAE_teste","R2_treino")

cat("\n=== Regressão (votos) ===\n")
print(res_reg)

# ------------------------------------------------------------
# CLASSIFICAÇÃO (alvo = resultado_bin): SEM e COM incumbente
# ------------------------------------------------------------

# (A) sem incumbente
m_logit_A <- glm(resultado_bin ~ genero + despesa_total,
                 data = treino, family = binomial("logit"))
probA <- predict(m_logit_A, newdata = teste, type = "response")
cA <- ifelse(probA >= 0.5, 1, 0)

# (B) com incumbente
m_logit_B <- glm(resultado_bin ~ incumbente + genero + despesa_total,
                 data = treino, family = binomial("logit"))
probB <- predict(m_logit_B, newdata = teste, type = "response")
cB <- ifelse(probB >= 0.5, 1, 0)

precA <- precision(teste$resultado_bin, cA); recA <- recall(teste$resultado_bin, cA)
precB <- precision(teste$resultado_bin, cB); recB <- recall(teste$resultado_bin, cB)

res_cls <- rbind(
  c("sem_incumbente", accuracy(teste$resultado_bin, cA), precA, recA, f1(precA, recA)),
  c("com_incumbente", accuracy(teste$resultado_bin, cB), precB, recB, f1(precB, recB))
)
colnames(res_cls) <- c("especificacao","Acuracia","Precisao","Recall","F1")

cat("\n=== Classificação (resultado_bin) ===\n")
print(res_cls)

# Matriz de confusão para a especificação com incumbente
cat("\nMatriz de confusão (com_incumbente):\n")
print(table(Real = teste$resultado_bin, Predito = cB))


# ------------------------------------------------------------
# CLASSIFICAÇÃO com ÁRVORE DE DECISÃO (resultado_bin)
# ------------------------------------------------------------

# (A) sem incumbente
dt_A <- rpart(
  formula = resultado_bin ~ genero + despesa_total,
  data = treino,
  method = "class",
  control = rpart.control(minsplit = 20, cp = 0.001, maxdepth = 10)
)

prob_dt_A <- predict(dt_A, newdata = teste, type = "prob")[,2]
c_dt_A <- ifelse(prob_dt_A >= 0.5, 1, 0)

# (B) com incumbente
dt_B <- rpart(
  formula = resultado_bin ~ incumbente + genero + despesa_total,
  data = treino,
  method = "class",
  control = rpart.control(minsplit = 20, cp = 0.001, maxdepth = 10)
)

prob_dt_B <- predict(dt_B, newdata = teste, type = "prob")[,2]
c_dt_B <- ifelse(prob_dt_B >= 0.5, 1, 0)

# Métricas
prec_dt_A <- precision(teste$resultado_bin, c_dt_A); rec_dt_A <- recall(teste$resultado_bin, c_dt_A)
prec_dt_B <- precision(teste$resultado_bin, c_dt_B); rec_dt_B <- recall(teste$resultado_bin, c_dt_B)

res_dt <- rbind(
  c("sem_incumbente", accuracy(teste$resultado_bin, c_dt_A), prec_dt_A, rec_dt_A, f1(prec_dt_A, rec_dt_A)),
  c("com_incumbente", accuracy(teste$resultado_bin, c_dt_B), prec_dt_B, rec_dt_B, f1(prec_dt_B, rec_dt_B))
)
colnames(res_dt) <- c("especificacao","Acuracia","Precisao","Recall","F1")

cat("\n=== Árvore de decisão (resultado_bin) ===\n")
print(res_dt)

# Matriz de confusão para a especificação com incumbente
cat("\nMatriz de confusão (Árvore - com_incumbente):\n")
print(table(Real = teste$resultado_bin, Predito = c_dt_B))

# visualizar as árvores
rpart.plot(dt_A, main = "Árvore - sem incumbente")
rpart.plot(dt_B, main = "Árvore - com incumbente")


