setwd("~/Downloads/StatisticalAnalysis/MNIST/")

library(tidyverse)
library(ggplot2)
library(ggthemes)
library(viridis)
library(stringr)
library(rjson)
library(yaml)

data = read_csv("results_ffmnist.csv", col_names = c("stable","robust","robust_test","n_experiments",
                                                     "train_size","batch_size","subset_ratio","avg_test_acc","dict_test_accs","adv_test_accs","std",
                                                     "thetas","num_training_steps","iterations","w1_stability","w2_stability","w3_stability","logit_stability",
                                                     "gini_stability","l2","l0","W1_non_zero","W2_non_zero","W3_non_zero","l1_size","l2_size","lr","adv_acc0",
                                                     "adv_acc1","adv_acc2","adv_acc3","adv_acc4","adv_acc5"))
data2 = read_csv("results_ffmnist2.csv", col_names = c("stable","robust","robust_test","n_experiments",
                                                     "train_size","batch_size","subset_ratio","avg_test_acc","dict_test_accs","adv_test_accs","std",
                                                     "thetas","num_training_steps","iterations","w1_stability","w2_stability","w3_stability","logit_stability",
                                                     "gini_stability","l2","l0","W1_non_zero","W2_non_zero","W3_non_zero","l1_size","l2_size","lr","adv_acc0",
                                                     "adv_acc1","adv_acc2","adv_acc3","adv_acc4","adv_acc5"))


data_total = rbind(data,data2)#rbind(rbind(rbind(rbind(data,data2),data3),data4),data5)


adv_acc = sapply(strsplit(str_sub(data_total$adv_test_accs, start=2, end = -2),"\\s+|\n"),function(x) mean(as.numeric(x)))
data_total = cbind(data_total,adv_acc)

avg_logit_stability = sapply(strsplit(str_sub(data_total$logit_stability, start=2, end = -2),"\\s+|\n"),function(x) mean(as.numeric(x)) )
data_total = cbind(data_total,avg_logit_stability)

data_total$stable = factor(data_total$stable)
data_total$robustness = log(data_total$robust+1e-06)/max(abs(log(data_total$robust+1e-06))) + 1
data_total$lr = factor(data_total$lr)
data_total$l2_reg = log(data_total$l2+1e-06)/max(abs(log(data_total$l2+1e-06))) + 1
data_total$sparsity = (data_total$W1_non_zero +  data_total$W2_non_zero + data_total$W3_non_zero) / 3
data_total$l0_reg = log(data_total$l0+1e-06)/max(abs(log(data_total$l0+1e-06))) + 1

regression_columns = c("stable","robust","batch_size","l2","l0")
pred_columns = c("avg_test_acc", "w1_stability","w2_stability","w3_stability", "gini_stability")


model1 = lm(avg_test_acc ~ stable + robust + batch_size + l2 + l0, data = data_total)
summary(model1)

model2 = lm(gini_stability ~ stable + robust + batch_size + l2 + l0, data = data_total)
summary(model2)

model3 = lm(gini_stability ~ stable + robust + batch_size + l2 + l0 + stable:robust, data = data_total)
summary(model3)

model4 = lm(gini_stability ~ stable + robust + batch_size + l2 + l0 + stable:robust + stable:l0 + robust:l0, data = data_total)
summary(model4)

model5 = lm(gini_stability ~ stable + robust + batch_size + l2 + l0 + stable:robust + stable:l0 + robust:l0 + stable:robust:l0, data = data_total)
summary(model5)

model6 = lm(gini_stability ~ (stable + robust + batch_size + l2 + l0)^3, data = data_total)
summary(model6)

#####3
model7 = lm(gini_stability ~ (stable + robust + l0)^3 +  batch_size + l2, data = data_total)
summary(model7)

model8 = lm(avg_test_acc ~ (stable + robust + l0)^3 +  batch_size + l2, data = data_total)
summary(model8)

#######
model9 = lm(avg_test_acc ~ (stable + robustness + sparsity)^3 +  batch_size + l2, data = data_total)
summary(model9)

model10 = lm(avg_test_acc ~ (stable + robustness + sparsity + l0_reg)^3 +  batch_size + l2_reg, data = data_total)
summary(model10)

train_X = data_total[, c("stable","l0_reg")]
train_X = data_total[, c("stable", "robustness", "sparsity", "l0_reg", "batch_size", "l2_reg")]
train_y = data_total[, "w3_stability"]
#train_y = data_total[,"avg_test_acc"]

model10 = lm(w3_stability ~ (stable + robustness + l0_reg)^3 + sparsity + batch_size + l2_reg, data = data_total)
summary(model10)

grid <- iai::grid_search(
  iai::optimal_tree_regressor(
    random_seed = 123,
  ),
  max_depth = 1:3,
)
iai::fit(grid, train_X, train_y)
iai::get_learner(grid)


library(tree)


data_total$stable = factor(data_total$stable)
data_total$robustness = log(data_total$robust+1e-06)/max(abs(log(data_total$robust+1e-06))) + 1
data_total$lr = factor(data_total$lr)
data_total$l2_reg = log(data_total$l2+1e-06)/max(abs(log(data_total$l2+1e-06))) + 1
data_total$sparsity = (data_total$W1_non_zero +  data_total$W2_non_zero + data_total$W3_non_zero) / 3
data_total$l0_reg = log(data_total$l0+1e-06)/max(abs(log(data_total$l0+1e-06))) + 1


# data_total$gini_stability = data_total$gini_stability + 
#   0.01 * mean(abs(data_total$gini_stability)) * (runif(length(data_total$gini_stability)) * 2 - 1)
# data_total$avg_test_acc = data_total$avg_test_acc + 
#   0.01 * mean(abs(data_total$avg_test_acc)) * (runif(length(data_total$avg_test_acc)) * 2 - 1)
# data_total$avg_adv_test_acc = data_total$avg_adv_test_acc + 
#   0.01 * mean(abs(data_total$avg_adv_test_acc)) * (runif(length(data_total$avg_adv_test_acc)) * 2 - 1)
ggplot(data = data_total) + aes(x = avg_test_acc, y = gini_stability, color = robustness,shape = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_few() + xlab("Test Accuracy") + ylab("Stability (Gini)") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Robustness', shape = "Stability") +xlim(0.45,0.55) + ylim(0,5e-05) + 
  ggtitle("Stability-Accuracy Tradeoff (Enlarged)")
ggsave(paste0(dataset_name,"_stability_accuracy_robustness_large.png"))


ggplot(data = data_total) + aes(x = avg_test_acc, y = gini_stability, color = robustness, shape = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_few() + xlab("Test Accuracy") + ylab("Stability (Gini)") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Robustness', shape = "Stability") + 
  ggtitle("Stability-Accuracy Tradeoff")
ggsave(paste0(dataset_name,"_stability_accuracy_robustness.png"))


ggplot(data = data_total) + aes(x = avg_test_acc, y = gini_stability, color = lr) + 
  geom_point(alpha = 0.2,size=2) + theme_minimal() + xlab("Test Accuracy") + ylab("Stability (Gini)") +
  labs(color='Learning Rate') +   
  ggtitle("Impact of Learning Rate on the Stability-Accuracy Frontier")
ggsave(paste0(dataset_name,"_learning_rate.png"))


ggplot(data = data_total) + aes(x = avg_test_acc, y = gini_stability, color = l2_reg) + 
  geom_point(alpha = 0.2,size=2) + theme_minimal() + xlab("Test Accuracy") + ylab("Stability (Gini)") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Regularization') +   
  ggtitle("Impact of Regularization on the Stability-Accuracy Frontier")
ggsave(paste0(dataset_name,"_regularization.png"))


ggplot(data = data_total) + aes(x = avg_test_acc, y = gini_stability, color = batch_size) + 
  geom_point(alpha = 0.2,size=2) + theme_minimal() + xlab("Test Accuracy") + ylab("Stability (Gini)") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Batch Size') +   
  ggtitle("Impact of Batch Size on the Stability-Accuracy Frontier")
ggsave(paste0(dataset_name,"_batch_size.png"))


ggplot(data = data_total) + aes(x = avg_test_acc, y = gini_stability, color = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_minimal() + xlab("Test Accuracy") + ylab("Stability (Gini)") +
  labs(color='Stability') +   
  ggtitle("Impact of Stability on the Stability-Accuracy Frontier")
ggsave(paste0(dataset_name,"_stability_accuracy.png"))



data_total %>% select( c(`1e-05`, `1e-04`, `0.001`, `0.01`,`0.1`,robustness, stable,avg_test_acc)) %>% gather("id", "value", 1:5) %>% ggplot() + aes(x = avg_test_acc, y = value, color = robustness, shape = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_few() + xlab("Test Accuracy") + ylab("Adversarial Accuracy") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Robustness', shape = "Stability") +
  ggtitle("Adversarial-Accuracy Tradeoff")
ggsave(paste0(dataset_name,"_adversarial_accuracy_full.png"))



data_total %>% ggplot() + aes(x = avg_test_acc, y = `0.1`, color = robustness, shape = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_few() + xlab("Test Accuracy") + ylab("Adversarial Accuracy") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Robustness', shape = "Stability") +
  ggtitle("Adversarial-Accuracy Tradeoff (Highest)")
ggsave(paste0(dataset_name,"_adversarial_accuracy_1e-1.png"))


data_total %>% ggplot() + aes(x = avg_test_acc, y = `0.01`, color = robustness, shape = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_few() + xlab("Test Accuracy") + ylab("Adversarial Accuracy") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Robustness', shape = "Stability") +
  ggtitle("Adversarial-Accuracy Tradeoff (0.01)")
ggsave(paste0(dataset_name,"_adversarial_accuracy_1e-2.png"))



data_total %>% ggplot() + aes(x = avg_test_acc, y = `0.001`, color = robustness, shape = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_few() + xlab("Test Accuracy") + ylab("Adversarial Accuracy") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Robustness', shape = "Stability") +
  ggtitle("Adversarial-Accuracy Tradeoff (0.001)")
ggsave(paste0(dataset_name,"_adversarial_accuracy_1e-3.png"))


data_total %>% ggplot() + aes(x = avg_test_acc, y = `1e-4`, color = robustness, shape = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_few() + xlab("Test Accuracy") + ylab("Adversarial Accuracy") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Robustness', shape = "Stability") +
  ggtitle("Adversarial-Accuracy Tradeoff (0.0001)")
ggsave(paste0(dataset_name,"_adversarial_accuracy_1e-4.png"))


data_total %>% ggplot() + aes(x = avg_test_acc, y = `0.1`, color = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_few() + xlab("Test Accuracy") + ylab("Adversarial Accuracy") +
  scale_colour_manual(values = c("red", "blue", "green")) + 
  labs(color='Stability') +
  ggtitle("Adversarial-Accuracy Tradeoff (Highest) for Stability")
ggsave(paste0(dataset_name,"_adversarial_accuracy_stability.png"))


data_total %>% ggplot() + aes(x = avg_test_acc, y = sparsity, color = l0_reg, shape = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_few() + xlab("Test Accuracy") + ylab("Weight Sparsity") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Sparsity Parameter', shape = "Stability") +
  ggtitle("Sparsity-Accuracy Tradeoff")
ggsave(paste0(dataset_name,"_sparsity_accuracy.png"))


