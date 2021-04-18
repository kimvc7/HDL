setwd("~/Downloads/StatisticalAnalysis")

library(tidyverse)
library(ggplot2)
library(ggthemes)
library(viridis)
library(stringr)
library(rjson)
library(yaml)
library(rPref)
library(gdata)
library(plotly)

####################################################################################################################################
####################################################################################################################################
#Load and Merge Data
####################################################################################################################################
####################################################################################################################################


data = read_csv("results_ffcifar_michael.csv", col_names = c("stable","robust","robust_test","n_experiments",
                                                             "train_size","batch_size","subset_ratio","avg_test_acc","dict_test_accs","adv_test_accs","std",
                                                             "thetas","num_training_steps","iterations","w1_stability","w2_stability","w3_stability","logit_stability",
                                                             "gini_stability","l2","l0","W1_non_zero","W2_non_zero","W3_non_zero","l1_size","l2_size","lr","adv_acc0",
                                                             "adv_acc1","adv_acc2","adv_acc3","adv_acc4","adv_acc5"))
data2 = read_csv("results_ffcifar_leonard.csv", col_names = c("stable","robust","robust_test","n_experiments",
                                                              "train_size","batch_size","subset_ratio","avg_test_acc","dict_test_accs","adv_test_accs","std",
                                                              "thetas","num_training_steps","iterations","w1_stability","w2_stability","w3_stability","logit_stability",
                                                              "gini_stability","l2","l0","W1_non_zero","W2_non_zero","W3_non_zero","l1_size","l2_size","lr","adv_acc0",
                                                              "adv_acc1","adv_acc2","adv_acc3","adv_acc4","adv_acc5"))

data3 = read_csv("results_ffcifar_kim.csv", col_names = c("stable","robust","robust_test","n_experiments",
                                                          "train_size","batch_size","subset_ratio","avg_test_acc","dict_test_accs","adv_test_accs","std",
                                                          "thetas","num_training_steps","iterations","w1_stability","w2_stability","w3_stability","logit_stability",
                                                          "gini_stability","l2","l0","W1_non_zero","W2_non_zero","W3_non_zero","l1_size","l2_size","lr","adv_acc0",
                                                          "adv_acc1","adv_acc2","adv_acc3","adv_acc4","adv_acc5"))

data4 = read_csv("results_ffcifar_ivan.csv", col_names = c("stable","robust","robust_test","n_experiments",
                                                           "train_size","batch_size","subset_ratio","avg_test_acc","dict_test_accs","adv_test_accs","std",
                                                           "thetas","num_training_steps","iterations","w1_stability","w2_stability","w3_stability","logit_stability",
                                                           "gini_stability","l2","l0","W1_non_zero","W2_non_zero","W3_non_zero","l1_size","l2_size","lr","adv_acc0",
                                                           "adv_acc1","adv_acc2","adv_acc3","adv_acc4","adv_acc5"))

data5 = read_csv("results_ffcifar_alex.csv", col_names = c("stable","robust","robust_test","n_experiments",
                                                           "train_size","batch_size","subset_ratio","avg_test_acc","dict_test_accs","adv_test_accs","std",
                                                           "thetas","num_training_steps","iterations","w1_stability","w2_stability","w3_stability","logit_stability",
                                                           "gini_stability","l2","l0","W1_non_zero","W2_non_zero","W3_non_zero","l1_size","l2_size","lr","adv_acc0",
                                                           "adv_acc1","adv_acc2","adv_acc3","adv_acc4","adv_acc5"))


data_total = rbind(rbind(rbind(rbind(data,data2),data3),data4),data5)


####################################################################################################################################
####################################################################################################################################
#Preprocess Data
####################################################################################################################################
####################################################################################################################################

dataset_name = "cifar"
# the single most frankenstein line of code Michael has written in R so far
regex_result = apply(do.call(rbind,lapply(gsub("\\s+"," ",str_replace_all(data_total$adv_test_accs, "(array\\(|\\)|\n|\r\n)",  "")),
                                          yaml.load)),c(1,2),function(x) mean(x[[1]]))
data_total = cbind(data_total,regex_result)
avg_logit_stability = sapply(strsplit(str_sub(data_total$logit_stability, start=2, end = -2),"\\s+|\n"),function(x) mean(as.numeric(x)) )
data_total = cbind(data_total,avg_logit_stability)
data_total$stable = factor(data_total$stable)
data_total$robustness = log(data_total$robust+1e-06)/max(abs(log(data_total$robust+1e-06))) + 1
data_total$lr = factor(data_total$lr)
data_total$l2_reg = log(data_total$l2+1e-06)/max(abs(log(data_total$l2+1e-06))) + 1
data_total$sparsity = (data_total$W1_non_zero +  data_total$W2_non_zero + data_total$W3_non_zero) / 3
data_total$l0_reg = log(data_total$l0+1e-06)/max(abs(log(data_total$l0+1e-06))) + 1

#Find points in the Pareto Frontier
data_total$ID <- seq.int(nrow(data_total))
data_total$sparse <- as.factor(data_total$l0_reg != 0)
pareto <- psel(data_total, high(avg_test_acc) * high(`0.01`)  * low(avg_logit_stability) * low(sparsity))
data_total$Pareto <- (data_total$ID %in% pareto$ID)
data_total$Pareto = as.factor(data_total$Pareto)


####################################################################################################################################
####################################################################################################################################
#2D Graphs using only Pareto points
####################################################################################################################################
####################################################################################################################################

#Plot Pareto curve
fig<- plot_ly(data_total, x = ~avg_test_acc, y = ~`0.01`, z = ~avg_logit_stability, 
              marker = list( line = list(color = 'rgb(0, 0, 0)', width = 1)), color = ~sparsity, opacity = 0.1)
fig <- fig %>% add_markers()
fig <- fig %>% layout(scene = list(xaxis = list(title = 'Accuracy'),
                                   yaxis = list(title = 'Adv Accuracy'),
                                   zaxis = list(title = 'Stability')))
fig <- fig %>%
  add_trace( x = pareto$avg_test_acc, y = pareto$`0.01`, z = pareto$avg_logit_stability, color = pareto$sparsity, opacity = 1,
             marker = list(line = list(color = 'rgb(0, 0, 0)', width = 1)), name = 'Opacity 1.0', showlegend = F) 
fig



ggplot(data = pareto) + aes(x = avg_test_acc, y = avg_logit_stability, color = robustness, shape = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_few() + xlab("Test Accuracy") + ylab("Stability ((Logit)") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Robustness', shape = "Stability") + 
  ggtitle("Stability-Accuracy Tradeoff")
ggsave(paste0(dataset_name,"pareto_logit_stability_accuracy_robustness.png"))


ggplot(data = pareto) + aes(x = avg_test_acc, y = avg_logit_stability, color = lr) + 
  geom_point(alpha = 0.2,size=2) + theme_minimal() + xlab("Test Accuracy") + ylab("Stability ((Logit)") +
  labs(color='Learning Rate') +   
  ggtitle("Impact of Learning Rate on the Stability-Accuracy Frontier")
ggsave(paste0(dataset_name,"pareto_logit_learning_rate.png"))


ggplot(data = pareto) + aes(x = avg_test_acc, y = avg_logit_stability, color = l2_reg) + 
  geom_point(alpha = 0.2,size=2) + theme_minimal() + xlab("Test Accuracy") + ylab("Stability ((Logit)") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Regularization') +   
  ggtitle("Impact of Regularization on the Stability-Accuracy Frontier")
ggsave(paste0(dataset_name,"pareto_logit_regularization.png"))


ggplot(data = pareto) + aes(x = avg_test_acc, y = avg_logit_stability, color = batch_size) + 
  geom_point(alpha = 0.2,size=2) + theme_minimal() + xlab("Test Accuracy") + ylab("Stability ((Logit)") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Batch Size') +   
  ggtitle("Impact of Batch Size on the Stability-Accuracy Frontier")
ggsave(paste0(dataset_name,"pareto_logit_batch_size.png"))


ggplot(data = pareto) + aes(x = avg_test_acc, y = avg_logit_stability, color = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_minimal() + xlab("Test Accuracy") + ylab("Stability (Logit)") +
  labs(color='Stability') +   
  ggtitle("Impact of Stability on the Stability-Accuracy Frontier")
ggsave(paste0(dataset_name,"pareto_logit_stability_accuracy.png"))


ggplot(data = pareto) + aes(x = avg_test_acc, y = gini_stability, color = robustness,shape = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_few() + xlab("Test Accuracy") + ylab("Stability (Gini)") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Robustness', shape = "Stability") +xlim(0.45,0.55) + ylim(0,5e-05) + 
  ggtitle("Stability-Accuracy Tradeoff (Enlarged)")
ggsave(paste0(dataset_name,"pareto_stability_accuracy_robustness_large.png"))


ggplot(data = pareto) + aes(x = avg_test_acc, y = gini_stability, color = robustness, shape = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_few() + xlab("Test Accuracy") + ylab("Stability (Gini)") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Robustness', shape = "Stability") + 
  ggtitle("Stability-Accuracy Tradeoff")
ggsave(paste0(dataset_name,"pareto_stability_accuracy_robustness.png"))


ggplot(data = pareto) + aes(x = avg_test_acc, y = gini_stability, color = lr) + 
  geom_point(alpha = 0.2,size=2) + theme_minimal() + xlab("Test Accuracy") + ylab("Stability (Gini)") +
  labs(color='Learning Rate') +   
  ggtitle("Impact of Learning Rate on the Stability-Accuracy Frontier")
ggsave(paste0(dataset_name,"pareto_learning_rate.png"))


ggplot(data = pareto) + aes(x = avg_test_acc, y = gini_stability, color = l2_reg) + 
  geom_point(alpha = 0.2,size=2) + theme_minimal() + xlab("Test Accuracy") + ylab("Stability (Gini)") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Regularization') +   
  ggtitle("Impact of Regularization on the Stability-Accuracy Frontier")
ggsave(paste0(dataset_name,"pareto_regularization.png"))


ggplot(data = pareto) + aes(x = avg_test_acc, y = gini_stability, color = batch_size) + 
  geom_point(alpha = 0.2,size=2) + theme_minimal() + xlab("Test Accuracy") + ylab("Stability (Gini)") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Batch Size') +   
  ggtitle("Impact of Batch Size on the Stability-Accuracy Frontier")
ggsave(paste0(dataset_name,"pareto_batch_size.png"))


ggplot(data = pareto) + aes(x = avg_test_acc, y = gini_stability, color = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_minimal() + xlab("Test Accuracy") + ylab("Stability (Gini)") +
  labs(color='Stability') +   
  ggtitle("Impact of Stability on the Stability-Accuracy Frontier")
ggsave(paste0(dataset_name,"pareto_stability_accuracy.png"))



pareto %>% select( c(`1e-05`, `1e-04`, `0.001`, `0.01`,`0.1`,robustness, stable,avg_test_acc)) %>% gather("id", "value", 1:5) %>% ggplot() + aes(x = avg_test_acc, y = value, color = robustness, shape = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_few() + xlab("Test Accuracy") + ylab("Adversarial Accuracy") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Robustness', shape = "Stability") +
  ggtitle("Adversarial-Accuracy Tradeoff")
ggsave(paste0(dataset_name,"pareto_adversarial_accuracy_full.png"))



pareto %>% ggplot() + aes(x = avg_test_acc, y = `0.01`, color = robustness, shape = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_few() + xlab("Test Accuracy") + ylab("Adversarial Accuracy") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Robustness', shape = "Stability") +
  ggtitle("Adversarial-Accuracy Tradeoff (0.01)")
ggsave(paste0(dataset_name,"pareto_adversarial_accuracy_1e-2.png"))



pareto %>% ggplot() + aes(x = avg_test_acc, y = `0.001`, color = robustness, shape = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_few() + xlab("Test Accuracy") + ylab("Adversarial Accuracy") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Robustness', shape = "Stability") +
  ggtitle("Adversarial-Accuracy Tradeoff (0.001)")
ggsave(paste0(dataset_name,"pareto_adversarial_accuracy_1e-3.png"))



pareto %>% ggplot() + aes(x = avg_test_acc, y = `0.1`, color = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_few() + xlab("Test Accuracy") + ylab("Adversarial Accuracy") +
  scale_colour_manual(values = c("red", "blue", "green")) + 
  labs(color='Stability') +
  ggtitle("Adversarial-Accuracy Tradeoff (Highest) for Stability")
ggsave(paste0(dataset_name,"pareto_adversarial_accuracy_stability.png"))


pareto %>% ggplot() + aes(x = avg_test_acc, y = sparsity, color = l0_reg, shape = stable) + 
  geom_point(alpha = 0.2,size=2) + theme_few() + xlab("Test Accuracy") + ylab("Weight Sparsity") +
  scale_color_gradient(low="red", high="blue")+ 
  labs(color='Sparsity Parameter', shape = "Stability") +
  ggtitle("Sparsity-Accuracy Tradeoff")
ggsave(paste0(dataset_name,"pareto_sparsity_accuracy.png"))


####################################################################################################################################
####################################################################################################################################
#Analysis per Dimension in Pareto Frontier
####################################################################################################################################
####################################################################################################################################


robust_requirement = function(x){df = subset(pareto, pareto$`0.01` >= x); return(sum(df$robust==0.01)/nrow(df))}
sparse_requirement = function(x){df = subset(pareto, pareto$sparsity <= x); return(sum(df$sparse==TRUE)/nrow(df))}
stable_requirement = function(x){df = subset(pareto, pareto$avg_logit_stability <= x); return(sum(df$stable==1)/nrow(df))}

robust_M = max(pareto$`0.01`)
robust_m = min(pareto$`0.01`)
robust_domain = robust_m + (1:500)*((robust_M - robust_m)/500)
robust_images = sapply(robust_domain, robust_requirement)
plot(robust_domain, robust_images)

stable_M = max(pareto$avg_logit_stability)
stable_m = min(pareto$avg_logit_stability)
stable_domain = stable_m + (1:500)*((stable_M - stable_m)/500)
stable_images = sapply(stable_domain, stable_requirement)
plot(stable_domain, stable_images)

sparse_M = max(pareto$sparsity)
sparse_m = min(pareto$sparsity)
sparse_domain = sparse_m + (1:500)*((sparse_M - sparse_m)/500)
sparse_images = sapply(sparse_domain, sparse_requirement)
plot(sparse_domain, sparse_images)

