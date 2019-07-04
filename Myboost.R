##---------------------------##
#     Input Parameters       #
##---------------------------##

## (1) dat: data frame with predictors ONLY, not include trt01p, aval and evnt
## (2) labels: embedded label
##     Use below mapping function 
## embed trt01p, aval and evnt into y, which serves as the 'label' for xgboost ##
#  labels <- rep(NA,N)
#  labels[dat$trt01p==1 & dat$evnt==1] <- -1000-dat$aval[dat$trt01p==1 & dat$evnt==1] 
#  labels[dat$trt01p==1 & dat$evnt==0] <- -1-dat$aval[dat$trt01p==1 & dat$evnt==0] 
#  labels[dat$trt01p==0 & dat$evnt==1] <- 1000+dat$aval[dat$trt01p==0 & dat$evnt==1] 
#  labels[dat$trt01p==0 & dat$evnt==0] <- dat$aval[dat$trt01p==0 & dat$evnt==0] 

##---------------------------##
#     Return Values          #
##---------------------------##
# Return a vector with 2 elements, first one for wherther signal has top rank of variable impoertance
# second is a classification error in terms of signal group 


library(survminer)
library(survival)
library(dplyr)
library(survRM2)
library(xgboost)
library(DiagrammeR)
library(purrr)


MyBoost <- function(dat){
  
  dat$trt01p <- as.numeric(dat$trt01p=="Pembro")
  N <- nrow(dat)
  labels <- rep(NA,N)
  labels[dat$trt01p==1 & dat$evnt==1] <- -1000-dat$aval[dat$trt01p==1 & dat$evnt==1] 
  labels[dat$trt01p==1 & dat$evnt==0] <- -1-dat$aval[dat$trt01p==1 & dat$evnt==0] 
  labels[dat$trt01p==0 & dat$evnt==1] <- 1000+dat$aval[dat$trt01p==0 & dat$evnt==1] 
  labels[dat$trt01p==0 & dat$evnt==0] <- dat$aval[dat$trt01p==0 & dat$evnt==0] 
  
  dat$evnt <- NULL; dat$aval <- NULL; dat$trt01p <- NULL;
  #----------------------------------------------------------------------#
  #                     Step 2: Customized loss and error function                  #
  #----------------------------------------------------------------------#

  rmst_loss <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    
    ## (0) Decode y to trt01p, aval and evnt ##
    trt01p<-rep(NA,length(labels))
    evnt<-rep(NA,length(labels))
    aval<-rep(NA,length(labels))
    trt01p[labels< 0] <- 1
    trt01p[labels>= 0] <- 0
    evnt[abs(labels)>= (1000)] <- 1
    evnt[abs(labels)< (1000)] <- 0
    aval[abs(labels)>= (1000)] <- abs(labels[abs(labels)>= (1000)])-1000
    aval[labels<0 & labels>-1000] <- -labels[labels<0 & labels>-1000]-1
    aval[labels>=0 & labels<1000] <- labels[labels>=0 & labels<1000]
    
    arm.val <- c(1,0)
    ## (1) Get Time to event Data Ready ##
    km.dat <- data.frame(trt01p,evnt,aval)
    km.dat$pred <- 1/(1+exp(-preds))
    km.dat$predg <- exp(preds)/(1+exp(preds))^2
    km.dat$predh <- exp(preds)*(1-exp(preds))/(1+exp(preds))^3
    km.dat<-km.dat[order(km.dat$aval),]
    n<-dim(km.dat)[1]
    
    ## (2) Set up gradient and Hessian ##
    utime <- unique(km.dat$aval)
    dt <- utime-c(0,utime[1:length(utime)-1])
    
    rmst.diff.r1 <- 0
    rmst.diff.r2 <- 0
    rmst.diff.r1.g <- 0
    rmst.diff.r2.g <- 0
    rmst.diff.r1.h <- 0
    rmst.diff.r2.h <- 0
    
    
    for(i in 0:(length(utime)-1)){
      if(i==0){
        H1.r1 <- 0
        gH1.r1 <- 0
        hH1.r1 <- 0
        H0.r1 <- 0
        gH0.r1 <- 0
        hH0.r1 <- 0
        
        H1.r2 <- 0
        gH1.r2 <- 0
        hH1.r2 <- 0
        H0.r2 <- 0
        gH0.r2 <- 0
        hH0.r2 <- 0
        
      } else {
        denom <- subset(km.dat,aval>=utime[i])
        nume <- subset(km.dat,aval==utime[i])
        
        gH1.r1.denom <- sum((denom$trt01p==arm.val[1])*denom$pred)
        gH0.r1.denom <- sum((denom$trt01p==arm.val[2])*denom$pred)
        
        gH1.r2.denom <- sum((denom$trt01p==arm.val[1])*(1-denom$pred))
        gH0.r2.denom <- sum((denom$trt01p==arm.val[2])*(1-denom$pred)) 
        
        ## H1 r1 ##
        if(gH1.r1.denom > 0){
          ## H1 ##
          H1.r1 <- H1.r1 + sum((nume$trt01p==arm.val[1])*(nume$evnt==1)*nume$pred) / gH1.r1.denom
          ## dH1/dp ##
          gH1.r1 <- gH1.r1 + (((km.dat$aval==utime[i])*(km.dat$trt01p==arm.val[1])*(km.dat$evnt==1)*sum((denom$trt01p==arm.val[1])*denom$pred)) - ( sum((nume$trt01p==arm.val[1])*(nume$evnt==1)*nume$pred)*(km.dat$aval>=utime[i])*(km.dat$trt01p==arm.val[1])  )) / gH1.r1.denom^2
          ## d2H1/dp2 ##
          hH1.r1 <- hH1.r1 + (-2)*(((km.dat$aval==utime[i])*(km.dat$trt01p==arm.val[1])*(km.dat$evnt==1)*sum((denom$trt01p==arm.val[1])*denom$pred)) - ( sum((nume$trt01p==arm.val[1])*(nume$evnt==1)*nume$pred)*(km.dat$aval>=utime[i])*(km.dat$trt01p==arm.val[1])  ))*(km.dat$trt01p==arm.val[1])*(km.dat$aval>=utime[i])/gH1.r1.denom^3
          
          
        }
        
        ## H0 r1##
        if(gH0.r1.denom > 0){
          ## H0 ##
          H0.r1 <- H0.r1 + sum((nume$trt01p==arm.val[2])*(nume$evnt==1)*nume$pred) / gH0.r1.denom
          ## dH1/dp ##
          gH0.r1 <- gH0.r1 + (((km.dat$aval==utime[i])*(km.dat$trt01p==arm.val[2])*(km.dat$evnt==1)*sum((denom$trt01p==arm.val[2])*denom$pred)) - ( sum((nume$trt01p==arm.val[2])*(nume$evnt==1)*nume$pred)*(km.dat$aval>=utime[i])*(km.dat$trt01p==arm.val[2])  )) / gH0.r1.denom^2
          ## d2H1/dp2 ##
          hH0.r1 <- hH0.r1 + (-2)*(((km.dat$aval==utime[i])*(km.dat$trt01p==arm.val[2])*(km.dat$evnt==1)*sum((denom$trt01p==arm.val[2])*denom$pred)) - ( sum((nume$trt01p==arm.val[2])*(nume$evnt==1)*nume$pred)*(km.dat$aval>=utime[i])*(km.dat$trt01p==arm.val[2])  ))*(km.dat$trt01p==arm.val[2])*(km.dat$aval>=utime[i])/gH0.r1.denom^3
          
        }
        
        #rmst.diff.r1 <- rmst.diff.r1 + (exp(-ch.trt.r1)-exp(-ch.cntl.r1))*dt[i]
        ## H1 r2 ##
        if(gH1.r2.denom > 0){
          ## H1 ##
          H1.r2 <- H1.r2 + sum((nume$trt01p==arm.val[1])*(nume$evnt==1)*(1-nume$pred)) / gH1.r2.denom
          ## dH1/dp ##
          gH1.r2 <- gH1.r2 + ((-1*(km.dat$aval==utime[i])*(km.dat$trt01p==arm.val[1])*(km.dat$evnt==1)*sum((denom$trt01p==arm.val[1])*(1-denom$pred))) - ( sum((nume$trt01p==arm.val[1])*(nume$evnt==1)*(1-nume$pred))*(-1)*(km.dat$aval>=utime[i])*(km.dat$trt01p==arm.val[1])  )) / gH1.r2.denom^2
          ## d2H1/dp2 ##
          hH1.r2 <- hH1.r2 + (2)*((-1*(km.dat$aval==utime[i])*(km.dat$trt01p==arm.val[1])*(km.dat$evnt==1)*sum((denom$trt01p==arm.val[1])*(1-denom$pred))) - ( sum((nume$trt01p==arm.val[1])*(nume$evnt==1)*(1-nume$pred))*(-1)*(km.dat$aval>=utime[i])*(km.dat$trt01p==arm.val[1])  ))*(km.dat$trt01p==arm.val[1])*(km.dat$aval>=utime[i])/gH1.r2.denom^3
          
        }
        
        ## H0 r2 ##
        if(gH0.r2.denom > 0){
          ## H0 ##
          H0.r2 <- H0.r2 + sum((nume$trt01p==arm.val[2])*(nume$evnt==1)*(1-nume$pred)) / gH0.r2.denom
          ## dH1/dp ##
          gH0.r2 <- gH0.r2 + ((-1*(km.dat$aval==utime[i])*(km.dat$trt01p==arm.val[2])*(km.dat$evnt==1)*sum((denom$trt01p==arm.val[2])*(1-denom$pred))) - ( sum((nume$trt01p==arm.val[2])*(nume$evnt==1)*(1-nume$pred))*(-1)*(km.dat$aval>=utime[i])*(km.dat$trt01p==arm.val[2])  )) / gH0.r2.denom^2
          ## d2H1/dp2 ##
          hH0.r2 <- hH0.r2 + 2*((-1*(km.dat$aval==utime[i])*(km.dat$trt01p==arm.val[2])*(km.dat$evnt==1)*sum((denom$trt01p==arm.val[2])*(1-denom$pred))) - ( sum((nume$trt01p==arm.val[2])*(nume$evnt==1)*(1-nume$pred))*(-1)*(km.dat$aval>=utime[i])*(km.dat$trt01p==arm.val[2])  ))*(km.dat$trt01p==arm.val[2])*(km.dat$aval>=utime[i])/gH0.r2.denom^3
          
        }
        
      }
      
      
      rmst.diff.r1 <- rmst.diff.r1 + (exp(-H1.r1)-exp(-H0.r1))*dt[i+1]
      rmst.diff.r2 <- rmst.diff.r2 + (exp(-H1.r2)-exp(-H0.r2))*dt[i+1]
      ## Gradient ##
      rmst.diff.r1.g <- rmst.diff.r1.g + km.dat$predg*(-(exp(-H1.r1)*gH1.r1)+(exp(-H0.r1)*gH0.r1))*dt[i+1]
      rmst.diff.r2.g <- rmst.diff.r2.g + km.dat$predg*(-(exp(-H1.r2)*gH1.r2)+(exp(-H0.r2)*gH0.r2))*dt[i+1]
      
      ## Hessian ##
      rmst.diff.r1.h <- rmst.diff.r1.h + km.dat$predg^2*(exp(-H1.r1)*gH1.r1^2 - exp(-H1.r1)*hH1.r1 - exp(-H0.r1)*gH0.r1^2 + exp(-H0.r1)*hH0.r1)*dt[i+1] + (-(exp(-H1.r1)*gH1.r1)+(exp(-H0.r1)*gH0.r1))* km.dat$predh
      rmst.diff.r2.h <- rmst.diff.r2.h + km.dat$predg^2*(exp(-H1.r2)*gH1.r2^2 - exp(-H1.r2)*hH1.r2 - exp(-H0.r2)*gH0.r2^2 + exp(-H0.r2)*hH0.r2)*dt[i+1] + (-(exp(-H1.r2)*gH1.r2)+(exp(-H0.r2)*gH0.r2))* km.dat$predh
      
    }
    #obj <- 2*( sum(km.dat$pred)*rmst.diff.r1 - sum(1-km.dat$pred)*rmst.diff.r2    )
    g <- (-1)*(sum(km.dat$pred)*rmst.diff.r1.g  - sum(1-km.dat$pred)*rmst.diff.r2.g) 
    h <- (-1)*(sum(km.dat$pred)*rmst.diff.r1.h-sum(1-km.dat$pred)*rmst.diff.r2.h)
    #g.p <- (sum(km.dat$pred)*rmst.diff.r1.g + rmst.diff.r1 - sum(1-km.dat$pred)*rmst.diff.r2.g + rmst.diff.r2)
    #h.p <- (2*rmst.diff.r1.g + sum(km.dat$pred)*rmst.diff.r1.h + 2*rmst.diff.r2.g - sum(1-km.dat$pred)*rmst.diff.r2.h)
    #g <-  km.dat$predg*(-1)*g.p
    #h <- (-1)*( (km.dat$predg)^2 * h.p  + g.p*km.dat$predh)
    #h<- (-2)*(g^2+obj/2*( (km.dat$predg)^2 * h.p  + g.p*km.dat$predh))
    
    
    return(list(grad = g, hess = h))
    #return(list(grad = g, hess = rep(1,n)))
  }
  
  
  
  evalerror <- function(preds, dtrain) {
    ## (0) Decode y to trt01p, aval and evnt ##
    labels <- getinfo(dtrain, "label")
    trt01p<-rep(NA,length(labels))
    evnt<-rep(NA,length(labels))
    aval<-rep(NA,length(labels))
    trt01p[labels< 0] <- 1
    trt01p[labels>= 0] <- 0
    evnt[abs(labels)>= (1000)] <- 1
    evnt[abs(labels)< (1000)] <- 0
    aval[abs(labels)>= (1000)] <- abs(labels[abs(labels)>= (1000)])-1000
    aval[labels<0 & labels>-1000] <- -labels[labels<0 & labels>-1000]-1
    aval[labels>=0 & labels<1000] <- labels[labels>=0 & labels<1000]
    
    arm.val <- c(1,0)
    
    ## (1) Get Time to event Data Ready ##
    km.dat <- data.frame(trt01p,evnt,aval)
    km.dat$pred <- 1/(1+exp(-preds))
    km.dat<-km.dat[order(km.dat$aval),]
    n<-dim(km.dat)[1]
    
    ## (2) Set up gradient and Hessian ##
    utime <- unique(km.dat$aval)
    dt <- utime-c(0,utime[1:length(utime)-1])
    
    rmst.diff.r1 <- 0
    rmst.diff.r2 <- 0
    
    for(i in 0:(length(utime)-1)){
      if(i==0){
        H1.r1 <- 0
        H0.r1 <- 0
        H1.r2 <- 0
        H0.r2 <- 0
        
      } else {
        denom <- subset(km.dat,aval>=utime[i])
        nume <- subset(km.dat,aval==utime[i])
        
        gH1.r1.denom <- sum((denom$trt01p==arm.val[1])*denom$pred)
        gH0.r1.denom <- sum((denom$trt01p==arm.val[2])*denom$pred)
        
        gH1.r2.denom <- sum((denom$trt01p==arm.val[1])*(1-denom$pred))
        gH0.r2.denom <- sum((denom$trt01p==arm.val[2])*(1-denom$pred)) 
        
        ## H1 r1 ##
        if(gH1.r1.denom > 0){
          ## H1 ##
          H1.r1 <- H1.r1 + sum((nume$trt01p==arm.val[1])*(nume$evnt==1)*nume$pred) / gH1.r1.denom
          
        }
        
        ## H0 r1##
        if(gH0.r1.denom > 0){
          ## H0 ##
          H0.r1 <- H0.r1 + sum((nume$trt01p==arm.val[2])*(nume$evnt==1)*nume$pred) / gH0.r1.denom
          
        }
        
        #rmst.diff.r1 <- rmst.diff.r1 + (exp(-ch.trt.r1)-exp(-ch.cntl.r1))*dt[i]
        ## H1 r2 ##
        if(gH1.r2.denom > 0){
          ## H1 ##
          H1.r2 <- H1.r2 + sum((nume$trt01p==arm.val[1])*(nume$evnt==1)*(1-nume$pred)) / gH1.r2.denom
          
        }
        
        ## H0 r2 ##
        if(gH0.r2.denom > 0){
          ## H0 ##
          H0.r2 <- H0.r2 + sum((nume$trt01p==arm.val[2])*(nume$evnt==1)*(1-nume$pred)) / gH0.r2.denom
          
        }
        
      }
      
      
      rmst.diff.r1 <- rmst.diff.r1 + (exp(-H1.r1)-exp(-H0.r1))*dt[i+1]
      rmst.diff.r2 <- rmst.diff.r2 + (exp(-H1.r2)-exp(-H0.r2))*dt[i+1]
      
    }
    
    err <- (-1)*( sum(km.dat$pred)*rmst.diff.r1 - sum(1-km.dat$pred)*rmst.diff.r2    )
    
    return(list(metric = "RMST_error", value = err))
  }
  
  #----------------------------------------------------------------------#
  #                     Step 3: Let's boost                  #
  #----------------------------------------------------------------------#
  col<- c('x.1','x.2','x.3','x.4','x.5','y.1','y.2','y.3','y.4','y.5')
  #col<- c('signal','x.1','x.2','x.3','x.4','x.5')
  dat[col] <- lapply(dat[col], as.factor)
  #dat<-dat[with(dat, order(trt01p,signal,x.1,x.2,x.3,x.4,x.5,y.1,y.2,y.3,y.4,y.5)),]
  sparse_matrix <- Matrix::sparse.model.matrix(~.-1, data = dat)
  

  model <- xgboost(data = sparse_matrix, label = labels, max.depth = 6,eta=0.1,booster='gbtree',
                   nrounds = 10, objective = rmst_loss, eval_metric = evalerror,verbose = 2,nfold=5,early_stopping_rounds = 5)
  
  # model$evaluation_log %>%
  #   dplyr::summarise(
  #     ntrees.train = which(train_RMST_error_mean == min(train_RMST_error_mean))[1],
  #     rmse.train   = min(train_RMST_error_mean),
  #     ntrees.test  = which(test_RMST_error_mean == min(test_RMST_error_mean))[1],
  #     rmse.test   = min(test_RMST_error_mean)
  #     )
  # 
  # ggplot(model$evaluation_log) +
  #   geom_line(aes(iter, train_RMST_error_mean), color = "red") +
  #   geom_line(aes(iter, test_RMST_error_mean), color = "blue")
  
  ### grid search   ###
  
  # create hyperparameter grid
  hyper_grid <- expand.grid(
    eta = c(.01, .05, .1, .3),
    max_depth = c(1, 3, 5, 7),
    min_child_weight = c(1, 3, 5, 7),
    subsample = c(.65, .8, 1), 
    colsample_bytree = c(.8, .9, 1),
    optimal_trees = 0,               # a place to dump results
    min_RMSE = 0                     # a place to dump results
  )
  
  
  for(i in 1:nrow(hyper_grid)) {
    
    # create parameter list
    params <- list(
      eta = hyper_grid$eta[i],
      max_depth = hyper_grid$max_depth[i],
      min_child_weight = hyper_grid$min_child_weight[i],
      subsample = hyper_grid$subsample[i],
      colsample_bytree = hyper_grid$colsample_bytree[i]
    )
    
    # reproducibility
    set.seed(123)
    
    # train model
    xgb.tune <- xgb.cv(
      params = params,
      data = sparse_matrix,
      label = labels,
      nrounds = 1000,
      nfold = 5,
      objective = rmst_loss,  
      verbose = 0,               # silent,
      early_stopping_rounds = 5 # stop if no improvement for 10 consecutive trees
    )
    
    # add min training error and trees to grid
    hyper_grid$optimal_trees[i] <- which.min(xgb.tune$evaluation_log$test_rmse_mean)
    hyper_grid$min_RMSE[i] <- min(xgb.tune$evaluation_log$test_rmse_mean)
  }
  
  hyper_grid %>%
    dplyr::arrange(min_RMSE) %>%
    head(10)

  
  
  ## End of grid search ##
  
  
  
  
  f <- predict(model, sparse_matrix)
  boost.pred<-1/(1+exp(-f))
  #error <- model$evaluation_log
  #plot(error$iter,-1*error$train_RMST_error,xlab='iter',ylab='obj')
  importance_matrix <- xgb.importance(sparse_matrix@Dimnames[[2]], model = model)
  #xgb.plot.importance(importance_matrix)
  #xgb.plot.tree(feature_names = sparse_matrix@Dimnames[[2]], model = model)
  #xgb.plot.multi.trees(model = model, feature_names = sparse_matrix@Dimnames[[2]], features.keep = 3)
  
  res <- c(grepl('signal',importance_matrix$Feature[1]),mean(1*( boost.pred>=0.5)!=dat$signal))
  
  return(res)
  
}




