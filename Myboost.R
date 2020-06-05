##---------------------------##
#     Input Parameters       #
##---------------------------##

## (1) dat: data frame with predictors, treatment(1 for treatment and 0 for control), time to event (in month) and event indictor (1 for event 0 for censoring)

##---------------------------##
#     Return Values          #
##---------------------------##

# a Xgboost model


##---------------------------------------------------##
#    Customized loss and evaluation function          #
##---------------------------------------------------##

library(xgboost)


MyBoost <- function(dat){

  ### little trick to embed trt01p,aval and evnt into labels for Xgboost input  ###
  N <- nrow(dat)
  labels <- rep(NA,N)
  labels[dat$trt01p==1 & dat$evnt==1] <- -1000-dat$aval[dat$trt01p==1 & dat$evnt==1] 
  labels[dat$trt01p==1 & dat$evnt==0] <- -1-dat$aval[dat$trt01p==1 & dat$evnt==0] 
  labels[dat$trt01p==0 & dat$evnt==1] <- 1000+dat$aval[dat$trt01p==0 & dat$evnt==1] 
  labels[dat$trt01p==0 & dat$evnt==0] <- dat$aval[dat$trt01p==0 & dat$evnt==0] 
  
  dat$evnt <- NULL; dat$aval <- NULL; dat$trt01p <- NULL;


  ###  Customized Loss and Error Function ###
  
  Myloss <- function(preds, dtrain) {
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
    n<-dim(km.dat)[1]
    km.dat$id<-c(1:n)
    km.dat$f <- preds
    km.dat$pred <- 1/(1+exp(-preds))
    km.dat$predg <- exp(preds)/(1+exp(preds))^2
    km.dat$predh <- exp(preds)*(1-exp(preds))/(1+exp(preds))^3
    km.dat<-km.dat[order(km.dat$aval),]
    
    
    ## (2) Set up gradient and Hessian ##
    utime <- unique(km.dat$aval[km.dat$evnt==1])
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
        nume <- subset(km.dat,aval==utime[i] & evnt==1)
        
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
          #hH1.r1 <- hH1.r1 + (-2)*(((km.dat$aval==utime[i])*(km.dat$trt01p==arm.val[1])*(km.dat$evnt==1)*sum((denom$trt01p==arm.val[1])*denom$pred)) - ( sum((nume$trt01p==arm.val[1])*(nume$evnt==1)*nume$pred)*(km.dat$aval>=utime[i])*(km.dat$trt01p==arm.val[1])  ))*(km.dat$trt01p==arm.val[1])*(km.dat$aval>=utime[i])/gH1.r1.denom^3
          
          
        }
        
        ## H0 r1##
        if(gH0.r1.denom > 0){
          ## H0 ##
          H0.r1 <- H0.r1 + sum((nume$trt01p==arm.val[2])*(nume$evnt==1)*nume$pred) / gH0.r1.denom
          ## dH1/dp ##
          gH0.r1 <- gH0.r1 + (((km.dat$aval==utime[i])*(km.dat$trt01p==arm.val[2])*(km.dat$evnt==1)*sum((denom$trt01p==arm.val[2])*denom$pred)) - ( sum((nume$trt01p==arm.val[2])*(nume$evnt==1)*nume$pred)*(km.dat$aval>=utime[i])*(km.dat$trt01p==arm.val[2])  )) / gH0.r1.denom^2
          ## d2H1/dp2 ##
          #hH0.r1 <- hH0.r1 + (-2)*(((km.dat$aval==utime[i])*(km.dat$trt01p==arm.val[2])*(km.dat$evnt==1)*sum((denom$trt01p==arm.val[2])*denom$pred)) - ( sum((nume$trt01p==arm.val[2])*(nume$evnt==1)*nume$pred)*(km.dat$aval>=utime[i])*(km.dat$trt01p==arm.val[2])  ))*(km.dat$trt01p==arm.val[2])*(km.dat$aval>=utime[i])/gH0.r1.denom^3
          
        }
        
        #rmst.diff.r1 <- rmst.diff.r1 + (exp(-ch.trt.r1)-exp(-ch.cntl.r1))*dt[i]
        ## H1 r2 ##
        if(gH1.r2.denom > 0){
          ## H1 ##
          H1.r2 <- H1.r2 + sum((nume$trt01p==arm.val[1])*(nume$evnt==1)*(1-nume$pred)) / gH1.r2.denom
          ## dH1/dp ##
          gH1.r2 <- gH1.r2 + ((-1*(km.dat$aval==utime[i])*(km.dat$trt01p==arm.val[1])*(km.dat$evnt==1)*sum((denom$trt01p==arm.val[1])*(1-denom$pred))) - ( sum((nume$trt01p==arm.val[1])*(nume$evnt==1)*(1-nume$pred))*(-1)*(km.dat$aval>=utime[i])*(km.dat$trt01p==arm.val[1])  )) / gH1.r2.denom^2
          ## d2H1/dp2 ##
          #hH1.r2 <- hH1.r2 + (2)*((-1*(km.dat$aval==utime[i])*(km.dat$trt01p==arm.val[1])*(km.dat$evnt==1)*sum((denom$trt01p==arm.val[1])*(1-denom$pred))) - ( sum((nume$trt01p==arm.val[1])*(nume$evnt==1)*(1-nume$pred))*(-1)*(km.dat$aval>=utime[i])*(km.dat$trt01p==arm.val[1])  ))*(km.dat$trt01p==arm.val[1])*(km.dat$aval>=utime[i])/gH1.r2.denom^3
          
        }
        
        ## H0 r2 ##
        if(gH0.r2.denom > 0){
          ## H0 ##
          H0.r2 <- H0.r2 + sum((nume$trt01p==arm.val[2])*(nume$evnt==1)*(1-nume$pred)) / gH0.r2.denom
          ## dH1/dp ##
          gH0.r2 <- gH0.r2 + ((-1*(km.dat$aval==utime[i])*(km.dat$trt01p==arm.val[2])*(km.dat$evnt==1)*sum((denom$trt01p==arm.val[2])*(1-denom$pred))) - ( sum((nume$trt01p==arm.val[2])*(nume$evnt==1)*(1-nume$pred))*(-1)*(km.dat$aval>=utime[i])*(km.dat$trt01p==arm.val[2])  )) / gH0.r2.denom^2
          ## d2H1/dp2 ##
          #hH0.r2 <- hH0.r2 + 2*((-1*(km.dat$aval==utime[i])*(km.dat$trt01p==arm.val[2])*(km.dat$evnt==1)*sum((denom$trt01p==arm.val[2])*(1-denom$pred))) - ( sum((nume$trt01p==arm.val[2])*(nume$evnt==1)*(1-nume$pred))*(-1)*(km.dat$aval>=utime[i])*(km.dat$trt01p==arm.val[2])  ))*(km.dat$trt01p==arm.val[2])*(km.dat$aval>=utime[i])/gH0.r2.denom^3
          
        }
        
      }
      
      
      rmst.diff.r1 <- rmst.diff.r1 + (exp(-H1.r1)-exp(-H0.r1))*dt[i+1]
      rmst.diff.r2 <- rmst.diff.r2 + (exp(-H1.r2)-exp(-H0.r2))*dt[i+1]
      ## Gradient ##
      rmst.diff.r1.g <- rmst.diff.r1.g + (-(exp(-H1.r1)*gH1.r1)+(exp(-H0.r1)*gH0.r1))*dt[i+1]
      rmst.diff.r2.g <- rmst.diff.r2.g + (-(exp(-H1.r2)*gH1.r2)+(exp(-H0.r2)*gH0.r2))*dt[i+1]
      
      ## Hessian ##
      #rmst.diff.r1.h <- rmst.diff.r1.h + (exp(-H1.r1)*gH1.r1^2 - exp(-H1.r1)*hH1.r1 - exp(-H0.r1)*gH0.r1^2 + exp(-H0.r1)*hH0.r1)*dt[i+1] 
      #rmst.diff.r2.h <- rmst.diff.r2.h + (exp(-H1.r2)*gH1.r2^2 - exp(-H1.r2)*hH1.r2 - exp(-H0.r2)*gH0.r2^2 + exp(-H0.r2)*hH0.r2)*dt[i+1]
      
    }
    
    
    g.p <- (sum(km.dat$pred)*rmst.diff.r1.g + rmst.diff.r1 - sum(1-km.dat$pred)*rmst.diff.r2.g + rmst.diff.r2)
    #h.p <- (2*rmst.diff.r1.g + sum(km.dat$pred)*rmst.diff.r1.h + 2*rmst.diff.r2.g - sum(1-km.dat$pred)*rmst.diff.r2.h)
    g <-  km.dat$predg*(-1)*g.p
    #h <- (-1)*( (km.dat$predg)^2 * h.p  + g.p*km.dat$predh)
    g <- g[order(km.dat$id)]
    #h <- h[order(km.dat$id)]
    h<-rep(0.00001,n)
    
    
    return(list(grad = g, hess = h))

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
    utime <- unique(km.dat$aval[km.dat$evnt==1])
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
        nume <- subset(km.dat,aval==utime[i] & evnt==1)
        
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
    
    return(list(metric = "OTR_error", value = err))
  }
  
  
  ### Let's boost ###
  
  # Grid Search #
  
  hyper_grid <- expand.grid(
    eta = c(0.001,.005, .01, .05, .1),
    max_depth = c(2,4,6),
    #subsample = c(.65, .8, 1),
    #colsample_bytree = c(.8, 1),
    #lambda =c(1,3,5),
    optimal_trees = 0,
    min_error = 0
  )
  
 cat("CV to find the optimal parameter setting \n")
  for(i in 1:nrow(hyper_grid)) {
    print(i)
    
    # create parameter list #
    params <- list(
      eta = hyper_grid$eta[i],
      max_depth = hyper_grid$max_depth[i],
      lambda = 1,
      min_child_weight = 0,
      subsample = 1,
      colsample_bytree = 1
    )
    
    
    # train model
    xgb.tune <- xgb.cv(
      params = params,
      data = as.matrix(dat),
      label = labels,
      nrounds = 500,
      nfold = 5,
      objective = Myloss,
      eval_metric = evalerror,
      maximize=F,
      verbose = 0,               # silent,
      early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
    )
    
    # add min training error and trees to grid
    hyper_grid$optimal_trees[i] <- which.min(xgb.tune$evaluation_log$test_OTR_error_mean)
    hyper_grid$min_error[i] <- min(xgb.tune$evaluation_log$test_OTR_error_mean)
  }
  
  
  ### Train a model based on the best CV parameter set ###
  
  hyper_grid <- hyper_grid[order(hyper_grid$min_error),]
  cat(" Top Five Fitting per CV \n")
  print (hyper_grid[1:5,])
  dtrain <- xgb.DMatrix(as.matrix(dat),label = labels)
  param <- list(max_depth = hyper_grid$max_depth[1], eta = hyper_grid$eta[1], silent = 1, 
                objective = Myloss, eval_metric = evalerror,verbose = 1,lambda=1,base_score=0,colsample_bytree=1,min_child_weight=0)
  watchlist <- list(train = dtrain)
  
  cat("Train Model based on Optimal Parameter Setting from CV \n")
  model <- xgb.train(param, dtrain, nrounds = hyper_grid$optimal_trees[1],watchlist)
  
  cat('Model Fitting Finished \n')
  
  return(model)
  
}
  
