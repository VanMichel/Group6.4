netup <- function(d){
  
  #create h list, append values
  h <- list()
  for(i in 1:length(d)){
    new_vec <- rep(NA,d[i])
    h[[length(h) + 1]] <- new_vec
  }
  
  #create W
  W <- list()
  for(i in 1:(length(d)-1)){
    area <- d[i]* d[i+1]
    W[[length(W) + 1]] <- matrix(runif(area,0,0.2), nrow = d[i+1],ncol = d[i])
  }
  
  #create b
  b <- list()
  for(i in 1:(length(d)-1)){
    b[[length(b) + 1]] <- c(runif(d[i+1],0,0.2))
  }
  
  network_list <- list("h" = h, "W" = W, "b" = b)
  return(network_list)
}



softmax <- function(vec_out){
  copy_vec <- vec_out
  
  for(i in 1:length(copy_vec)){
    copy_vec[i] <- exp(vec_out[i])/sum(exp(vec_out))
  }
  
  return(copy_vec)
}




forward <- function(nn,inp){
  copy_network <- nn
  
  copy_network$h[[1]] <- inp
  for(i in 1:(length(copy_network$h)-1)){
    
    #apply relu to hidden layers only
    if(i!=(length(copy_network$h)-1)){
      copy_network$h[[i+1]] <- pmax(copy_network$W[[i]]%*%copy_network$h[[i]]+copy_network$b[[i]],0)
    }else{ #and softmax to the output layer
      copy_network$h[[i+1]] <- softmax(copy_network$W[[i]]%*%copy_network$h[[i]]+copy_network$b[[i]])
    }
    
  }
  
  return(copy_network)
}



backward <- function(nn,k){
  
  #label_pos <- which(k==1)
  
  #if k is an integer with the class number: 
  loss_deriv <- tail(nn$h,1)[[1]]
  loss_deriv[k] <- loss_deriv[k] - 1
  
  #calculate the loss for the current inp vector
  #loss_deriv <- tail(nn$h,1)[[1]]-k
  
  dh <- list()
  dh[[length(nn$W)+1]] <- loss_deriv
  
  dW <- list()
  db <- list()
  
  #backpropagate the loss
  for(i in rev(seq(1:length(nn$W)))){
    
    dh[[i]] <- t(nn$W[[i]])%*%pmax(dh[[i+1]],0)
    dW[[i]] <- pmax(dh[[i+1]],0)%*%t(nn$h[[i]])
    db[[i]] <- pmax(dh[[i+1]],0)
  }
  
  #add derivatives to the results list
  derivatives <- list("dh" = dh, "dW" = dW, "db" = db)
  return(c(nn, derivatives))
}



train <- function(nn,inp,k,eta=.01,mb=10,nstep=10000){
  
  #inp is a matrix whose rows are the datapoints
  #k is the corresponding vector of classes 1,2,3,..
  
  copy_network <- nn

  #lists for the gradients used to update W and b
  dLdW <- list()
  dLdb <- list()
  
  #initialize to zero 
  for(i in 1:length(copy_network$W)){
    dLdW[[i]] <- 0
    dLdb[[i]] <- 0
  }
    

  for (n in 1:nstep) {
    
    if(mb == 1) {
      isample <- sample(1:nrow(inp), mb)
      xdat <- inp[isample,]
      kdat <- k[isample]
      
      #compute node values
      copy_network <- forward(copy_network,xdat)
      #run backward to compute the gradients
      copy_network_grads <- backward(copy_network, kdat)
      
      for(i in 1:length(copy_network$W)){
        dLdW[[i]] <- copy_network_grads$dW[[i]]
        dLdb[[i]] <- copy_network_grads$db[[i]]
      }
      
    }
    
    else{
      #sample mb datapoint for gradient calculation
      isample <- sample(1:nrow(inp), mb)
      xdat <- inp[isample,]
      kdat <- k[isample]
    
      for (i in 1:mb){
        #compute node values
        copy_network <- forward(copy_network,xdat[i,])
        #run backward to compute the gradients
        copy_network_grads <- backward(copy_network, kdat[i])
      
        #sum of gradients up to current i
        for(j in 1:length(copy_network$W)){
        dLdW[[j]] <- dLdW[[j]] + copy_network_grads$dW[[j]]
        dLdb[[j]] <- dLdb[[j]] + copy_network_grads$db[[j]]
      }
    }
    }
    
    #update W and b in copy_network using the average of the mb gradients
    for(j in 1:length(copy_network$W)){
    copy_network$W[[j]] <- copy_network$W[[j]] - eta*dLdW[[j]]/mb 
    copy_network$b[[j]] <- copy_network$b[[j]] - eta*dLdb[[j]]/mb 
    }
    
    #reset gradients to zero for new gradient calculation
    for(j in 1:length(copy_network$W)){
      dLdW[[j]] <- 0
      dLdb[[j]] <- 0
    }
  }
  
  #return the trained network
  return(copy_network)
}


iris_values <- cbind(iris[[1]],iris[[2]],iris[[3]],iris[[4]])
iris_classes <- rep(1,length(iris[[5]]))
iris_classes[grep("versicolor",iris[[5]])] <- 2
iris_classes[grep("virginica",iris[[5]])] <- 3

ii_test <- which((1:length(iris_classes))%%5==0)
test_values <- iris_values[ii_test,]
test_classes <- iris_classes[ii_test]
predicted <- rep(0,length(ii_test))

train_values <- iris_values[-ii_test,]
train_classes <- iris_classes[-ii_test]
trained_network <- train(netup(c(4,8,7,3)),train_values,train_classes,mb=10)

for (i in 1:length(ii_test)) {
  result <- softmax(forward(trained_network,test_values[i,])$h[[4]])
  predicted[i] <- which(result==max(result))
}

predicted
test_classes
