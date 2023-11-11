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
  label_pos <- which(k==1)
  
  #calculate the loss for the current inp vector
  loss_deriv <- tail(nn$h,1)[[1]]-k
  
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
  
}