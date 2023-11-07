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
    W[[length(W) + 1]] <- matrix(runif(area,0,0.2), nrow = d[i],ncol = d[i+1])
  }
  
  #create b
  b <- list()
  for(i in 1:(length(d)-1)){
    b[[length(b) + 1]] <- c(runif(d[i+1],0,0.2))
  }
  
  network_list <- list("h" = h, "W" = W, "b" = b)
  return(network_list)
}

forward <- function(nn,inp){
  copy_network <- nn
  
  copy_network$h[[1]] <- inp
  for(i in 1:(length(copy_network$h)-1)){
    copy_network$h[[i+1]] <- pmax(copy_network$W[[i]]%*%copy_network$h[[i]]+copy_network$b[[i]],0)
  }
  
  return(copy_network)
}

backward <- function(nn,k){
  
}

train <- function(nn,inp,k,eta=.01,mb=10,nstep=10000){
  
}