#!/usr/bin/env python
# coding: utf-8

# In[1]:


# install neptune
install.packages('neptune', dependencies = TRUE)

# install other packages for this tutorial
install.packages(c('digest', 'mlbench', 'randomForest'), dependencies = TRUE)


# In[2]:


reticulate:: install_miniconda()
reticulate:: py_config()


# In[3]:


# load libraries
library(neptune)
library(digest)
library(mlbench)
library(randomForest)

SEED=1234
set.seed(SEED)

# load dataset 
data(Sonar)
dataset <- Sonar
x <- dataset[,1:60]   # predictors
y <- dataset[,61]     # labels


# In[4]:


init_neptune(project_name = 'shared/r-integration',
             api_token = 'ANONYMOUS',
             python_path='/root/.local/share/r-miniconda/envs/r-reticulate/bin/python'
             )


# In[5]:


params = list(ntree=100,
              mtry=10,
              maxnodes=20
              )

create_experiment(name='training on Sonar', 
                  tags=c('random-forest','sonar'),
                  params = params
)


# In[ ]:


set_property(property = 'data-version', value = digest(dataset))
set_property(property = 'seed', value = SEED)


# In[ ]:


model <- randomForest(x = x, y = y,
  ntree=params$ntree, mtry = params$mtry, maxnodes = params$maxnodes,
  importance = TRUE
  )


# In[ ]:


log_metric('mean OOB error', mean(model$err.rate[,1]))
log_metric('error class M', model$confusion[1,3])
log_metric('error class R', model$confusion[2,3])


# In[ ]:


for (err in (model$err.rate[,1])) {
  log_metric('OOB errors', err)
}


# In[ ]:


save(model, file="model.Rdata")
log_artifact('model.Rdata')


# In[ ]:


for (t in c(1,2)){
  jpeg('temp_plot.jpeg')
  varImpPlot(model,type=t)
  dev.off()
  log_image('feature_importance', 'temp_plot.jpeg')
}


# In[ ]:


stop_experiment()


# In[ ]:




