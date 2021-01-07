install.packages('neptune', dependencies = TRUE)

library(neptune)

init_neptune(project_name = 'shared/r-integration',
             api_token = 'ANONYMOUS',
            )

create_experiment(name='minimal example')

log_metric('accuracy', 0.92)

for (i in 0:100){
  log_metric('random_training_metric', i * 0.6)
}

stop_experiment()

install.packages(c('digest', 'mlbench', 'randomForest'), dependencies = TRUE)

library(digest)
library(mlbench)
library(randomForest)

# load dataset
data(Sonar)
dataset <- Sonar
x <- dataset[,1:60]   # predictors
y <- dataset[,61]     # labels

params = list(ntree=625,
              mtry=13,
              maxnodes=50
              )

create_experiment(name='training on Sonar',
                  params = params
)

set_property(property = 'data-version', value = digest(dataset))

SEED=1234
set.seed(SEED)
set_property(property = 'seed', value = SEED)

model <- randomForest(x = x, y = y,
  ntree=params$ntree, mtry = params$mtry, maxnodes = params$maxnodes,
  importance = TRUE
  )

log_metric('mean OOB error', mean(model$err.rate[,1]))
log_metric('error class M', model$confusion[1,3])
log_metric('error class R', model$confusion[2,3])

for (err in (model$err.rate[,1])) {
  log_metric('OOB errors', err)
}

save(model, file="model.Rdata")
log_artifact('model.Rdata')

for (t in c(1,2)){
  jpeg('importance_plot.jpeg')
  varImpPlot(model,type=t)
  dev.off()
  log_image('feature_importance', 'importance_plot.jpeg')
}

stop_experiment()