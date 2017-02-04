
# ~~~~~~~~~~~~~~~~~~~~REQUIRED PACKAGES FOR THE MODEL~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

library(readr)
library(dplyr)
library(reshape2)
library(Matrix)
library(moments)
library(stringr)
library(xgboost)
library(moments)
library(mice)
library(VIM)
library(FSelector)
library(lubridate)
library(mlr)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LOAD THE DATA~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

getwd()
setwd("C:/Users/ruthv/OneDrive/BAPM/Self study/Kaggle/Rossman")

Train_rossman = fread("train.csv",nrows = -1,na.strings = "",stringsAsFactors = T)
Train_store = fread("store.csv",nrows = -1,na.strings = "",stringsAsFactors = T)
Test_rossman = fread("test.csv",nrows = -1,na.strings = "",stringsAsFactors = T)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~REMOVING THE VARIABLES MENTIONED BELOW~~~~~~~~~~~~~~~~~~~~~~

Train_rossman$Customers = NULL
Test_rossman$Id = NULL

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ADDING A NEW COLUMN SALES FOR THE ROW BIND TO WORK~~~~~~~~~~~

Test_rossman[,"Sales"] = sample(x = 0:1,size = dim(Test_rossman)[1],replace = TRUE)

Total_data = rbind(Train_rossman,Test_rossman)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~JOINED THE TWO TABLES STORE AND TRAIN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

Total_combined = full_join(Total_data,Train_store,by = "Store")


#~~~~~~~~~~~~~~~~~~~~~~~~~~LOOKING AT THE STRUCTURE OF THE DATA~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

str(Total_combined)

#~~~~~~~~~~~~~~~~~~~~~CONVERTING EACH VARIABLE TO IT'S RESPECTIVE DATA TYPE~~~~~~~~~~~~~~~~~~~~~~~~

Total_combined$DayOfWeek = as.factor(Total_combined$DayOfWeek)
Total_combined$Open = as.factor(Total_combined$Open)
Total_combined$Promo = as.factor(Total_combined$Promo)
Total_combined$StateHoliday = as.factor(Total_combined$StateHoliday)
Total_combined$SchoolHoliday = as.factor(Total_combined$SchoolHoliday)
Total_combined$StoreType = as.factor(Total_combined$StoreType)
Total_combined$Assortment = as.factor(Total_combined$Assortment)
Total_combined$Promo2 = as.factor(Total_combined$Promo2)

str(Total_combined)

#~~~~~~~~~~~~~~~~~~~~~~~~~dIVIDING THE DATE COLUMN INTO DAY,MONTH AND YEAR~~~~~~~~~~~~~~~~~~~~~

Total_combined$Date = as_date(Total_combined$Date)
Total_combined$Day = day(Total_combined$Date)
Total_combined$Month = month(Total_combined$Date)
Total_combined$Year = year(Total_combined$Date)

#~~~~~~~~~~~~~~~~~~~~~~~~~CHANGING THE DATA TYPES FOR THE DATE VARIABLES

str(Total_combined)
Total_combined$Day = as.factor(Total_combined$Day)
Total_combined$Month = as.factor(Total_combined$Month)
Total_combined$Year = as.factor(Total_combined$Year)

#~~~~~~~~~~~~~~~~~~~~~~~~IMPUTING THE MISSING VALUES IN COMPETETION DISTANCE VARIABLE WITH 0
# SINCE THERE IS A POSSIBILITY THAT THERE ARE NO COMPETITION STORES NEARBY~~~~~~~~~~~

which(is.na(Total_combined$CompetitionDistance))
Total_combined$CompetitionDistance[is.na(Total_combined$CompetitionDistance)] = 0

#~~~~~~~~~~~~~~~~~~~~~~~~IMPUTING THE MISSING VALUES IN PROMO2 SINCE WEEK VARIABLE WITH 0 SINCE
# IF THERE IS NO PROMO2 THEN THAT PROMO WOULD NOT REMAIN IN THE COMING WEEKS~~~~~~~~~~~~~~~~~~~~~

which(is.na(Total_combined$Promo2SinceWeek))

Total_combined$Promo2SinceWeek = as.integer(Total_combined$Promo2SinceWeek)
Total_combined$Promo2SinceWeek[is.na(Total_combined$Promo2SinceWeek)] = 0

#~~~~~~~~~~~~~~~~~~~~~~~~IMPUTING THE MISSING VALUES IN Promo2SinceYear VARIABLE WITH 0 SINCE
# IF THERE IS NO PROMO2 THEN THAT PROMO WOULD NOT REMAIN IN THE COMING WEEKS~~~~~~~~~~~~~~~~~~~~~

Total_combined$Promo2SinceYear = as.integer(Total_combined$Promo2SinceYear)
Total_combined$Promo2SinceYear[is.na(Total_combined$Promo2SinceYear)] = 0

#~~~~~~~~~~~~~~~~~~~~~~~~IMPUTING THE MISSING VALUES IN PromoInterval VARIABLE WITH 0 SINCE
# IF THERE IS NO PROMO2 THEN THAT PROMO WILL NOT BE ANY PROMO INTERVAL~~~~~~~~~~~~~~~~~~~~~

Total_combined$PromoInterval = as.character(Total_combined$PromoInterval)
Total_combined$PromoInterval[is.na(Total_combined$PromoInterval)] = 0

#~~~~~~~~~~~~~~~~~~~~~~~~IMPUTING THE MISSING VALUES IN CompetitionOpenSinceMonth VARIABLE WITH 0 SINCE
# THE COMPETITION MIGHT HAVE JUST STARTED THE STORE~~~~~~~~~~~~~~~~~~~~~

Total_combined$CompetitionOpenSinceMonth[is.na(Total_combined$CompetitionOpenSinceMonth)] = mean(Total_combined$CompetitionOpenSinceMonth[!is.na(Total_combined$CompetitionOpenSinceMonth)])
Total_combined$CompetitionOpenSinceMonth = as.factor(Total_combined$CompetitionOpenSinceMonth)

#~~~~~~~~~~~~~~~~~~~~~~~~IMPUTING THE MISSING VALUES IN CompetitionOpenSinceYear VARIABLE WITH 0 SINCE
# THE COMPETITION MIGHT HAVE JUST STARTED THE STORE~~~~~~~~~~~~~~~~~~~~~

Total_combined$CompetitionOpenSinceYear[is.na(Total_combined$CompetitionOpenSinceYear)] = mean(Total_combined$CompetitionOpenSinceYear[!is.na(Total_combined$CompetitionOpenSinceYear)])
Total_combined$CompetitionOpenSinceYear = as.factor(Total_combined$CompetitionOpenSinceYear)

#~~~~~~~~~~~~~~~~IMPUTING A FEW OBSERVATIONS IN OPEN VARIABLE WITH 0 SINCE IT HAS THE MAXIMUM
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~COUNT~~~~~~~~~~~~~~~~~~~~``

which(is.na(Total_combined$Open))

which((Total_combined$Open[1:1017209] == "0"))
table(Total_combined$Open)
Total_combined$Open[is.na(Total_combined$Open)] = 0

Total_combined$Promo2SinceWeek = as.factor(Total_combined$Promo2SinceWeek)
Total_combined$Promo2SinceYear = as.factor(Total_combined$Promo2SinceYear)
Total_combined$PromoInterval = as.factor(Total_combined$PromoInterval)

#~~~~~~~~~~~~~~~~~~~~~~LOOKING AT THE VARAIBLES ONCE MORE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

str(Total_combined)
summarizeColumns(Total_combined)

#~~~~~~~~~~~~~~~~~~~~~REMOVING THE VARIABLES BELOW DUE TO MULTICOLINEARITY BETWEEN THEM AND THE
#~~~~~~~~~~~~~~~~~~~~MODEL VARIABLES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``

nonimpvar = c("Year","SchoolHoliday","StateHoliday","Open","Promo2SinceYear","PromoInterval","Promo2","CompetitionOpenSinceMonth","Date","Store")

Total_combined1 = Total_combined[,!(names(Total_combined)) %in% nonimpvar]
dim(Total_combined1)
summarizeColumns(Total_combined1)

#~~~~~~~~~~~~~~~~~~~~~~sPLITTING THE DATA BACK INTO TRAIN AND TEST~~~~~~~~~~~~~~~~~~~~~~~~~

Train_total = Total_combined1[1:1017209,]
Test_total = Total_combined1[1017210:1058297,]
dim(Train_total)
dim(Test_total)
str(Train_total)



#~~~~~~~~~~~~~~~~~~~~~~~~~SHUFFLING THE OBSERVATIONS FOR TRAINING AND VALIDATION~~~~~~~~~~~~~~~~

Train_total1 = Train_total[sample(nrow(Train_total)),]

#~~~~~~~~~~~~~~~~~~~~~~~~~REMOVING ALL THE OBSERVATIONS WITH SALES 0 SINCE THEY WOULD MAKE OUR 
# PREDICTIONS MORE SKEWED AND further more IT WAS MENTIONED IN THE COMPETITION description 
# TO REMOve THEM.

Train_total2 = Train_total1[Train_total1$Sales > 0,]

#~~~~~~~~~~~~~~~~~~~~THE SALES AFTER THE O SALES OBSERVATIONS HAVE BEEN REMOVED SEEMS TO BE RIGHT
# SKEEWED. HENCE DOING A LOG TRANSFORMATION.

Train_total2$Sales = log(Train_total2$Sales)


dim(Train_total2)
remove(Test_rossman)
remove(Total_combined)
remove(Total_data)
remove(Train_rossman)
remove(Train_store)

Total_combined1$Day = as.integer(Total_combined1$Day)
Total_combined1$Promo2SinceWeek = as.integer(Total_combined1$Promo2SinceWeek)
Total_combined1$Promo = as.integer(Total_combined1$Promo)
Total_combined1$CompetitionOpenSinceYear = as.integer(Total_combined1$CompetitionOpenSinceYear)
Total_combined1$DayOfWeek = as.integer(Total_combined1$DayOfWeek)
Total_combined1$Month = as.integer(Total_combined1$Month)
Train_total2$Month = as.integer(Train_total2$Month)

varimp = information.gain(formula = Sales~CompetitionOpenSinceYear,data = Train_total2)
varimp


# ~~~~~~~~~~~~~~~~~~~~`SAMPLING THE TRAIN FOR FASTER PROCESSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
sample = sample(nrow(Train_total2),size = 300000,replace = FALSE)
Train_sample = Train_total2[sample,]
Train_rest_sample = Train_total2[-sample,]
sample2 = sample(nrow(Train_rest_sample),size = 300000,replace = FALSE)
Valid_sample = Train_rest_sample[sample2,]
Train_rest_sample1 = Train_rest_sample[-sample2,]
sample3 = sample(nrow(Train_rest_sample1),size = 10000,replace = FALSE)
Test_sample = Train_rest_sample1[sample3,]

varimp = information.gain(formula = Sales~.,data = Train_sample)
varimp

write_csv(Train_sample,"Train_sample_rossman.csv")
write_csv(Test_sample,"Test_sample_rossman.csv")
write_csv(Valid_sample,"Valid_sample_rossman.csv")
write_csv(Train_total2,"Train_total_rossman.csv")
write_csv(Test_total,"Test_total_rossman.csv")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~RANDOMFOREST IMPLEMENTATION USING H2O PACKAGE~~~~~~~~~~~~~~~~~~~
library(h2o)
h2o_init = h2o.init(nthreads = 8,max_mem_size = "10g")

#~~~~~~~~~~~~~~~CONVERTING THE DATA INTO H2O DATA FRAMES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``

f = names(Train_sample)
t = c("Sales","CompetitionDistance","DayOfWeek","Promo","Day")
Train_sample_h2o = as.h2o(Train_sample)
Valid_sample_h2o = as.h2o(Valid_sample)
Test_sample_h2o = as.h2o(Test_sample)
Test_total_h2o = as.h2o(Test_total)
Train_total_h2o = as.h2o(Train_total2)

#~~~~~~~~~~~~~~~~~~~~~~~~~`BUILDING THE MODEL~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RF_Train_sample_h2o = h2o.randomForest(x = f,y = "Sales",
                                       training_frame = Train_sample_h2o,nfolds = 5,ntrees = 50
                                       ,mtries = 4,max_depth = 30,seed = 100)

#~~~~~~~~~~~~~~~~~~~~~PLOTTING THE MODEL WITH RESPECT TO NUMBER OF TREES AND RMSE. WE COULD SEE
# THAT INCREASING THE NUMBER OF TREES ISN'T DOING A BETER JOB FOR THE MODEL. INCREASING THE DEPTH 
#  AND MTRIES IS DOING A BETTER JOB. DUE TO LACK OF SUFFICIENT COMPUTATIONAL POWER FURTHER TUNING 
# HAS BEEN HARD TO DO. HENCE THE RMSPE ENDED UP AT 0.14. WITH ENOUGH COMPUTATIONAL POWER THIS CAN 
# FURTHER BE REDUCED TO CLOSE TO 0.11.

plot(RF_Train_sample_h2o)

# ~~~~~~~~~~~~~~PREDICTING THE VALIDATION DATA THAT HAS BEEN SPLIT BEFORE~~~~~~~~~~~~~~~~~~~~~~~~`` 

pred_valid = predict(RF_Train_sample_h2o,Valid_sample_h2o)
pred_valid = as.data.frame(pred_valid)
names(pred_valid) = "Sales"

#~~~~~~~~~~~~~~~~~~~~~~~~~~ CALCULATING THE RMSPE FOR MODEL ASSESSMENT. USING THIS THE MODEL PARAMETERS CAN BE  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RMSPE =  sqrt(mean(((exp(Valid_sample$Sales) - exp(pred_valid))/exp(Valid_sample$Sales))^2))
RMSPE

#~~~~~~~~~~~~~~~~~~~~~~~~PREDICTING THE ACTUAL TEST DATA AND SUBMITING IT~~~~~~~~~~~~~~~~~~~~~

Test_final_pred <- predict(RF_Train_sample_h2o, Test_total_h2o)
Test_final_pred = exp(Test_final_pred)
names(Test_final_pred) = "Sales"
Test1 <- read_csv("test.csv",col_names = T)
submit = data.frame(Id = Test1$Id,Sales = as.data.frame(Test_final_pred$Sales))
write.csv(submit, "submit_Test_full_h2o_RF.csv",row.names = F)


