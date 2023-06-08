rm(list=ls(all=TRUE))
setwd("C:\\Users\\SachinSHARMA\\OneDrive - CIMMYT\\CIMMYT\\Sharing")

#Finalized datasets
wheat_18 <- read.csv("CSISA_IND_LDS_Whe_2018_Data.csv", header=T, stringsAsFactors = F)
dim(wheat_18)


library(plyr)
library(dplyr)
library(tidyr)
library(lattice)
library(ggplot2)
library(gridExtra)
library(vcd)
library(gmodels)
library(epiR)
library(data.table)
library(leaflet)
library(reshape)
library(agricolae)
library(corrplot)
library(randomForest)
library(ggmap)
library(ggplot2)
library(sp)
library(ggmap)
library(sp)
library(ggplot2)
library(devtools)
library(lattice)
library(MASS)
library(neuralnet)
library(gridExtra)
library(plotrix)
library(devtools)
library(rpart)
library(rpart.plot)
library(lattice)
library(splitstackshape)

wheat_18UPB <- wheat_18
dim(wheat_18UPB)
table(wheat_18UPB$A.q102_state)
dput(names(wheat_18UPB))
colnames(wheat_18UPB)<-sub("^[^.]*.","",colnames(wheat_18UPB)) # Removing redundant characters from the variable names
mf=subset(wheat_18UPB, select= c("q101_country", "q102_state", 
                                 "q103_district", "q104_subDistrict", "q105_village", 
                                 "q111_fGender", "q112_fEdu", "q301_LLU", 
                                 "q302_acreConv", "q303_cultLand", "q304_cropCultLand", "q305_cropLargestArea", 
                                 "q306_cropLarestAreaAcre", "q307_largestPlotType", "q401_soilTexture", 
                                 "q402_drainClass", "q403_soilPerception", 
                                 "q404_prevCrop", "q405_prevCropTillage", 
                                 "q406_prevCropHarvest", "q407_cropResiduePcnt", "q408_residueBurnt", 
                                 "q409_varType", "q410_varName", "q411_LandPrep", 
                                 "q412_RotUseYear", "q413_CropEst",  
                                 "q414_rcNursEstDate", "q415_seedingSowingTransDate", 
                                 "q416_whSDelayReason", "q417_whComDelayReason", "q418_nursDetFactor", "q419_transDetFactor", 
                                 "q420_cropSeedAmt", "q421_seedSource", "q422_otherSeedSource", 
                                 "q5105_statSHC", "q5101_FYM", "q5102_typeFYM", "q5103_amtFYM", 
                                 "q5104_applyMineralFert", "q5202_basalDAP", "q5203_basalNPK", 
                                 "q5204_basalUrea", "q5205_basalNPKS", "q5206_basalMoP", "q5207_basalSSP", 
                                 "q5208_basalTSP", "q5209_basalZnSO4", "q5210_basalGypsum", "q5211_basalBoron", 
                                 "q52111_otherBasalFert", "q52112_otherBasalFertAmt",  
                                 "q5212_1tdDAP", "q5213_1appDaysDAP", "q5214_1tdNPK", "q5215_1appDaysNPK", 
                                 "q5216_1tdUrea", "q5217_1appDaysUrea", "q5218_1tdNPKS", "q5219_1appDaysNPKS", 
                                 "q5220_1tdMoP", "q5221_1appDaysMoP", "q5223_1tdSSP", "q5224_1appDaysSSP", 
                                 "q5225_1tdTSP", "q5226_1appDaysTSP", "q5227_1tdZnSO4", "q5228_1appDaysZnSO4", 
                                 "q5229_1tdGypsum", "q5230_1appDaysGypsum", "q5231_1tdBoron", 
                                 "q5232_1appDaysBoron", "q52321_1tdOtherFert", "q52322_1tdOtherFertAmt", 
                                 "q52323_1appDaysOtherFert", "q5233_2tdDAP", 
                                 "q5234_2appDaysDAP", "q5235_2tdNPK", "q5236_2appDaysNPK", "q5237_2tdUrea", 
                                 "q5238_2appDaysUrea", "q5239_2tdNPKS", "q5240_2appDaysNPKS", 
                                 "q5241_2tdMoP", "q5242_2appDaysMoP", "q5243_2tdSSP", "q5244_2appDaysSSP", 
                                 "q5245_2tdTSP", "q5246_2appDaysTSP", "q5247_2tdZnSO4", "q5248_2appDaysZnSO4", 
                                 "q5249_2tdGypsum", "q5250_2appDaysGypsum", "q5251_2tdBoron", 
                                 "q5252_2appDaysBoron", "q52521_2tdOtherFert", "q52522_2tdOtherFertAmt", 
                                 "q52523_2appDaysOtherFert", "q5253_3tdDAP", 
                                 "q5254_3appDaysDAP", "q5255_3tdNPK", "q5256_3appDaysNPK", "q5257_3tdUrea", 
                                 "q5258_3appDaysUrea", "q5259_3tdNPKS", "q5260_3appDaysNPKS", 
                                 "q5261_3tdMoP", "q5262_3appDaysMoP", "q5263_3tdSSP", "q5264_3appDaysSSP", 
                                 "q5265_3tdTSP", "q5266_3appDaysTSP", "q5267_3tdZnSO4", "q5268_3appDaysZnSO4", 
                                 "q5269_3tdGypsum", "q5270_3appDaysGypsum", "q5271_3tdBoron", 
                                 "q5272_3appDaysBoron", "q52721_3tdOtherFert", "q52722_3tdOtherFertAmt", 
                                 "q52723_3appDaysOtherFert", "q51071_gradeNPK", "q51211_gradeNPKS",  
                                 "q5112_priceDAP", "q5113_priceNPK", "q5114_priceUrea", "q5127_priceMoP", 
                                 "q5115_priceZnSO4", "q5116_priceGypsum", "q5117_priceBoron", 
                                 "q5126_priceNPKS", "q5128_priceSSP", "q5129_priceTSP", "q5273_fertInfoSource", 
                                 "q5274_fertInfoOther", "q5275_sourceDAP", "q5276_otherSourceDAP", 
                                 "q5277_sourceNPK", "q5278_otherSourceNPK", "q5279_sourceUrea", 
                                 "q5280_otherSourceUrea", "q5281_sourceZnSO4", "q5282_otherSourceZnSO4", 
                                 "q5283_sourceGypsum", "q5284_otherSourceGypsum", "q5285_sourceBoron", 
                                 "q5286_otherSourceBoron", "q5287_fertOnTime", "q5288_avgDelayWeeks", 
                                 "q5301_irrigAvail", "q5302_irrigSource", "q5303_irrigOtherSource", 
                                 "q5304_irrigGrthStage", "q5305_irrigTimes", "q5306_drySpellReason", 
                                 "q5307_drySpellOther", "q5401_irrigDecision", "q5402_irrigOtherDecsn", 
                                 "q5403_notIrrigReason", "q5404_notIrrigOther", "q5405_tubewellDepthFeet", 
                                 "q5406_pumpEnergy", "q5407_layFlatPipe", "q5501_droughtGS", 
                                 "q5502_droughtSeverity", "q5503_floodGS", "q5504_floodSeverity", 
                                 "q5505_weedSeverity", "q5506_insectSeverity", "q5507_insecticides", 
                                 "q5508_insecticidesName", "q5509_diseaseSeverity", "q5510_pesticides", 
                                 "q5511_pesticidesName", "q5512_lodgingPercent", "herbAppTimes",  
                                 "q5601_1herbName", "q5602_1herbAppDays", "q5603_2herbName", "q5604_2herbAppDays", 
                                 "q5605_3herbName", "q5606_3herbAppDays","manualWeedTimes" ,"q5607_1manualWeeding", 
                                 "q5608_2manualWeeding", "q5609_3manualWeeding",  
                                 "q5701_topTenWeeds", "q5702_top1stWeed", "q5703_top2ndWeed", 
                                 "q5704_top3rdWeed", "q5705_top4thWeed", "q5706_top5thWeed", 
                                 "q601_harvestDate", "cropDurationDays", "q602_harvestMethod", 
                                 "q603_residueBurnt", "q604_threshing", "q605_totalGrainYieldQUINTAL", 
                                 "q606_largestPlotYieldQUNITAL",  
                                 "q607_farmGatePrice", "q608_fiveYearGProd", "q701_hhMember", 
                                 "q702_hhMemAg", "q703_marketSaleShare", "q704_agIncomeShare", 
                                 "q705_cropShareAg", "q706_cropSP", "q707_cropAvgSP", "q708_marketDistance", 
                                 "q801_newMgmtUse", "instanceID", "largestPlotGPS.Latitude", "largestPlotGPS.Longitude"))


#Splitting the dates
mf <- mf %>% separate(q406_prevCropHarvest, into = c("q406_2Day","q406_1Month","q4066_3Year"), sep="-")
mf <- mf %>% separate(q415_seedingSowingTransDate, into = c("q415_2Day","q415_1Month","q415_3Year"), sep="-")
mf <- mf %>% separate(q601_harvestDate, into = c("q601_2Day","q601_1Month","q601_3Year"), sep="-")

###Replacing blank cells with 0
mf <- mf %>% mutate(q5103_amtFYM = if_else(is.na(q5103_amtFYM), 0, q5103_amtFYM))
mf <- mf %>% mutate(q5202_basalDAP = if_else(is.na(q5202_basalDAP), 0, q5202_basalDAP))
mf <- mf %>% mutate(q5203_basalNPK = if_else(is.na(q5203_basalNPK), 0, q5203_basalNPK))
mf <- mf %>% mutate(q5204_basalUrea = if_else(is.na(q5204_basalUrea), 0, q5204_basalUrea))
mf$q5205_basalNPKS <- as.numeric(mf$q5205_basalNPKS)
mf <- mf %>% mutate(q5205_basalNPKS = if_else(is.na(q5205_basalNPKS), 0, q5205_basalNPKS))
mf <- mf %>% mutate(q5206_basalMoP = if_else(is.na(q5206_basalMoP), 0, q5206_basalMoP))
mf$q5207_basalSSP <- as.numeric(mf$q5207_basalSSP)
mf <- mf %>% mutate(q5207_basalSSP = if_else(is.na(q5207_basalSSP), 0, q5207_basalSSP))
mf$q5208_basalTSP <- as.numeric(mf$q5208_basalTSP)
mf <- mf %>% mutate(q5208_basalTSP = if_else(is.na(q5208_basalTSP), 0, q5208_basalTSP))
mf <- mf %>% mutate(q5209_basalZnSO4 = if_else(is.na(q5209_basalZnSO4), 0, q5209_basalZnSO4))
mf$q5210_basalGypsum <- as.numeric(mf$q5210_basalGypsum)
mf <- mf %>% mutate(q5210_basalGypsum = if_else(is.na(q5210_basalGypsum), 0, q5210_basalGypsum))
mf <- mf %>% mutate(q5211_basalBoron = if_else(is.na(q5211_basalBoron), 0, q5211_basalBoron))
mf$q52112_otherBasalFertAmt <- as.numeric(mf$q52112_otherBasalFertAmt)
mf <- mf %>% mutate(q52112_otherBasalFertAmt = if_else(is.na(q52112_otherBasalFertAmt), 0, q52112_otherBasalFertAmt))
mf <- mf %>% mutate(q5212_1tdDAP = if_else(is.na(q5212_1tdDAP), 0, q5212_1tdDAP))
mf$q5213_1appDaysDAP <- as.numeric(mf$q5213_1appDaysDAP)
mf <- mf %>% mutate(q5213_1appDaysDAP = if_else(is.na(q5213_1appDaysDAP), 0, q5213_1appDaysDAP))
mf$q5214_1tdNPK <- as.numeric(mf$q5214_1tdNPK)
mf <- mf %>% mutate(q5214_1tdNPK = if_else(is.na(q5214_1tdNPK), 0, q5214_1tdNPK))
mf$q5215_1appDaysNPK <- as.numeric(mf$q5215_1appDaysNPK)
mf <- mf %>% mutate(q5215_1appDaysNPK = if_else(is.na(q5215_1appDaysNPK), 0, q5215_1appDaysNPK))
mf <- mf %>% mutate(q5216_1tdUrea = if_else(is.na(q5216_1tdUrea), 0, q5216_1tdUrea))
mf$q5217_1appDaysUrea <- as.numeric(mf$q5217_1appDaysUrea)
mf <- mf %>% mutate(q5217_1appDaysUrea = if_else(is.na(q5217_1appDaysUrea), 0, q5217_1appDaysUrea))
mf$q5218_1tdNPKS <- as.numeric(mf$q5218_1tdNPKS)
mf <- mf %>% mutate(q5218_1tdNPKS = if_else(is.na(q5218_1tdNPKS), 0, q5218_1tdNPKS))
mf$q5219_1appDaysNPKS <- as.numeric(mf$q5219_1appDaysNPKS)
mf <- mf %>% mutate(q5219_1appDaysNPKS = if_else(is.na(q5219_1appDaysNPKS), 0, q5219_1appDaysNPKS))
mf <- mf %>% mutate(q5220_1tdMoP = if_else(is.na(q5220_1tdMoP), 0, q5220_1tdMoP))
mf$q5221_1appDaysMoP <- as.numeric(mf$q5221_1appDaysMoP)
mf <- mf %>% mutate(q5221_1appDaysMoP = if_else(is.na(q5221_1appDaysMoP), 0, q5221_1appDaysMoP))
mf <- mf %>% mutate(q5223_1tdSSP = if_else(is.na(q5223_1tdSSP), 0, q5223_1tdSSP))
mf$q5224_1appDaysSSP <- as.numeric(mf$q5224_1appDaysSSP)
mf <- mf %>% mutate(q5224_1appDaysSSP = if_else(is.na(q5224_1appDaysSSP), 0, q5224_1appDaysSSP))
mf$q5225_1tdTSP <- as.numeric(mf$q5225_1tdTSP)
mf <- mf %>% mutate(q5225_1tdTSP = if_else(is.na(q5225_1tdTSP), 0, q5225_1tdTSP))
mf$q5226_1appDaysTSP <-as.numeric(mf$q5226_1appDaysTSP)
mf <- mf %>% mutate(q5226_1appDaysTSP = if_else(is.na(q5226_1appDaysTSP), 0, q5226_1appDaysTSP))
mf <- mf %>% mutate(q5227_1tdZnSO4 = if_else(is.na(q5227_1tdZnSO4), 0, q5227_1tdZnSO4))
mf$q5228_1appDaysZnSO4 <-as.numeric(mf$q5228_1appDaysZnSO4)
mf <- mf %>% mutate(q5228_1appDaysZnSO4 = if_else(is.na(q5228_1appDaysZnSO4), 0, q5228_1appDaysZnSO4))
mf$q5228_1appDaysZnSO4 <-as.numeric(mf$q5228_1appDaysZnSO4)
mf$q5230_1appDaysGypsum <- as.numeric(mf$q5230_1appDaysGypsum)
mf <- mf %>% mutate(q5230_1appDaysGypsum = if_else(is.na(q5230_1appDaysGypsum), 0, q5230_1appDaysGypsum))
mf <- mf %>% mutate(q5231_1tdBoron = if_else(is.na(q5231_1tdBoron), 0, q5231_1tdBoron))
mf$q5229_1tdGypsum <-as.numeric(mf$q5229_1tdGypsum)
mf <- mf %>% mutate(q5229_1tdGypsum = if_else(is.na(q5229_1tdGypsum), 0, q5229_1tdGypsum))
mf$q5232_1appDaysBoron <-as.numeric(mf$q5232_1appDaysBoron)
mf <- mf %>% mutate(q5232_1appDaysBoron = if_else(is.na(q5232_1appDaysBoron), 0, q5232_1appDaysBoron))
mf$q52322_1tdOtherFertAmt <-as.numeric(mf$q52322_1tdOtherFertAmt)
mf <- mf %>% mutate(q52322_1tdOtherFertAmt = if_else(is.na(q52322_1tdOtherFertAmt), 0, q52322_1tdOtherFertAmt))
mf$q52323_1appDaysOtherFert <-as.numeric(mf$q52323_1appDaysOtherFert)
mf <- mf %>% mutate(q52323_1appDaysOtherFert = if_else(is.na(q52323_1appDaysOtherFert), 0, q52323_1appDaysOtherFert))
mf <- mf %>% mutate(q5233_2tdDAP = if_else(is.na(q5233_2tdDAP), 0, q5233_2tdDAP))
mf$q5234_2appDaysDAP <-as.numeric(mf$q5234_2appDaysDAP)
mf <- mf %>% mutate(q5234_2appDaysDAP = if_else(is.na(q5234_2appDaysDAP), 0, q5234_2appDaysDAP))
mf$q5235_2tdNPK<- as.numeric(mf$q5235_2tdNPK)
mf <- mf %>% mutate(q5235_2tdNPK = if_else(is.na(q5235_2tdNPK), 0, q5235_2tdNPK))
mf$q5236_2appDaysNPK <-as.numeric(mf$q5236_2appDaysNPK)
mf <- mf %>% mutate(q5236_2appDaysNPK = if_else(is.na(q5236_2appDaysNPK), 0, q5236_2appDaysNPK))
mf <- mf %>% mutate(q5237_2tdUrea = if_else(is.na(q5237_2tdUrea), 0, q5237_2tdUrea))
mf$q5238_2appDaysUrea <-as.numeric(mf$q5238_2appDaysUrea)
mf <- mf %>% mutate(q5238_2appDaysUrea = if_else(is.na(q5238_2appDaysUrea), 0, q5238_2appDaysUrea))
mf$q5239_2tdNPKS <-as.numeric(mf$q5239_2tdNPKS)
mf <- mf %>% mutate(q5239_2tdNPKS = if_else(is.na(q5239_2tdNPKS), 0, q5239_2tdNPKS))
mf$q5240_2appDaysNPKS <-as.numeric(mf$q5240_2appDaysNPKS)
mf <- mf %>% mutate(q5240_2appDaysNPKS = if_else(is.na(q5240_2appDaysNPKS), 0, q5240_2appDaysNPKS))
mf <- mf %>% mutate(q5241_2tdMoP = if_else(is.na(q5241_2tdMoP), 0, q5241_2tdMoP))
mf$q5242_2appDaysMoP <-as.numeric(mf$q5242_2appDaysMoP)
mf <- mf %>% mutate(q5242_2appDaysMoP = if_else(is.na(q5242_2appDaysMoP), 0, q5242_2appDaysMoP))
mf$q5243_2tdSSP <-as.numeric(mf$q5243_2tdSSP)
mf <- mf %>% mutate(q5243_2tdSSP = if_else(is.na(q5243_2tdSSP), 0, q5243_2tdSSP))
mf$q5244_2appDaysSSP <-as.numeric(mf$q5244_2appDaysSSP)
mf <- mf %>% mutate(q5244_2appDaysSSP = if_else(is.na(q5244_2appDaysSSP), 0, q5244_2appDaysSSP))
mf$q5245_2tdTSP <-as.numeric(mf$q5245_2tdTSP)
mf <- mf %>% mutate(q5245_2tdTSP = if_else(is.na(q5245_2tdTSP), 0, q5245_2tdTSP))
mf$q5246_2appDaysTSP <-as.numeric(mf$q5246_2appDaysTSP)
mf <- mf %>% mutate(q5246_2appDaysTSP = if_else(is.na(q5246_2appDaysTSP), 0, q5246_2appDaysTSP))
mf <- mf %>% mutate(q5247_2tdZnSO4 = if_else(is.na(q5247_2tdZnSO4), 0, q5247_2tdZnSO4))
mf$q5248_2appDaysZnSO4 <-as.numeric(mf$q5248_2appDaysZnSO4)
mf <- mf %>% mutate(q5248_2appDaysZnSO4 = if_else(is.na(q5248_2appDaysZnSO4), 0, q5248_2appDaysZnSO4))
mf$q5249_2tdGypsum <- as.numeric(mf$q5249_2tdGypsum)
mf <- mf %>% mutate(q5249_2tdGypsum = if_else(is.na(q5249_2tdGypsum), 0, q5249_2tdGypsum))
mf$q5250_2appDaysGypsum <-as.numeric(mf$q5250_2appDaysGypsum)
mf <- mf %>% mutate(q5250_2appDaysGypsum = if_else(is.na(q5250_2appDaysGypsum), 0, q5250_2appDaysGypsum))
mf <- mf %>% mutate(q5251_2tdBoron = if_else(is.na(q5251_2tdBoron), 0, q5251_2tdBoron))
mf$q5252_2appDaysBoron <-as.numeric(mf$q5252_2appDaysBoron)
mf <- mf %>% mutate(q5252_2appDaysBoron = if_else(is.na(q5252_2appDaysBoron), 0, q5252_2appDaysBoron))
mf$q52522_2tdOtherFertAmt <-as.numeric(mf$q52522_2tdOtherFertAmt)
mf <- mf %>% mutate(q52522_2tdOtherFertAmt = if_else(is.na(q52522_2tdOtherFertAmt), 0, q52522_2tdOtherFertAmt))
mf$q52523_2appDaysOtherFert <- as.numeric(mf$q52523_2appDaysOtherFert)
mf <- mf %>% mutate(q52523_2appDaysOtherFert = if_else(is.na(q52523_2appDaysOtherFert), 0, q52523_2appDaysOtherFert))
mf <- mf %>% mutate(q5253_3tdDAP = if_else(is.na(q5253_3tdDAP), 0, q5253_3tdDAP))
mf$q5254_3appDaysDAP <-as.numeric(mf$q5254_3appDaysDAP)
mf <- mf %>% mutate(q5254_3appDaysDAP = if_else(is.na(q5254_3appDaysDAP), 0, q5254_3appDaysDAP))
mf$q5255_3tdNPK <-as.numeric(mf$q5255_3tdNPK)
mf <- mf %>% mutate(q5255_3tdNPK = if_else(is.na(q5255_3tdNPK), 0, q5255_3tdNPK))
mf$q5256_3appDaysNPK <-as.numeric(mf$q5256_3appDaysNPK)
mf <- mf %>% mutate(q5256_3appDaysNPK = if_else(is.na(q5256_3appDaysNPK), 0, q5256_3appDaysNPK))
mf <- mf %>% mutate(q5257_3tdUrea = if_else(is.na(q5257_3tdUrea), 0, q5257_3tdUrea))
mf$q5258_3appDaysUrea <-as.numeric(mf$q5258_3appDaysUrea)
mf <- mf %>% mutate(q5258_3appDaysUrea = if_else(is.na(q5258_3appDaysUrea), 0, q5258_3appDaysUrea))
mf$q5259_3tdNPKS <-as.numeric(mf$q5259_3tdNPKS)
mf <- mf %>% mutate(q5259_3tdNPKS = if_else(is.na(q5259_3tdNPKS), 0, q5259_3tdNPKS))
mf$q5260_3appDaysNPKS <-as.numeric(mf$q5260_3appDaysNPKS)
mf <- mf %>% mutate(q5260_3appDaysNPKS = if_else(is.na(q5260_3appDaysNPKS), 0, q5260_3appDaysNPKS))
mf$q5261_3tdMoP <- as.numeric(mf$q5261_3tdMoP)
mf <- mf %>% mutate(q5261_3tdMoP = if_else(is.na(q5261_3tdMoP), 0, q5261_3tdMoP))
mf$q5262_3appDaysMoP <-as.numeric(mf$q5262_3appDaysMoP)
mf <- mf %>% mutate(q5262_3appDaysMoP = if_else(is.na(q5262_3appDaysMoP), 0, q5262_3appDaysMoP))
mf$q5263_3tdSSP <-as.numeric(mf$q5263_3tdSSP)
mf <- mf %>% mutate(q5263_3tdSSP = if_else(is.na(q5263_3tdSSP), 0, q5263_3tdSSP))
mf$q5264_3appDaysSSP <-as.numeric(mf$q5264_3appDaysSSP)
mf <- mf %>% mutate(q5264_3appDaysSSP = if_else(is.na(q5264_3appDaysSSP), 0, q5264_3appDaysSSP))
mf$q5265_3tdTSP <-as.numeric(mf$q5265_3tdTSP)
mf <- mf %>% mutate(q5265_3tdTSP = if_else(is.na(q5265_3tdTSP), 0, q5265_3tdTSP))
mf$q5266_3appDaysTSP <-as.numeric(mf$q5266_3appDaysTSP)
mf <- mf %>% mutate(q5266_3appDaysTSP = if_else(is.na(q5266_3appDaysTSP), 0, q5266_3appDaysTSP))
mf$q5267_3tdZnSO4 <- as.numeric(mf$q5267_3tdZnSO4)
mf <- mf %>% mutate(q5267_3tdZnSO4 = if_else(is.na(q5267_3tdZnSO4), 0, q5267_3tdZnSO4))
mf$q5268_3appDaysZnSO4 <-as.numeric(mf$q5268_3appDaysZnSO4)
mf <- mf %>% mutate(q5268_3appDaysZnSO4 = if_else(is.na(q5268_3appDaysZnSO4), 0, q5268_3appDaysZnSO4))
mf$q5269_3tdGypsum <-as.numeric(mf$q5269_3tdGypsum)
mf <- mf %>% mutate(q5269_3tdGypsum = if_else(is.na(q5269_3tdGypsum), 0, q5269_3tdGypsum))
mf$q5270_3appDaysGypsum <-as.numeric(mf$q5270_3appDaysGypsum)
mf <- mf %>% mutate(q5270_3appDaysGypsum = if_else(is.na(q5270_3appDaysGypsum), 0, q5270_3appDaysGypsum))
mf <- mf %>% mutate(q5271_3tdBoron = if_else(is.na(q5271_3tdBoron), 0, q5271_3tdBoron))
mf$q5272_3appDaysBoron <-as.numeric(mf$q5272_3appDaysBoron)
mf <- mf %>% mutate(q5272_3appDaysBoron = if_else(is.na(q5272_3appDaysBoron), 0, q5272_3appDaysBoron))
mf$q52722_3tdOtherFertAmt <-as.numeric(mf$q52722_3tdOtherFertAmt)
mf <- mf %>% mutate(q52722_3tdOtherFertAmt = if_else(is.na(q52722_3tdOtherFertAmt), 0, q52722_3tdOtherFertAmt))
mf$q52723_3appDaysOtherFert <- as.numeric(mf$q52723_3appDaysOtherFert)
mf <- mf %>% mutate(q52723_3appDaysOtherFert = if_else(is.na(q52723_3appDaysOtherFert), 0, q52723_3appDaysOtherFert))
mf$q5288_avgDelayWeeks <-as.numeric(mf$q5288_avgDelayWeeks)
mf <- mf %>% mutate(q5288_avgDelayWeeks = if_else(is.na(q5288_avgDelayWeeks), 0, q5288_avgDelayWeeks))
mf$q5305_irrigTimes <-as.numeric(mf$q5305_irrigTimes)
mf <- mf %>% mutate(q5305_irrigTimes = if_else(is.na(q5305_irrigTimes), 0, q5305_irrigTimes))

#Converting crop largest area in acres
mf$q306_cropLarestAreaAcre <- mf$q305_cropLargestArea/mf$q302_acreConv

#Calculating yield t/ha
mf$Yieldkgha <- ((mf$q606_largestPlotYieldQUNITAL/mf$q306_cropLarestAreaAcre)*2.5*100)

#Wheat yield distribution through histogram 
hist(mf$Yieldkgha, 
     xlab="Wheat yield (kg/ha)",
     ylab="Frequency")

summary(mf$Yieldkgha)

#Wheat yield by states
tapply(mf$Yieldkgha, mf$q102_state, summary)

#Wheat yield by districts
tapply(mf$Yieldkgha, mf$q103_district, summary)

#Wheat sowing date in Julian days
mf$q415_2Day <- as.numeric(mf$q415_2Day)
mf$q415_1Month <- as.numeric(mf$q415_1Month)
mf$Sowingfactor <- mf$q415_1Month
mf$Sowingfactor <- ifelse(mf$Sowingfactor==1, 365,
                          ifelse(mf$q415_1Month==10, 273,
                                 ifelse(mf$q415_1Month==11, 304,
                                        ifelse(mf$q415_1Month==12, 334,
                                               mf$Sowingfactor))))
mf$Sowingdate <- mf$Sowingfactor+mf$q415_2Day

#Distribution of sowing time
hist(mf$Sowingdate, 
     xlab="Sowing date (DOY)",
     ylab="Frequency")

#Sowing times summary by states and districts
tapply(mf$Sowingdate, mf$q102_state, summary)
tapply(mf$Sowingdate, mf$q103_district, summary)

#Check sowing date and yield relations 
cor(mf$Sowingdate, mf$Yieldkgha)

##CI Intervals (Sowing time vs wheat yeild)
summary(mf$Sowingdate)
sd(mf$Sowingdate)

#Check sowing date and yield relations 
plot(mf$Sowingdate, mf$Yieldkgha)  

#fit the regression line
abline(lm(mf$Yieldkgha~mf$Sowingdate), col="red")

#Regress yield and sowing time
S1 <- lm(Yieldkgha~Sowingdate, data=mf)
summary(S1)

##Wheat sowing time distributions - by states
ggplot(mf, aes(x=q102_state, y=Sowingdate, fill=q102_state)) + 
  geom_violin(trim=T)+
  geom_boxplot(width=0.1)+
  geom_line(y =1)  +
  #coord_flip()+
  labs(title="Wheat sowing time by states", x="States", y = "Wheat yield (kg/ha)")+
  stat_summary(fun.y = "mean", geom = "text", label=".", size= 10, color= "blue")+
  scale_y_continuous("Wheat sowing time (DOY)",
                     limits=c(290,390),
                     breaks=seq(290, 390, 10))

##Wheat sowing time distributions - by districts
ggplot(mf, aes(x = reorder(q103_district, Sowingdate, FUN = mean), y = Sowingdate), fill=q103_district)+
  #geom_violin(trim=T)+
  geom_boxplot(width=0.7)+
  geom_line(y =1)  +
  labs(title="Wheat sowing time by districts", x="Districts", y = "Wheat yield (kg/ha)")+
  stat_summary(fun.y = "mean", geom = "text", label=".", size= 10, color= "blue")+
  theme(axis.text.x=element_text(color = "black", size=11, angle=90, vjust=.8, hjust=0.8))+
  scale_y_continuous("Wheat sowing time (DOY)",
                     limits=c(290,390),
                     breaks=seq(290, 390, 10))

##Fertlizer applications
#Total urea
mf$Urea_tot_lp_ha <- (mf$q5204_basalUrea+
                        mf$q5216_1tdUrea+
                        mf$q5237_2tdUrea+
                        mf$q5257_3tdUrea)
mf$Urea_tot_lp_ha <- (mf$Urea_tot_lp_ha/mf$q306_cropLarestAreaAcre)*2.5
hist(mf$Urea_tot_lp_ha)
summary(mf$Urea_tot_lp_ha)


#Total DAP 
mf$DAP_tot_lp_ha <- (mf$q5202_basalDAP+
                       mf$q5212_1tdDAP+
                       mf$q5233_2tdDAP+
                       mf$q5253_3tdDAP)
mf$DAP_tot_lp_ha <- (mf$DAP_tot_lp_ha/mf$q306_cropLarestAreaAcre)*2.5
hist(mf$DAP_tot_lp_ha)
summary(mf$DAP_tot_lp_ha)

#Total Potash
mf$Potash_tot_lp_ha <- (mf$q5206_basalMoP+
                          mf$q5220_1tdMoP+
                          mf$q5241_2tdMoP+
                          mf$q5261_3tdMoP)
mf$Potash_tot_lp_ha <- (mf$Potash_tot_lp_ha/mf$q306_cropLarestAreaAcre)*2.5
hist(mf$Potash_tot_lp_ha)
summary(mf$Potash_tot_lp_ha)
#Total SSP 
mf$SSP_tot_lp_ha <- (mf$q5207_basalSSP+
                       mf$q5223_1tdSSP+
                       mf$q5243_2tdSSP+
                       mf$q5263_3tdSSP)
mf$SSP_tot_lp_ha <- (mf$SSP_tot_lp_ha/mf$q306_cropLarestAreaAcre)*2.5
hist(mf$SSP_tot_lp_ha)
summary(mf$SSP_tot_lp_ha)

# Total NPK
mf$NPK_tot_lp_ha <- (mf$q5203_basalNPK+
                       mf$q5214_1tdNPK+
                       mf$q5235_2tdNPK+
                       mf$q5255_3tdNPK)
mf$NPK_tot_lp_ha <- (mf$NPK_tot_lp_ha/mf$q306_cropLarestAreaAcre)*2.5
hist(mf$NPK_tot_lp_ha)
summary(mf$NPK_tot_lp_ha)

# Total NPKS
mf$NPKS_tot_lp_ha <- (mf$q5205_basalNPKS+
                        mf$q5218_1tdNPKS+
                        mf$q5239_2tdNPKS+
                        mf$q5259_3tdNPKS)
mf$NPKS_tot_lp_ha <- (mf$NPKS_tot_lp_ha/mf$q306_cropLarestAreaAcre)*2.5
hist(mf$NPKS_tot_lp_ha)
summary(mf$NPKS_tot_lp_ha)

# Total TSP
mf$TSP_tot_lp_ha <- (mf$q5208_basalTSP+
                       mf$q5225_1tdTSP+
                       mf$q5245_2tdTSP+
                       mf$q5265_3tdTSP)
mf$TSP_tot_lp_ha <- (mf$TSP_tot_lp_ha/mf$q306_cropLarestAreaAcre)*2.5
hist(mf$TSP_tot_lp_ha)
summary(mf$TSP_tot_lp_ha)

# Total ZnSO4
mf$ZnSO4_tot_lp_ha <- (mf$q5209_basalZnSO4+
                         mf$q5227_1tdZnSO4+
                         mf$q5247_2tdZnSO4+
                         mf$q5267_3tdZnSO4)
mf$ZnSO4_tot_lp_ha <- (mf$ZnSO4_tot_lp_ha/mf$q306_cropLarestAreaAcre)*2.5
hist(mf$ZnSO4_tot_lp_ha)
summary(mf$ZnSO4_tot_lp_ha)

# Total Gypsum
mf$Gypsum_tot_lp_ha <- (mf$q5210_basalGypsum+
                          mf$q5229_1tdGypsum+
                          mf$q5249_2tdGypsum+
                          mf$q5269_3tdGypsum)
mf$Gypsum_tot_lp_ha <- (mf$Gypsum_tot_lp_ha/mf$q306_cropLarestAreaAcre)*2.5
hist(mf$Gypsum_tot_lp_ha)
summary(mf$Gypsum_tot_lp_ha)

# Total Boron
mf$Boron_tot_lp_ha <- (mf$q5211_basalBoron+
                         mf$q5231_1tdBoron+
                         mf$q5251_2tdBoron+
                         mf$q5271_3tdBoron)
mf$Boron_tot_lp_ha <- (mf$Boron_tot_lp_ha/mf$q306_cropLarestAreaAcre)*2.5
hist(mf$Boron_tot_lp_ha)
summary(mf$Boron_tot_lp_ha)


# Total other fertilizer
mf$q52112_otherBasalFertAmt <- as.numeric(mf$q52112_otherBasalFertAmt)
mf$OtherFert_tot_lp_ha <- (mf$q52112_otherBasalFertAmt+
                             mf$q52322_1tdOtherFertAmt+
                             mf$q52522_2tdOtherFertAmt+
                             mf$q52722_3tdOtherFertAmt)
mf$OtherFert_tot_lp_ha <- (mf$OtherFert_tot_lp_ha/mf$q306_cropLarestAreaAcre)*2.5

#TOtal N applied in largest plot (N (kg/ha))
mf$N_kgha <- mf$Urea_tot_lp_ha*0.46+mf$DAP_tot_lp_ha*.18+
  mf$NPK_tot_lp_ha*.12+mf$NPKS_tot_lp_ha*.20
hist(mf$N_kgha)
summary(mf$N_kgha)
sd(mf$N_kgha)

#Nitrogen application  by states
ddply(mf, .(q102_state), summarize, mean=mean(N_kgha), sd=sd(N_kgha))
#Nitrogen application rates by states and districts

##Nitrogen application by distrcits
ddply(mf, .(q103_district), summarize, mean=mean(N_kgha), sd=sd(N_kgha))

#P2O5 (kg/ha)
mf$P2O5_kgha =(mf$DAP_tot_lp_ha*.46+mf$NPK_tot_lp_ha*.32+
                 mf$NPKS_tot_lp_ha*.20+mf$TSP_tot_lp_ha*.48+
                 mf$SSP_tot_lp_ha*.16)
hist(mf$P2O5_kgha)
summary(mf$P2O5_kgha)
sd(mf$P2O5_kgha)

#Distribution of P2O5 application  by states and districts
ddply(mf, .(q102_state), summarize, mean=mean(P2O5_kgha), sd=sd(P2O5_kgha))
ddply(mf, .(q103_district), summarize, mean=mean(P2O5_kgha), sd=sd(P2O5_kgha))

#K2O (kg/ha)
mf$K2O_kgha <- mf$Potash_tot_lp_ha*.60 + mf$NPK_tot_lp_ha*.16
hist(mf$K2O_kgha)
summary(mf$K2O_kgha)
sd(mf$K2O_kgha)

#Distribution of K2O application  by states and districts
ddply(mf, .(q102_state), summarize, mean=mean(K2O_kgha), sd=sd(K2O_kgha))
ddply(mf, .(q103_district), summarize, mean=mean(K2O_kgha), sd=sd(K2O_kgha))

#ZnZo4 (kg/ha)
mf$ZnSo4_kgha <- mf$ZnSO4_tot_lp_ha 

#Boron (kg/ha)
mf$Boron_kgha <- mf$Boron_tot_lp_ha

#Gypsum (kg/ha)
mf$Gypsum_kgha <- mf$Gypsum_tot_lp_ha

#FYM (kg/ha)
mf$FYM_kgha <- (mf$q5103_amtFYM/mf$q306_cropLarestAreaAcre)*2.5
hist(mf$FYM_kgha)
summary(mf$FYM_kgha)
sd(mf$FYM_kgha)

#Seed rate
mf$Seedrate_kgha <- (mf$q420_cropSeedAmt/mf$q306_cropLarestAreaAcre)*2.5
hist(mf$Seedrate_kgha)
cor(mf$Seedrate_kgha, mf$Yieldkgha)
summary(mf$Seedrate_kgha)

##Seed rate summary by states, distrcits and crop establishment methods
ddply(mf, .(q102_state), summarize, mean=mean(Seedrate_kgha), sd=sd(Seedrate_kgha))
ddply(mf, .(q103_district), summarize, mean=mean(Seedrate_kgha), sd=sd(Seedrate_kgha))
ddply(mf, .(q413_CropEst), summarize, mean=mean(Seedrate_kgha), sd=sd(Seedrate_kgha))

##Change the seed rate by conventional and zero tillage 
mf$q413_CropEst_Seed <- mf$q413_CropEst
mf$q413_CropEst_Seed[mf$q413_CropEst_Seed == "Broadcasting"] <- "Conventional"
mf$q413_CropEst_Seed[mf$q413_CropEst_Seed == "LineSowingAfterTillage"] <- "Conventional"
mf$q413_CropEst_Seed[mf$q413_CropEst_Seed == "SurfaceSeeding"] <- "Conventional"
ddply(mf, .(q413_CropEst_Seed), summarize, mean=mean(Seedrate_kgha), sd=sd(Seedrate_kgha))

##Wheat yield and irrigation

Irrigation_no <- cut(mf$q5305_irrigTimes,  breaks = c(-1, 0, 1, 2, 8),
                     labels = c("No", "1", "2", ">=3"))

table(Irrigation_no)
#Yield vs irrigation times
cor (mf$Sowingdate, mf$q5305_irrigTimes)

#Summary statistics of the no of irrigation and yield
ddply(mf, .(Irrigation_no), summarize, mean=mean(Yieldkgha), sd=sd(Yieldkgha))

##Wheat yield by cropping systems 
table(mf$q404_prevCrop)
mf$q404_prevCrop[mf$q404_prevCrop == "RIce"] <- "Rice"
mf$q404_prevCrop[mf$q404_prevCrop == "Rice "] <- "Rice"
mf$q404_prevCrop[mf$q404_prevCrop == "Bajra"] <- "Others"
mf$q404_prevCrop[mf$q404_prevCrop == "Banana"] <- "Others"
mf$q404_prevCrop[mf$q404_prevCrop == "Greengram"] <- "Others"
mf$q404_prevCrop[mf$q404_prevCrop == "GreenManure"] <- "Others"
mf$q404_prevCrop[mf$q404_prevCrop == "Groundnut"] <- "Others"
mf$q404_prevCrop[mf$q404_prevCrop == "Jowar"] <- "Others"
mf$q404_prevCrop[mf$q404_prevCrop == "Mungbean"] <- "Others"
mf$q404_prevCrop[mf$q404_prevCrop == "OtherLentil"] <- "Others"
mf$q404_prevCrop[mf$q404_prevCrop == "Pulses"] <- "Others"
mf$q404_prevCrop[mf$q404_prevCrop == "Sugarcane"] <- "Others"
mf$q404_prevCrop[mf$q404_prevCrop == "Vegetables"] <- "Others"


ddply(mf, .(q404_prevCrop), summarize, mean=mean(Yieldkgha), sd=sd(Yieldkgha))

Croppingsytems
##Cropping systems 
irrso <- c(14, 107, 139, 54, 95)
labels <- c("Fallow-Wheat", "Maize-Wheat", "Rice-Wheat","Others-Wheat", "soyabean-Wheat")
pct <- round(irrso/sum(irrso)*100)
labels <- paste(labels, pct)
labels <- paste(labels,"%", sep="") 
pie(irrso,labels = labels, col=rainbow(length(labels)),
    main="Cropping systems (N=7648)")

##Wheat yield by drinage classes
table(mf$q402_drainClass)

ddply(mf, .(q402_drainClass), summarize, mean=mean(Yieldkgha), sd=sd(Yieldkgha))

##Rotavator tillage and yield 
mf$q412_RotUseYear <- as.numeric(mf$q412_RotUseYear)
mf <- mf %>% mutate(q412_RotUseYear = if_else(is.na(q412_RotUseYear), 0, q412_RotUseYear))

#Year wise rotavator tillage yield
mf$Rotavator_yrs <- cut(mf$q412_RotUseYear,  breaks = c(-1, 0, 1, 2, 10),
                        labels = c("OT", "RT_Y1", "RT_Y2", ">=RT_Y3"))

##Crop establishement and yield relationship 
table(mf$q413_CropEst)

ddply(mf, .(q413_CropEst), summarize, mean=mean(Yieldkgha), sd=sd(Yieldkgha))

#Two way cross-tabs
DCR <- xtabs(~ q413_CropEst +q103_district, mf)
DCR
##Name of the varieties 
table(mf$q410_varName)
str(mf$q410_varName)

##Wheat varieties
mf$variety <-mf$q410_varName
table(mf$variety)
mf$variety <-  ifelse(mf$variety=="Aditya" & mf$q410_varName=="Aditya", "Other",
                      ifelse(mf$variety=="Ankur" & mf$q410_varName=="Ankur", "Other",
                             ifelse(mf$variety=="Baaz" & mf$q410_varName=="Baaz", "Other",
                                    ifelse(mf$variety=="Bhrikuti" & mf$q410_varName=="Bhrikuti", "Other",
                                           ifelse(mf$variety=="BL 4341" & mf$q410_varName=="BL 4341", "Other",
                                                  ifelse(mf$variety=="BR26" & mf$q410_varName=="BR26", "Other",
                                                         ifelse(mf$variety=="DBW14" & mf$q410_varName=="DBW14", "Other",
                                                                ifelse(mf$variety=="Gautam" & mf$q410_varName=="Gautam", "Other",
                                                                       ifelse(mf$variety=="HD2781" & mf$q410_varName=="HD2781", "Other",
                                                                              ifelse(mf$variety=="HD3086" & mf$q410_varName=="HD3086", "Other",
                                                                                     ifelse(mf$variety=="K307" & mf$q410_varName=="K307", "Other",
                                                                                            ifelse(mf$variety=="KW412" & mf$q410_varName=="KW412", "Other",
                                                                                                   ifelse(mf$variety=="Loknath" & mf$q410_varName=="Loknath", "Other",
                                                                                                          ifelse(mf$variety=="Maheen" & mf$q410_varName=="Maheen", "Other",
                                                                                                                 ifelse(mf$variety=="Maina" & mf$q410_varName=="Maina", "Other",
                                                                                                                        ifelse(mf$variety=="NL2" & mf$q410_varName=="NL2", "NL1",
                                                                                                                               ifelse(mf$variety=="OtherAnshu" & mf$q410_varName=="OtherAnshu", "Other",
                                                                                                                                      ifelse(mf$variety=="OtherMicoGol" & mf$q410_varName=="OtherMicoGol", "Other",
                                                                                                                                             ifelse(mf$variety=="OtherPooja" & mf$q410_varName=="OtherPooja", "Other",
                                                                                                                                                    ifelse(mf$variety=="OtherUday raj21" & mf$q410_varName=="OtherUday raj21", "Other",
                                                                                                                                                           ifelse(mf$variety=="Otherunknown" & mf$q410_varName=="Otherunknown", "Other",
                                                                                                                                                                  ifelse(mf$variety=="Radha4" & mf$q410_varName=="Radha4", "Other",
                                                                                                                                                                         ifelse(mf$variety=="Suryamukhi" & mf$q410_varName=="Suryamukhi", "Other",
                                                                                                                                                                                ifelse(mf$variety=="Vijay" & mf$q410_varName=="Vijay", "Other",
                                                                                                                                                                                       ifelse(mf$variety=="WH1105" & mf$q410_varName=="WH1105", "Other",
                                                                                                                                                                                              ifelse(mf$variety=="WH711" & mf$q410_varName=="WH711", "Other",
                                                                                                                                                                                                     ifelse(mf$variety=="OtherUnknown" & mf$q410_varName=="OtherUnknown", "Other",
                                                                                                                                                                                                            ifelse(mf$variety=="SUPER172" & mf$q410_varName=="SUPER172", "Other",
                                                                                                                                                                                                                   ifelse(mf$variety=="ShriRam 303" & mf$q410_varName=="ShriRam 303", "Sriram303",
                                                                                                                                                                                                                          ifelse(mf$variety=="OtherLocal" & mf$q410_varName=="OtherLocal", "Other",
                                                                                                                                                                                                                                 ifelse(mf$variety=="OtherBansal" & mf$q410_varName=="OtherBansal", "Other",
                                                                                                                                                                                                                                        ifelse(mf$variety=="LOK1" & mf$q410_varName=="LOK1", "Lok1",
                                                                                                                                                                                                                                               ifelse(mf$variety=="DBW17" & mf$q410_varName=="DBW17", "Other",
                                                                                                                                                                                                                                                      ifelse(mf$variety=="HD2985" & mf$q410_varName=="HD2985", "Other",
                                                                                                                                                                                                                                                             ifelse(mf$variety=="HI1563" & mf$q410_varName=="HI1563", "Other",
                                                                                                                                                                                                                                                                    ifelse(mf$variety=="Kundan" & mf$q410_varName=="Kundan", "Other",
                                                                                                                                                                                                                                                                           ifelse(mf$variety=="Local" & mf$q410_varName=="Local", "Other",
                                                                                                                                                                                                                                                                                  ifelse(mf$variety=="PBW2076" & mf$q410_varName=="PBW2076", "Other",
                                                                                                                                                                                                                                                                                         ifelse(mf$variety=="PBW550" & mf$q410_varName=="PBW550", "Other",
                                                                                                                                                                                                                                                                                                ifelse(mf$variety=="RR21" & mf$q410_varName=="RR21", "Other",
                                                                                                                                                                                                                                                                                                       mf$variety))))))))))))))))))))))))))))))))))))))))






