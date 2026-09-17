## Dissertation analysis script
### Results through 15 July 2025
### Original exploratory R workflow for the UK Nature-based Solutions farmer survey.

# Resolve paths relative to this script so the analysis can run from GitHub.
this_file <- tryCatch({
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    normalizePath(sub("^--file=", "", file_arg[1]))
  } else if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    rstudioapi::getSourceEditorContext()$path
  } else {
    file.path(getwd(), "NbS_TUSSIE.R")
  }
}, error = function(e) file.path(getwd(), "NbS_TUSSIE.R"))

code_dir <- dirname(this_file)
repo_root <- dirname(code_dir)
data_dir <- file.path(repo_root, "data")
if (!dir.exists(data_dir)) {
  stop("Could not find the data/ folder. Open/run this script from the GitHub package root or code/ folder.")
}

# Open data set
library(readxl)
data_labels <- read_xlsx(file.path(data_dir, "Results_text.xlsx"))
data_v <- read_xlsx(file.path(data_dir, "Results_values.xlsx"))
data_sep <- read_xlsx(file.path(data_dir, "Results_sep.xlsx"))
data <- merge(data_labels, data_sep, by = "ResponseId")

#paquetes
library(dplyr)
library(ggplot2)
library(forcats)
library(FactoMineR)
library(factoextra)
library(gplots)
library(tidyverse)
library(stargazer)
library(fastlogitME)

data <- filter(data,!is.na(q1.1)) #elimine personas que no respondieron la encuesta, solo la abrieron
complete_surveyed <- filter(data,!is.na(q5.3)) #eliminé personas que no completaron la encuesta. no usar esta sino la de arriba, es solo para ver cuántas son

#arreglar variables
#pregunta q1.10: do you receive funding from any of the following... para de cada uno individual ver gráfica en qualtrics
data$funding <- ifelse(data$q1.10!="No funding", 1, 0)

data$funding_rep <- ifelse(!is.na(data$q1.10),1,0) ##respondio pregunta de funding
data$fundingPub <- ifelse(data$q1.10_1==1|data$q1.10_4==1|data$q1.10_5==1|data$q1.10_6==1|data$q1.10_7==1|data$q1.10_8==1|data$q1.10_9==1|data$q1.10_10==1|data$q1.10_11==1|data$q1.10_12==1|data$q1.10_13==1|data$q1.10_14==1|data$q1.10_15==1|data$q1.10_16,1,0) #=1 si recibe recursos publicos 
data$fundingPub <- ifelse(data$funding_rep==1&is.na(data$fundingPub),0,data$fundingPub)  
data$fundingPriv <- ifelse(data$q1.10_18==1|data$q1.10_19==1|data$q1.10_20==1,1,0) #recibe fondos privados
data$fundingPriv <- ifelse(data$funding_rep==1&is.na(data$fundingPriv),0,data$fundingPriv)
data$fundingOther <- ifelse(data$funding_rep==1&is.na(data$q1.10_17),0,data$q1.10_17) 
data$fundYes <- ifelse(data$fundingPub==1|data$fundingPriv==1|data$fundingOther==1,1,0) #recibe fondosde cualquier tipo


data$funding_cat <- ifelse(data$fundingOther==1,"Other","None")
data$funding_cat <- ifelse(data$fundingPub==1,"Public",data$funding_cat)
data$funding_cat <- ifelse(data$fundingPriv==1,"Private",data$funding_cat)
data$funding_cat <- ifelse(data$fundingPriv==1&data$fundingPub==1,"Both",data$funding_cat)



#extreme weather events
data$events <- with(data, ifelse(q2.1!="None", 1,0))


#hacer cada event por separado?

data$partner <- with (data, ifelse(q3.1=="No",0,1))
data$partnerlab <- with (data, ifelse(q3.1=="No","No","Yes"))


#merge questions q3.2.1 and 3.2.2 en las categorias que aplica
data$collabImportant <- with(data, ifelse(!is.na(data$q3.2.1_1), data$q3.2.1_1, data$q3.2.2_1)) #Collaborating with other farmers and working at the landscape scale is important
data$collabProd <- with(data, ifelse(!is.na(data$q3.2.1_2), data$q3.2.1_2, data$q3.2.2_2)) #Collaborating in projects with other farmers increases my farm’s productivity

#Do you participate in NbS
data$NbSyes <- with(data, ifelse(q4.1!="No",1,0))
data$NbS_rep <- ifelse(!is.na(data$q4.1),1,0) ##respondio pregunta de NbS
data$NbSindiv <- ifelse(data$q4.1_1==1,1,0) #=1 si hace NbS individualmente
data$NbSindiv <- ifelse(data$NbS_rep==1&is.na(data$NbSindiv),0,data$NbSindiv)  
data$NbScollab <- ifelse(data$q4.1_2==1,1,0) #=1 si hace NbS collaboration with other farmers
data$NbScollab <- ifelse(data$NbS_rep==1&is.na(data$NbScollab),0,data$NbScollab)  

#Do any of your neighbours implement NbS
#contestar I don't know se tomo como NA. NA=I don't know
data$neigh<- with(data, ifelse(is.na(q4.5),"I don't know",q4.5))
data$neighNbS<- with(data, ifelse(q4.5=="Yes",1,0)) 
cor.test(data$NbSyes, data$neighNbS)

#Do you agree with any of the following
data$q4.6_1 <- with(data, ifelse(!is.na(q4.6.1_1),q4.6.1_1,q4.6.2_1)) #NbS have higher costs than benefits
data$q4.6_2 <- with(data, ifelse(!is.na(q4.6.1_2),q4.6.1_2,q4.6.2_2)) #help protect my farm against extreme weather events
data$q4.6_3 <- with(data, ifelse(!is.na(q4.6.1_3),q4.6.1_3,q4.6.2_3)) # Nature-based solutions can improve crop yield and farm productivity
data$q4.6_4 <- with(data, ifelse(!is.na(q4.6.1_4),q4.6.1_4,q4.6.2_4)) # The effects of Nature-based Solutions can be seen in the short term
data$q4.6_5 <- with(data, ifelse(!is.na(q4.6.1_5),q4.6.1_5,q4.6.2_5)) # Implementing Nature-based Solutions requires sacrificing some of my productive land
data$q4.6_7 <- with(data, ifelse(!is.na(q4.6.1_7),q4.6.1_7,q4.6.2_6)) # Nature-based Solutions are most effective when implemented in collaboration with other farmers at the landscape level


#Descriptive analysis 
##después agregar error_bar a todas las geom_bars

data$NbS<- with(data, ifelse(NbScollab==1,"Implements NbS in collaboration",0))
data$NbS<- with(data, ifelse(NbSindiv==1,"Implements NbS individually",data$NbS))
data$NbS <- with(data, ifelse(NbScollab==1&NbSindiv==1, "Both",data$NbS))
data$NbS <- with(data, ifelse(NbSyes==0,"None",data$NbS))

data2<- filter(data,!is.na(NbSyes))
data3<- filter(data,!is.na(q2.2))

ggplot(data2, aes(x=factor(events), y=NbSyes, fill=factor(events)))+
  geom_bar(stat="summary",fun="mean", position="dodge")+
  labs(x="Has experienced weather events on farm",y="Implements NbS")+
  stat_summary(fun = mean,geom = "text", aes(label = scales::percent(..y.., accuracy = 1)), vjust = -0.5,position = position_dodge(width = 0.9))

ggplot(data2, aes(x=factor(NbS), y=events, fill=factor(NbS)))+
  geom_bar(stat="summary",fun="mean", position="dodge")+
  labs(y="Has experiences weather events on farm",x="Implements NbS")+
  stat_summary(fun = mean,geom = "text", aes(label = scales::percent(..y.., accuracy = 1)), vjust = -0.5,position = position_dodge(width = 0.9))

ggplot(data3, aes(x=factor(q2.2), y=NbSyes, fill=factor(q2.2)))+ ##Resultado mas interesante
  geom_bar(stat="summary",fun="mean", position="dodge")+
  labs(x="Has experienced weather events on farm",y="Implements NbS")+
  stat_summary(fun = mean,geom = "text", aes(label = scales::percent(..y.., accuracy = 1)), vjust = -0.5,position = position_dodge(width = 0.9))

#Agregar never a how frequently y plottear con eso
data$EvFreq <- with(data, ifelse(events==0,"Never",q2.2))
data2$EvFreq <- with(data2, ifelse(events==0,"Never",q2.2))
  
ggplot(data2, aes(x=factor(EvFreq, levels = c("Never","Once in the last five years","Once every 2-3 years","Once a year","More than once a year")), y=NbSyes, fill=factor(EvFreq)))+ ##Resultado mas interesante
  geom_bar(stat="summary",fun="mean", position="dodge")+
  labs(x="Has experienced weather events on farm",y="Implements NbS",fill="")+
  stat_summary(fun = mean,geom = "text", aes(label = scales::percent(..y.., accuracy = 1)), vjust = -0.5,position = position_dodge(width = 0.9))



ggplot(data3, aes(x=factor(q2.3), y=NbSyes, fill=factor(q2.3)))+
  geom_bar(stat="summary",fun="mean", position="dodge")+
  labs(x="Impacts of weather events on farm",y="Implements NbS")+
  stat_summary(fun = mean,geom = "text", aes(label = scales::percent(..y.., accuracy = 1)), vjust = -0.5,position = position_dodge(width = 0.9))


data$CCattr<- with(data, ifelse(q2.4=="Not at all caused by climate change,Partly caused by climate change,Mainly caused by climate change",NA,q2.4))
data$CCattr<- with(data, ifelse(q2.4=="Partly caused by climate change,Mainly caused by climate change","Partly",CCattr))
data$CCattr<- with(data, ifelse(q2.4=="Partly caused by climate change","Partly",CCattr))
data$CCattr<- with(data, ifelse(q2.4=="Mainly caused by climate change","Mainly",CCattr))
data$CCattr<- with(data, ifelse(q2.4=="Not at all caused by climate change","Not at all",CCattr))

data4 <- filter(data,!is.na(data$CCattr))

ggplot(data4, aes(x=factor(CCattr,levels=c("Not at all","Partly","Mainly")), y=NbSyes, fill=factor(CCattr,levels=c("Not at all","Partly","Mainly"))))+ ##Resultado mas interesante
  geom_bar(stat="summary",fun="mean", position="dodge")+
  labs(x="Weather events attribution to Climate change",y="Implements NbS", fill="CC attribution")+
  stat_summary(fun = mean,geom = "text", aes(label = scales::percent(..y.., accuracy = 1)), vjust = -0.5,position = position_dodge(width = 0.9))



ggplot(data4, aes(x=factor(CCattr,levels=c("Not at all","Partly","Mainly")), 
                  y=NbSyes, fill=factor(events)))+ ##Nada interesante
  geom_bar(stat="summary",fun="mean", position="dodge")+
  labs(x="Weather events attribution to Climate change",y="Implements NbS")+
  stat_summary(fun = mean,geom = "text", aes(label = scales::percent(..y.., accuracy = 1)), vjust = -0.5,position = position_dodge(width = 0.9))







#correspondance analysis
table1<- table(data$EvFreq,data$CCattr)
chi1<-chisq.test(table1)
resCA1<- CA(data)
EV1<-get_eigenvalue(resCA1)
cabiplot1<- fviz_ca_biplot(resCA1,map="rowprincipal", col.col="cadetblue", col.row="coral3",arrows = c( TRUE, TRUE),repel=TRUE) +
  labs(title="Relationship between climate change attribution and experiences with weather events")

table2<-table(data$NbS, data$CCattr)
resCA2<- CA(table2)
EV2<-get_eigenvalue(resCA2)
cabiplot2<- fviz_ca_biplot(resCA2,map="rowprincipal", col.col="cadetblue", col.row="coral3",arrows = c( TRUE, TRUE),repel=TRUE)+
  labs(title="Relationship between climate change attribution and implementation of NbS")

table3<-table(data$NbS, data$q4.6_2)
resCA3<- CA(table3)
EV3<-get_eigenvalue(resCA3)
cabiplot3<- fviz_ca_biplot(resCA3,map="rowprincipal", col.col="cadetblue", col.row="coral3",arrows = c( TRUE, TRUE),repel=TRUE)+
  labs(title="Relationship between implementation of NbS and they protect farm against extreme weather events")


# governance
data5<-filter(data,!is.na(q5.1_1))
gov1<-ggplot(data5, aes(x=factor(q5.1_1), y=NbSyes))+
  geom_bar(stat="summary",fun="mean", position="dodge",color="darkslategray3")+
  labs(x="Government promotes NbS",y="Implements NbS")



#risk perception
(risk1<- ggplot(data, aes(x=q2.5_1,y=NbSyes))+
  geom_bar(stat = "summary", fun="mean", position = "dodge")+
  labs(fill="Experienced weather events", y="Implements NbS", x="CC poses risk to farm yeild in the present"))

table4<-table(data$q2.5_1, data$EvFreq)
resCA4<- CA(table4)
EV4<-get_eigenvalue(resCA4)
cabiplot4<- fviz_ca_biplot(resCA4,map="rowprincipal", col.col="cadetblue", col.row="coral3",arrows = c( TRUE, TRUE),repel=TRUE)+
  labs(title="Relationship between risk to yeild today and experience with events")

table5<-table(data$q2.5_2, data$EvFreq)
resCA5<- CA(table5)
EV5<-get_eigenvalue(resCA5)
cabiplot5<- fviz_ca_biplot(resCA5,map="rowprincipal", col.col="cadetblue", col.row="coral3",arrows = c( TRUE, TRUE),repel=TRUE)+
  labs(title="Relationship between risk to farm bussiness today and experience with events")

table6<-table(data$q2.5_3, data$EvFreq)
resCA6<- CA(table6)
EV6<-get_eigenvalue(resCA6)
cabiplot6<- fviz_ca_biplot(resCA6,map="rowprincipal", col.col="cadetblue", col.row="coral3",arrows = c( TRUE, TRUE),repel=TRUE)+
  labs(title="Relationship between risk to farm yield in the future and experience with events")

table7<-table(data$q2.5_4, data$EvFreq)
resCA7<- CA(table7)
EV7<-get_eigenvalue(resCA7)
cabiplot7<- fviz_ca_biplot(resCA7,map="rowprincipal", col.col="cadetblue", col.row="coral3",arrows = c( TRUE, TRUE),repel=TRUE)+
  labs(title="Relationship between risk to farm bussiness in the future and experience with events")


#funding
graphFunding<- ggplot(data, aes(x=NbS,y=funding))+ #nada interesante
  geom_bar (stat = "summary",fun="mean")
(graphFunding2<-ggplot(data2, aes(x=factor(funding_cat, levels = c("Public", "Private", "Both", "Other", "None")),y=NbSyes))+
  geom_bar(stat="summary", fun="mean", fill="blue")+
  labs(x="Receives funding",y= "Implements NbS"))
summary(factor(data$funding_cat))


#income
data6<-filter(data, !is.na(q1.9))
(gIncome<- ggplot(data6, aes(x=factor(q1.9,levels = c("< 10%","10% to 20%", "20% to 30%", "30% to 40%", "40% to 50%", "50% to 60%", "> 60%")),y=NbSyes))+
  geom_bar(stat = "summary", fun="mean")+
    labs(x="Income other from farming", y="Implements NbS"))

data7<-filter(data2,!is.na(q1.5)) #Relacion importante entre implementar NbS y ingreso. No sé por qué me sale con NA la gráfica. checar
(Income2 <- ggplot(data7, aes(x=factor(q1.5, levels = c("< ¬£10,000", "¬£10,000 to ¬£20,000", "¬£20,000 to ¬£50,000", "¬£50,000 to ¬£100,000", "¬£100,000 to ¬£200,000", "> ¬£200,000")), y=NbSyes))+
  geom_bar(stat = "summary", fun="mean")+
  labs(x="Income", y="Implements NbS"))


data$edu <- with(data, ifelse(q1.4=="GCSE or A-levels","GCSE or A-levels",NA))
data$edu <- with(data, ifelse(q1.4=="Further education ‚Äì college course, diploma, trade training, etc.","Further education",data$edu))
data$edu <- with(data, ifelse(q1.4=="Higher education ‚Äì undergraduate level (e.g. BA, BSc)","Undergraduate",data$edu))
data$edu <- with(data, ifelse(q1.4=="Higher education ‚Äì postgraduate level (e.g. MA, MSc, PhD)","Postgraduate",data$edu))
data$edu <- with(data, ifelse(q1.4=="Other:","Other",data$edu))

(edu<- ggplot(data, aes(x=factor(edu, levels=c("GCSE or A-levels", "Further education","Undergraduate","Postgraduate","Other")),y=NbSyes))+
    geom_bar(stat = "summary", fun="mean")+
    labs(x="Highest level of educational attainment", y="Implements NbS"))

#perceptions
#no me ha salido esta grafica
data8<-filter(data, !is.na(q4.6_1))
tabcosts<-table(data$NbS,data$q4.6_1)
prop.table(tabcosts,1)
tabcosts2<-table(data$NbSyes,data$q4.6_1)
prop.table(tabcosts2,1)
tabcosts3<-table(data$NbScollab,data$q4.6_1)
prop.table(tabcosts3,1)

costs<- ggplot(data8, aes(fill=factor(NbS, levels = c("None", "Both", "Implements NbS in collaboration", "Implements NbS individually")),x=factor(q4.6_1)))+
  geom_bar(position="fill")+
  labs(x="NbS have higher costs than benefits",fill="Implements NbS:", y="Proportion")

costs4<- ggplot(data8, aes(y=(NbSyes),x=factor(q4.6_1)))+
  geom_bar(stat="summary", fun="mean", fill="chocolate1")+
  labs(x="NbS have higher costs than benefits")+
  stat_summary(fun = mean,geom = "text", aes(label = scales::percent(..y.., accuracy = 1)), vjust = -0.5,position = position_dodge(width = 0.9))



tableC<-table(data$q4.6_1, data$NbS)
resCAC<- CA(tableC)
EVC<-get_eigenvalue(resCAC)
cabiplotC<- fviz_ca_biplot(resCAC,map="rowprincipal", col.col="cadetblue", col.row="coral3",arrows = c( TRUE, TRUE),repel=TRUE)+
  labs(title="Relationship between cost perception and NbS implementation")


#intento de poner sd error bar
##weather events attribution to CC
dataCCatt<- data%>% group_by(CCattr)%>%summarise(propNbS=mean(NbSyes), sdNbS=sd(NbSyes), countNbS=sum(NbS_rep))
dataCCatt<-filter(dataCCatt, !is.na(CCattr))

ggplot(dataCCatt, aes(x=factor(CCattr,levels=c("Not at all","Partly","Mainly")), y=propNbS))+ ##Resultado mas interesante
  geom_bar(stat="summary", fun="mean", fill="grey")+
  labs(x="Weather events attribution to Climate change",y="Implements NbS", fill="CC attribution")+
  geom_errorbar(aes(ymin=propNbS-sdNbS, ymax = propNbS+sdNbS))+
  theme_minimal()

# la mayoría de la gente que se salió de la encuesta se salió después de responder su nivel de ingreso. Ver si hay relacion entre nivel de ingreso y que se hayan salido de la encuesta
data$q1.6rep<-with (data, ifelse(!is.na(q1.6),1,0))
tabRet<- table(data$q1.5, factor(data$q1.6rep))
prop.table(tabRet,1)
prop.table(tabRet,2)
#ver si cambio mucho distribucion
data$q1.5ret <- with (data, ifelse(!is.na(q1.6),q1.5,NA))
ggplot(data, aes(x=factor(q1.5, levels = c("< ¬£10,000", "¬£10,000 to ¬£20,000", "¬£20,000 to ¬£50,000", "¬£50,000 to ¬£100,000", "¬£100,000 to ¬£200,000", "> ¬£200,000")),fill=factor(q1.6rep)))+
  geom_bar()+
  labs(fill="Continued the survey")
#see proportion table
tabInc1<-table(factor(data$q1.5))
prop.table(tabInc1)

tabInc2<-table(factor(data$q1.5ret)) #los que no se salieron después de la pregunta
tabprob<-prop.table(tabInc2)

tabInc3 <- table(factor(complete_surveyed$q1.5)) #los que acabaron la encuesta
prop.table(tabInc3)



#hacer tabla de diferencia de medias?
#correlación entre dropear y cada level of income 
data$less10k <- with (data, ifelse(q1.5=="< ¬£10,000",1,0))
data$f10kto20k <- with (data, ifelse(q1.5=="¬£10,000 to ¬£20,000",1,0))
data$f20kto50k <- with (data, ifelse(q1.5=="¬£20,000 to ¬£50,000",1,0))
data$f50kto100k <- with (data, ifelse(q1.5=="¬£50,000 to ¬£100,000",1,0))
data$f100kto200k <- with (data, ifelse(q1.5=="¬£100,000 to ¬£200,000",1,0))
data$more200k <- with (data, ifelse(q1.5=="> ¬£200,000",1,0))
#correlación entre nivel de ingreso y los que acabaron el survey
data$Finished2<- with(data, ifelse(Finished=="True", 1, 0))



#self reported challenges
#correlation between reporting policy uncertainty and implementation costs + maintenance costs
data$chall_rep <-with(data, ifelse(!is.na(q5.2),1,0)) #respondió a pregunta de challenges
data$c_Impl<-with (data, ifelse(q5.2_4==1,1,0)) #implementation costs
data$c_Impl <- with(data, ifelse(chall_rep==1&is.na(c_Impl),0,data$c_Impl))

data$c_Maint<-with (data, ifelse(q5.2_5==1,1,0))
data$c_Maint <- with(data, ifelse(chall_rep==1&is.na(c_Maint),0,data$c_Maint)) #maintainance costs

data$c_policy<-with (data, ifelse(q5.2_10==1,1,0)) #policy uncertainty
data$c_policy <- with(data, ifelse(chall_rep==1&is.na(c_policy),0,data$c_policy)) 
cor.test(data$c_policy,data$c_Maint)
cor.test(data$c_policy,data$c_Impl)
prop.table(table(data$c_policy,data$c_Impl),2)
prop.table(table(data$c_policy,data$c_Maint),2)

data$c_tenancy <-with (data, ifelse(q5.2_14==1,1,0)) #tenancy constraints
data$c_tenancy <- with(data, ifelse(chall_rep==1&is.na(c_tenancy),0,data$c_tenancy)) 


#relación entre reto reportado de tenancy constraints y que sean tenants
data$own_rep <-with(data, ifelse(!is.na(q1.7),1,0)) #respondió respecto a si son farming land owner, tenant, worker, etc
data$own_tenant<-with (data, ifelse(q1.7_5==1|q1.7_6==1,1,0)) #tenants (either one of the two tenancy options on survey)
data$own_tenant <- with(data, ifelse(own_rep==1&is.na(own_tenant),0,data$own_tenant))


tenancy<-data%>%group_by(factor(own_tenant))%>%summarise(chal_ten=mean(c_tenancy, na.rm=TRUE))
tenancy<-data%>%group_by(factor(ownership))%>%summarise(chal_ten=mean(c_tenancy, na.rm=TRUE))

prop.table(table(data$c_tenancy,data$own_tenant),2)



data$own_yes<-with (data, ifelse(q1.7_1==1|q1.7_4==1,1,0)) #either farming landowner or non-farming landowner
data$own_yes <- with(data, ifelse(own_rep==1&is.na(own_yes),0,data$own_yes))

data$ownership <- with (data, ifelse(own_tenant==1, "Under tenancy", 0))
data$ownership <- with (data, ifelse(own_yes==1, "Owner occupied", data$ownership))
data$ownership <- with (data, ifelse(own_yes==1&own_tenant==1, "Mixed: owner & tenant", data$ownership))
data$ownership <- with (data, ifelse(ownership==0, "Other", data$ownership))

data$exc_owner<- with(data, ifelse(ownership=="Owner occupied",1,0))

data$own_rep2<- with(data, ifelse(own_rep==0, NA, own_rep))

data2<- filter(data,!is.na(ownership))

ggplot(data2, aes(x=factor(ownership,levels = c("Other","Mixed: owner & tenant","Under tenancy","Owner occupied"))))+
  geom_bar(aes(y = (..count..)/sum(..count..)), fill="blue3")+
  theme_minimal()+
  labs(x="", y="Proportion")+
 coord_flip()


  
  





#Correspondance analysis
#Preparación de datos:
##Quedarme con solo los que acabaron toda la encuesta
dataPCA <- filter(data, Finished=="True")
dataPCA[is.na(dataPCA)] <- 0 #cambiar todos NA a cero porque ya tengo todos los que respondieron
############Preguntar a Katie si con las dummies debo de dejar una categoría como con las regresiones, que no pones en la regresion dummy de hombre y de mujer sino solo una
#Region-what should I exclude? What to do with categories with little observations, i.e. Ireland --juntar todas las chiquitas en una categoría "other"? Excluirlas del análisis?
##no hubo ninguna observación del North East
dataPCA$regionSW<- with(dataPCA, ifelse(q1.1=="South West",1,0))
dataPCA$regionSE<- with(dataPCA, ifelse(q1.1=="South East",1,0))
dataPCA$regionLondon<- with(dataPCA, ifelse(q1.1=="Greater London",1,0))
dataPCA$regionWM<- with(dataPCA, ifelse(q1.1=="West Midlands",1,0))
dataPCA$regionEM<- with(dataPCA, ifelse(q1.1=="East Midlands",1,0))
dataPCA$regionEA<- with(dataPCA, ifelse(q1.1=="East Anglia",1,0))
dataPCA$regionNW<- with(dataPCA, ifelse(q1.1=="North West",1,0))
dataPCA$regionYH<- with(dataPCA, ifelse(q1.1=="Yorkshire and Humber",1,0))
dataPCA$regionW<- with(dataPCA, ifelse(q1.1=="Wales",1,0))
dataPCA$regionS<- with(dataPCA, ifelse(q1.1=="Scotland",1,0)) 
dataPCA$regionNI<- with(dataPCA, ifelse(q1.1=="Northern Ireland",1,0)) 

#agegroup - excluir uno?
dataPCA$age24 <- with(dataPCA, ifelse(q1.2=="< 24",1,0))
dataPCA$age25_34 <- with(dataPCA, ifelse(q1.2=="25 to 34",1,0))
dataPCA$age35_44 <- with(dataPCA, ifelse(q1.2=="35 to 44",1,0))
dataPCA$age45_54 <- with(dataPCA, ifelse(q1.2=="45 to 54",1,0))
dataPCA$age55_64 <- with(dataPCA, ifelse(q1.2=="55 to 64",1,0))
dataPCA$age65_74 <- with(dataPCA, ifelse(q1.2=="65 to 74",1,0))
dataPCA$age75 <- with(dataPCA, ifelse(q1.2==">75 years old",1,0))

dataPCA$woman <- with(dataPCA, ifelse(q1.3=="Woman",1,0))

#education - excluir uno?
dataPCA$eduGCSE <- with(dataPCA, ifelse(edu=="GCSE or A-levels",1,0))
dataPCA$eduFurther <- with(dataPCA, ifelse(edu=="Further education",1,0))
dataPCA$eduUnderG <- with(dataPCA, ifelse(edu=="Undergraduate",1,0))
dataPCA$eduPostG <- with(dataPCA, ifelse(edu=="Postgraduate",1,0))
dataPCA$eduOther <- with(dataPCA, ifelse(edu=="Other",1,0))

#income ya está

#farm size
dataPCA$size5 <- with(dataPCA, ifelse(q1.6=="< 5 ha.",1,0))
dataPCA$size5_20 <- with(dataPCA, ifelse(q1.6=="5 to 20 ha.",1,0))
dataPCA$size20_50 <- with(dataPCA, ifelse(q1.6=="20 to 50 ha.",1,0))
dataPCA$size50_100 <- with(dataPCA, ifelse(q1.6=="50 to 100 ha.",1,0))
dataPCA$size100_250 <- with(dataPCA, ifelse(q1.6=="100-250 ha.",1,0))
dataPCA$size250 <- with(dataPCA, ifelse(q1.6==">250 ha",1,0))

#rename las variables de tenancy
dataPCA$ownFarmOwner <- dataPCA$q1.7_1 #farming landowner
dataPCA$ownNonFarmOwner <- dataPCA$q1.7_4 #non farming landowner
dataPCA$ownFBT <- dataPCA$q1.7_5 #farm business tenancy
dataPCA$ownAH <- dataPCA$q1.7_6 #agricultural holdings tenancy
#variable dataPCA$own_tenant junta las dos. Usar solo esta en vez de las dos de arriba
dataPCA$ownWorker <- dataPCA$q1.7_7 #farm worker
dataPCA$ownFam <- dataPCA$q1.7_8 #farmily member
dataPCA$ownOther <- dataPCA$q1.7_9 #farm worker

#farm activities
dataPCA$act_cereal <- dataPCA$q1.8_1
dataPCA$act_horti <- dataPCA$q1.8_4
dataPCA$act_oil <- dataPCA$q1.8_11 #Cropping: biofuels/seed oils
dataPCA$act_dairy <- dataPCA$q1.8_5
dataPCA$act_lowGraz <- dataPCA$q1.8_6 #Lowland Grazing livestock
dataPCA$act_LFAgraz <- dataPCA$q1.8_7 #LFA grazing livestock
dataPCA$act_pig <- dataPCA$q1.8_8 #specialist pigs
dataPCA$act_poultry <- dataPCA$q1.8_9 #specialist poultry
dataPCA$act_mix <- dataPCA$q1.8_10 #mixed
dataPCA$act_other <- dataPCA$q1.8_12

#what percentage of your household income comes from activities other than farming (excluding subsidies)?
dataPCA$income_nofarm_10 <- with(dataPCA,ifelse(q1.9=="< 10%",1,0))
dataPCA$income_nofarm_10to20 <- with(dataPCA,ifelse(q1.9=="10% to 20%",1,0))
dataPCA$income_nofarm_20to30 <- with(dataPCA,ifelse(q1.9=="20% to 30%",1,0))
dataPCA$income_nofarm_30to40 <- with(dataPCA,ifelse(q1.9=="30% to 40%",1,0))
dataPCA$income_nofarm_40to50 <- with(dataPCA,ifelse(q1.9=="40% to 50%",1,0))
dataPCA$income_nofarm_50to60 <- with(dataPCA,ifelse(q1.9=="50% to 60%",1,0))
dataPCA$income_nofarm_60 <- with(dataPCA,ifelse(q1.9=="> 60%",1,0))

#para funding: usar los principales y juntar los demas
dataPCA$SFI <- dataPCA$q1.10_1 #sustainable farming incentives
dataPCA$CS <- dataPCA$q1.10_4 #country stewardships
dataPCA$CSHT <- dataPCA$q1.10_5 #country stewardships higher tier
dataPCA$CI <- dataPCA$q1.10_7 #capital items
dataPCA$FiPL <- dataPCA$q1.10_12 #farming in protected landscapes
dataPCA$pubOther <- with(dataPCA, ifelse(q1.10_6==1|q1.10_8==1|q1.10_9==1|q1.10_10==1| q1.10_11==1| q1.10_13==1| q1.10_14==1| q1.10_15==1| q1.10_16==1,1,0))
dataPCA$noFund <- dataPCA$q1.10_21

#Experience with weather changes: usar events==1 si han vivido cualquiera o cada categoría por separado? por ahora cada categoría por separado
dataPCA$flood <- dataPCA$q2.1_1
dataPCA$drought <- dataPCA$q2.1_4
dataPCA$wildfire <- dataPCA$q2.1_5
dataPCA$disease <- dataPCA$q2.1_6
dataPCA$pests <- dataPCA$q2.1_8
dataPCA$other <- dataPCA$q2.1_10
dataPCA$noEvents <- dataPCA$q2.1_11

#how often experiences weather events
dataPCA$ev_moreyear <- ifelse(dataPCA$q2.2=="More than once a year", 1,0)
dataPCA$ev_year <- ifelse(dataPCA$q2.2=="Once a year", 1,0)
dataPCA$ev_2to3year <- ifelse(dataPCA$q2.2=="Once every 2-3 years", 1,0)
dataPCA$ev_5year <- ifelse(dataPCA$q2.2=="Once in the last five years", 1,0)

#how much affected
dataPCA$afect_no <- with(dataPCA, ifelse(q2.3=="Not affected at all",1,0))
dataPCA$afect_some <- with(dataPCA, ifelse(q2.3=="Somewhat affected",1,0))
dataPCA$afect_severe <- with(dataPCA, ifelse(q2.3=="Severely affected",1,0))

#cc attribution
dataPCA$cc_no <- with(dataPCA, ifelse(q2.4=="Not at all caused by climate change",1,0))
dataPCA$cc_part <- with(dataPCA, ifelse(q2.4=="Partly caused by climate change",1,0))
dataPCA$cc_main <- with(dataPCA, ifelse(q2.4=="Mainly caused by climate change",1,0))







#Dataset for MCA
dataMCA <- filter(data, Finished=="True")
dataMCA[is.na(dataMCA)] <- 0 #cambiar todos NA a cero porque ya tengo todos los que respondieron


dataMCA$ownFarmOwner <- dataMCA$q1.7_1 #farming landowner
dataMCA$ownNonFarmOwner <- dataMCA$q1.7_4 #non farming landowner
dataMCA$ownWorker <- dataMCA$q1.7_7 #farm worker
dataMCA$ownFam <- dataMCA$q1.7_8 #farmily member
dataMCA$ownOther <- dataMCA$q1.7_9 #farm worker

#farm activities
dataMCA$act_cereal <- dataMCA$q1.8_1
dataMCA$act_horti <- dataMCA$q1.8_4
dataMCA$act_oil <- dataMCA$q1.8_11 #Cropping: biofuels/seed oils
dataMCA$act_dairy <- dataMCA$q1.8_5
dataMCA$act_lowGraz <- dataMCA$q1.8_6 #Lowland Grazing livestock
dataMCA$act_LFAgraz <- dataMCA$q1.8_7 #LFA grazing livestock
dataMCA$act_pig <- dataMCA$q1.8_8 #specialist pigs
dataMCA$act_poultry <- dataMCA$q1.8_9 #specialist poultry
dataMCA$act_mix <- dataMCA$q1.8_10 #mixed
dataMCA$act_other <- dataMCA$q1.8_12

#funding
dataMCA$SFI <- dataMCA$q1.10_1 #sustainable farming incentives
dataMCA$CS <- dataMCA$q1.10_4 #country stewardships
dataMCA$CSHT <- dataMCA$q1.10_5 #country stewardships higher tier
dataMCA$CI <- dataMCA$q1.10_7 #capital items
dataMCA$FiPL <- dataMCA$q1.10_12 #farming in protected landscapes
dataMCA$pubOther <- with(dataMCA, ifelse(q1.10_6==1|q1.10_8==1|q1.10_9==1|q1.10_10==1| q1.10_11==1| q1.10_13==1| q1.10_14==1| q1.10_15==1| q1.10_16==1,1,0))
dataMCA$noFund <- dataMCA$q1.10_21

#Experience with weather changes: usar events==1 si han vivido cualquiera o cada categoría por separado? por ahora cada categoría por separado
dataMCA$flood <- dataMCA$q2.1_1
dataMCA$drought <- dataMCA$q2.1_4
dataMCA$wildfire <- dataMCA$q2.1_5
dataMCA$disease <- dataMCA$q2.1_6
dataMCA$pests <- dataMCA$q2.1_8
dataMCA$event_other <- dataMCA$q2.1_10
dataMCA$noEvents <- dataMCA$q2.1_11

dataMCA$q2.2 <- with(dataMCA, ifelse(q2.2==0,"Not experienced events",q2.2))
dataMCA$q2.3 <- with(dataMCA, ifelse(q2.3==0,NA,q2.3))
dataMCA$CCattr <- with(dataMCA, ifelse(CCattr==0,NA,CCattr))

#risk perception: change name
dataMCA$risk_YieldToday <- dataMCA$q2.5_1 #cc poses risk to farms yield in the present moment
dataMCA$risk_BussiToday <- dataMCA$q2.5_2 #Climate change will pose risks to my farm's yield at the present
dataMCA$risk_YieldFuture <- dataMCA$q2.5_3  #Climate change will pose risks to my farm's yield in the future
dataMCA$risk_BussiFuture <- dataMCA$q2.5_4 # Climate change will pose risks to my farming business in the future

dataMCA$partner <- with (dataMCA, ifelse(q3.1=="Yes (If you wish, specify which kind of community partnership)","Yes","No"))

#community partnership benefits
dataMCA$partner_Nat<-dataMCA$q3.2.1_3 #My community partnership carries out nature positive activities
dataMCA$partner_Prod<-dataMCA$q3.2.1_4 #My community partnership carries out activities that increase my farm’s productivity

dataMCA$NbSNo <- with(dataMCA, ifelse(q4.1=="No",1,0))

#MCA
dataMCA2<- subset(dataMCA, select = c(q1.1,q1.2,q1.3,edu,q1.5,q1.6,own_tenant, ownFarmOwner, ownNonFarmOwner,ownWorker,ownFam,ownOther,act_cereal, act_horti,act_oil,act_dairy,act_lowGraz, act_LFAgraz,act_pig, act_poultry,act_mix,act_other,q1.9,SFI,CS,CSHT,CI,FiPL,pubOther,fundingPriv,noFund,flood, drought,wildfire,disease,pests,event_other,noEvents,q2.2, q2.3, CCattr,risk_YieldToday, risk_BussiToday, risk_YieldFuture, risk_BussiFuture, partner, collabImportant, collabProd,partner_Nat,partner_Prod, NbSindiv, NbScollab,NbSNo, q4.2_1,q4.2_4, q4.2_10, q4.2_5, q4.2_11, q4.2_12, q4.2_13, q4.2_6, q4.2_7, q4.2_14, q4.2_8, q4.2_9, q4.3,q4.4, neigh,q4.6_1,q4.6_2, q4.6_3,q4.6_4,q4.6_5,q4.6_7,q4.6.1_6,q5.1_1,q5.1_2, q5.1_3,q5.1_4,q5.2_1,q5.2_4,q5.2_5,q5.2_6,q5.2_7,q5.2_8,q5.2_9,q5.2_10,q5.2_12,q5.2_13,q5.2_14,q5.2_11,q5.2_15,q5.3_1,q5.3_4,q5.3_5,q5.3_6,q5.3_7,q5.3_8))

mcaAll<-MCA(dataMCA2, graph = TRUE)
print(mcaAll)

eig.val <- get_eigenvalue(mcaAll)

ind_coord<-mcaAll$ind$coord
print(ind_coord)

var_coord<-mcaAll$var$coord
print(var_coord)

fviz_screeplot(mcaAll, addlabels = TRUE, ylim = c(0, 45))

rownames(mcaAll$var$coord) <- make.unique(rownames(mcaAll$var$coord))
fviz_mca_biplot(mcaAll,
                repel = TRUE, # Avoid text overlapping (slow if many point)
                ggtheme = theme_minimal())

fviz_contrib(mcaAll, choice = "var", axes = 1, top = 15)
fviz_contrib(mcaAll, choice = "var", axes = 2, top = 15)



#MCA with less variables -attempt 2
dataMCA$SFIyes<-ifelse(dataMCA$SFI==1,"SFIyes","SFIno")
dataMCA$CSyes<-ifelse(dataMCA$CS==1,"CSyes","CSno")
dataMCA$CSHTyes<-ifelse(dataMCA$CSHT==1,"CSHTyes","CSHTno")
dataMCA$CIyes<-ifelse(dataMCA$CI==1,"CIyes","CIno")
dataMCA$FiPLyes<-ifelse(dataMCA$FiPL==1,"FiPLyes","FiPLno")
dataMCA$fundingPrivyes<-ifelse(dataMCA$fundingPriv==1,"fundPriv_yes","fundPriv_no")
dataMCA$pubOtheryes<-ifelse(dataMCA$pubOther==1,"pubOther_yes","pubOther_no")
dataMCA$noFundlab<-with(dataMCA,ifelse(noFund==1,"no_Funding","Yes_funding"))

dataMCA$NbSindiv_yes<-ifelse(dataMCA$NbSindiv==1,"NbSindiv_yes","NbSindiv_no")
dataMCA$NbScollab_yes<-ifelse(dataMCA$NbScollab==1,"NbScollab_yes","NbScollab_no")

dataMCA$q4.3sup <- with(dataMCA, ifelse(q4.3=="1-5%","exist_1-5%",0))
dataMCA$q4.3sup <- with(dataMCA, ifelse(q4.3=="5-10%","exist_5-10%",dataMCA$q4.3sup))
dataMCA$q4.3sup <- with(dataMCA, ifelse(q4.3=="10-25%","exist_10-25%",q4.3sup))
dataMCA$q4.3sup <- with(dataMCA, ifelse(q4.3=="Greater than 25%","exist_Greater25%",q4.3sup))
dataMCA$q4.3sup <- with(dataMCA, ifelse(q4.3=="Less than 1%","exist_Less1%",q4.3sup))

dataMCA$q4.4change <- with(dataMCA, ifelse(q4.4=="1-5%","change_1-5%",0))
dataMCA$q4.4change <- with(dataMCA, ifelse(q4.4=="5-10%","change_5-10%",dataMCA$q4.4change))
dataMCA$q4.4change <- with(dataMCA, ifelse(q4.4=="10-25%","change_10-25%",q4.4change))
dataMCA$q4.4change <- with(dataMCA, ifelse(q4.4=="Greater than 25%","change_Greater25%",q4.4change))
dataMCA$q4.4change <- with(dataMCA, ifelse(q4.4=="Less than 1%","change_Less1%",q4.4change))


dataMCA$govPromo <- with(dataMCA, ifelse(q5.1_1=="3 - Agree", "govPromo_Agree",0))
dataMCA$govPromo <- with(dataMCA, ifelse(q5.1_1=="1- Disagree", "govPromo_Disagree",dataMCA$govPromo))
dataMCA$govPromo <- with(dataMCA, ifelse(q5.1_1=="2- Neither agree nor disagree", "govPromo_neutral",dataMCA$govPromo))

dataMCA$govConf <- with(dataMCA, ifelse(q5.1_2=="3 - Agree", "govConf_Agree",0))
dataMCA$govConf <- with(dataMCA, ifelse(q5.1_2=="1- Disagree", "govConf_Disagree",dataMCA$govConf))
dataMCA$govConf <- with(dataMCA, ifelse(q5.1_2=="2- Neither agree nor disagree", "govConf_neutral",dataMCA$govConf))

dataMCA$neighLab<- with(dataMCA, ifelse(neigh=="Yes","neigh_Yes",0))
dataMCA$neighLab<- with(dataMCA, ifelse(neigh=="No","neigh_No",dataMCA$neighLab))
dataMCA$neighLab<- with(dataMCA, ifelse(neigh=="I don't know","neigh_DontKnow",dataMCA$neighLab))

dataMCA$partnerlab<-with(dataMCA, ifelse(partnerlab=="Yes","partner_Yes",dataMCA$partnerlab))
dataMCA$partnerlab<-with(dataMCA, ifelse(partnerlab=="No","partner_No",dataMCA$partnerlab))


dataMCA3<- subset(dataMCA, select = c(q1.2,q1.3,edu,q1.5,ownership,
                                    q1.9,
                                      SFIyes,
                                      CSyes,
                                      CSHTyes,
                                      CIyes,
                                      FiPLyes,
                                      fundingPrivyes,
                                      pubOtheryes,
                                      noFundlab,
                                      EvFreq,
                                      q2.3,
                                      CCattr,
                                      risk_YieldFuture,
                                      partnerlab,
                                      NbSindiv_yes,
                                      NbScollab_yes,
                                      q4.3sup,
                                      q4.4change,
                                      neighLab,
                                      govPromo,
                                      govConf
                                      ))


res.mca<-MCA(dataMCA3,quali.sup = 2, graph = TRUE)

eig.val <- get_eigenvalue(mcaSUB)
fviz_screeplot(res.mca)
fviz_contrib(res.mca, choice = "var", axes = 1, top = 25)
fviz_contrib(res.mca, choice = "var", axes = 2, top=25)
library("corrplot")
var <- get_mca_var(res.mca)
corrplot(var$cos2, is.corr=FALSE)
plot(res.mca, invisible = c("quali.sup", "ind"), cex=1, col.var = "darkblue", title = "Active categories", cex.main=2, col.main= "darkblue",repel=TRUE)
fviz_mca_var(res.mca,
                repel = F,
             col.var = "cos2", # Color by the quality of representation
             gradient.cols = c("lightgrey", "darkorange", "darkred"),
                ggtheme = theme_minimal())


##intento 3: menos variables: **USAR ESTE 
dataMCA$fundingPubyes<-ifelse(dataMCA$fundingPub==1,"fundPub_yes","fundPub_no")

dataMCA4<- subset(dataMCA, select = c(q1.5,
                                      q1.9,
                                      fundingPubyes,
                                      fundingPrivyes,
                                      EvFreq,
                                      CCattr,
                                      risk_YieldFuture,
                                      partnerlab,
                                      NbSindiv_yes,
                                      NbScollab_yes,
                                      q4.3sup,
                                      q4.4change,
                                      neighLab,
                                      govPromo,
                                      govConf
))

res.mca2<-MCA(dataMCA4, graph = TRUE)
summary(res.mca2, ncp=3,nbelements = Inf)
dimdesc(res.mca2)

eig.val <- get_eigenvalue(res.mca2)
fviz_screeplot(res.mca2, addlabels=T)
fviz_contrib(res.mca2, choice = "var", axes = 1, top = 25)
fviz_contrib(res.mca2, choice = "var", axes = 2, top=25)
fviz_contrib(res.mca2, choice = "var", axes = 3, top = 25)
fviz_contrib(res.mca2, choice = "var", axes = 4, top=25)

plot(res.mca2, invisible = "ind", autoLab = "y", cex=0.7, selectMod = "cos2 20") #enseña las 10 categorías with higher quality of representation
plot(res.mca2, invisible = "ind", autoLab = "y", cex=0.7, selectMod = "contrib 20") #enseña las 20 categorías with higher level of contribution

fviz_mca_var(res.mca2,
             repel = T,
             col.var = "cos2", # Color by the quality of representation
             gradient.cols = c("lightgrey", "darkorange", "darkorchid4"),
             ggtheme = theme_minimal())

#con Burt method - es lo mismo nada más el inertia se infla
res.mca3<-MCA(dataMCA4, graph = TRUE, method="Burt")
eig.val <- get_eigenvalue(res.mca3)
fviz_screeplot(res.mca3, addlabels=T)
fviz_contrib(res.mca3, choice = "var", axes = 1, top = 25)
fviz_contrib(res.mca3, choice = "var", axes = 2, top=25)
fviz_contrib(res.mca3, choice = "var", axes = 3, top = 25)
fviz_contrib(res.mca3, choice = "var", axes = 4, top=25)

plot(res.mca3, invisible = "ind", autoLab = "y", cex=0.7, selectMod = "cos2 10") #enseña las 10 categorías with higher quality of representation
plot(res.mca3, invisible = "ind", autoLab = "y", cex=0.7, selectMod = "contrib 20") #enseña las 20 categorías with higher level of contribution


#intento 4: agregar solo education y age
dataMCA5<- subset(dataMCA, select = c(edu,q1.5,
                                      q1.9,
                                      fundingPubyes,
                                      fundingPrivyes,
                                      pubOtheryes,
                                      EvFreq,
                                      CCattr,
                                      risk_YieldFuture,
                                      partnerlab,
                                      NbSindiv_yes,
                                      NbScollab_yes,
                                      q4.3sup,
                                      q4.4change,
                                      neighLab,
                                      govPromo,
                                      govConf
))

res.mca2<-MCA(dataMCA5, graph = TRUE)

eig.val <- get_eigenvalue(res.mca2)
fviz_screeplot(res.mca2, addlabels=T)
fviz_contrib(res.mca2, choice = "var", axes = 1, top = 25)
fviz_contrib(res.mca2, choice = "var", axes = 2, top=25)
fviz_contrib(res.mca2, choice = "var", axes = 3, top = 25)
fviz_contrib(res.mca2, choice = "var", axes = 4, top=25)

fviz_mca_var(res.mca2,
             repel = T,
             col.var = "cos2", # Color by the quality of representation
             gradient.cols = c("lightgrey", "darkorange", "darkorchid4"),
             ggtheme = theme_minimal())



##intento 5: menos variables: **USAR ESTE 
dataMCA$NbSy<- ifelse(dataMCA$NbSyes==1,"NbSyes","NbSno")
dataMCA5<-filter(dataMCA, !is.na(CCattr))
dataMCA5$income<-dataMCA5$q1.5

dataMCA5<- subset(dataMCA5, select = c(income,
                                       noFundlab,
                                       EvFreq,
                                       CCattr,
                                       risk_YieldFuture,
                                       partnerlab,
                                       NbSy,
                                       q4.4change,
                                       neighLab,
                                       govPromo
))

res.mca5<-MCA(dataMCA5, graph = TRUE)
summary(res.mca5, ncp=3,nbelements = Inf)
dimdesc(res.mca5)

eig.val <- get_eigenvalue(res.mca5)
get_mca_var(res.mca5, element = "var")
fviz_screeplot(res.mca5, addlabels=T)
fviz_contrib(res.mca5, choice = "var", axes = 1, top = 20)
fviz_contrib(res.mca5, choice = "var", axes = 2, top=20)
fviz_contrib(res.mca5, choice = "var", axes = 3, top = 25)
fviz_contrib(res.mca5, choice = "var", axes = 4, top=25)

plot(res.mca5, invisible = "ind", autoLab = "y", cex=0.7, selectMod = "cos2 20") #enseña las 10 categorías with higher quality of representation
plot(res.mca5, invisible = "ind", autoLab = "y", cex=0.7, selectMod = "contrib 20") #enseña las 20 categorías with higher level of contribution

fviz_mca_var(res.mca5,
             repel = T,
             col.var = "cos2", # Color by the quality of representation
             gradient.cols = c("lightgrey", "darkorange", "darkorchid4"),
             ggtheme = theme_minimal()+
               labs)

fviz_mca_ind(res.mca5,
             repel = T,
             habillage = dataMCA5$CCattr,
             ggtheme = theme_minimal())+labs(color= "Attribution to CC")

fviz_mca_ind(res.mca5,
             repel = T,
             habillage = "NbSy",
             ggtheme = theme_minimal())+labs(color= "Implements NbS")

fviz_mca_ind(res.mca5,
             repel = T,
             habillage = "q1.5",
             ggtheme = theme_minimal())+labs(color= "Income")

fviz_mca_ind(res.mca5,
             repel = T,
             habillage = "partnerlab",
             ggtheme = theme_minimal())+labs(color= "Belongs to partnership")

fviz_mca_ind(res.mca5,
             repel = T,
             habillage = "neighLab",
             ggtheme = theme_minimal())+labs(color= "Neighbour implements NbS")+scale_color_discrete(labels = c("I don't know", "No", "Yes"))

fviz_mca_ind(res.mca5,
             repel = T,
             habillage = "noFundlab",
             ggtheme = theme_minimal())+labs(color= "Receives funding")+scale_color_discrete(labels = c("No", "Yes"))

fviz_mca_ind(res.mca5,
             repel = T,
             habillage = "govPromo",
             ggtheme = theme_minimal())

dataMCA5$EvFreq<- factor(dataMCA5$EvFreq, levels = c("More than once a year","Once a year", "Once every 2-3 years","Once in the last five years","Never"))

fviz_mca_ind(res.mca5,
             repel = T,
             habillage = "EvFreq",
             ggtheme = theme_minimal())+labs(color= "Experienced weather events")

fviz_mca_ind(res.mca5,
             repel = T,
             habillage = "q4.4change",
             ggtheme = theme_minimal())

fviz_mca_ind(res.mca5,
             repel = T,
             habillage = "risk_YieldFuture",
             ggtheme = theme_minimal())+
  labs(color= "CC poses risk to my farm's yield in the future")


#clusters - kmeans

resMCA_km=data.frame(res.mca5$ind$coord)
set.seed(382)

fviz_nbclust(resMCA_km, kmeans, method = "wss")
fviz_nbclust(resMCA_km, kmeans, method = "silhouette")


km <- kmeans(resMCA_km, centers = 3, nstart = 25)
fviz_cluster(km, data = resMCA_km, geom = "point", ellipse.type = "convex") +
  theme_minimal()
print(km)

km2 <- kmeans(resMCA_km, centers = 4, nstart = 25)
fviz_cluster(km2, data = resMCA_km, geom = "point", ellipse.type = "convex") +
  theme_minimal()
print(km2)

km3 <- kmeans(resMCA_km, centers = 9, nstart = 25)
fviz_cluster(km3, data = resMCA_km, geom = "point", ellipse.type = "convex") +
  theme_minimal()


dataMCA4$cluster<-km$cluster
table(dataMCA4$cluster)

datakm<-dataMCA4 %>%
  group_by(cluster) %>%
  summarise(across(everything(), ~ names(sort(table(.), decreasing = TRUE))[1])) 

#clusters-hierarchical
res.hcpc <- HCPC (res.mca5, graph = FALSE)
fviz_dend(res.hcpc, show_labels = FALSE)
# Individuals facor map
fviz_cluster(res.hcpc, geom = "point", main = "Factor map")
# Description by variables
res.hcpc$desc.var$test.chi2
# Description by variable categories
res.hcpc$desc.var$category
#description by individuals
res.hcpc$desc.ind$para



vtest_data<-res.hcpc$desc.var$category
vtest_long <- bind_rows(
  vtest_long <- bind_rows(
    lapply(seq_along(vtest_data), function(i) {
      df <- as.data.frame(vtest_data[[i]])
      df$Cluster <- paste0("Cluster ", i)
      df$Category <- rownames(df)
      df
    })
  ))
heatmap_df <- vtest_long %>%
  select(Cluster, Category, v.test, p.value, `Cla/Mod`,`Mod/Cla`)
heatmap_df$vtest2<-with(heatmap_df,ifelse(v.test>0,"+","-"))


ggplot(heatmap_df, aes(x = Cluster, y = Category, fill = `Mod/Cla`)) + #con Mod Cla
  geom_tile()+
  geom_text(aes(label = paste0(round(`Mod/Cla`, 2)," ",vtest2)), size = 3) +
  scale_fill_gradient2(
    low = "orchid4", mid = "white", high = "darkred", midpoint = 0,
    name = "Mod/Cla"
  ) +
  theme_minimal() +
  labs(
       x = "Cluster", y = "Categories",caption = "'+' v.test > 1.96; '-' v-test < -1.96") +
  theme(axis.text.y = element_text(size = 8),panel.grid = element_blank(),
        panel.border = element_blank(), 
        axis.ticks = element_blank(),
        plot.caption = element_text(hjust = 1, size = 8))

ggplot(heatmap_df, aes(x = Cluster, y = Category, fill = `Cla/Mod`)) + #Con Cla/Mod
  geom_tile()+
  geom_text(aes(label = paste0(round(`Cla/Mod`, 2)," ",vtest2)), size = 3) +
  scale_fill_gradient2(
    low = "orchid4", mid = "white", high = "darkred", midpoint = 0,
    name = "Cla/Mod"
  ) +
  theme_minimal() +
  labs(
    x = "Cluster", y = "Categories",caption = "'+' v.test > 1.96; '-' v-test < -1.96") +
  theme(axis.text.y = element_text(size = 8),panel.grid = element_blank(),
        panel.border = element_blank(), 
        axis.ticks = element_blank(),
        plot.caption = element_text(hjust = 1, size = 8))

ggplot(heatmap_df, aes(x = Cluster, y = Category, fill = v.test)) + #Con v.test
  geom_tile()+
  scale_fill_gradient2(
    low = "orchid4", mid = "white", high = "darkred", midpoint = 0,
    name = "v.test"
  ) +
  theme_minimal() +
  labs(
    x = "Cluster", y = "Categories") +
  theme(axis.text.y = element_text(size = 8),panel.grid = element_blank(),
        panel.border = element_blank(), 
        axis.ticks = element_blank())
        

#logit models

data$risk_YieldToday <- data$q2.5_1 #cc poses risk to farms yield in the present moment
data$risk_BussiToday <- data$q2.5_2 #Climate change will pose risks to my farm's yield at the present
data$risk_YieldFuture <- data$q2.5_3  #Climate change will pose risks to my farm's yield in the future
data$risk_BussiFuture <- data$q2.5_4 # Climate change will pose risks to my farming business in the future
data$woman <- with(data, ifelse(q1.3=="Woman",1,0))

data$q1.1<-as.factor(data$q1.1)
data$q1.2<-as.factor(data$q1.2)
data$risk_YieldToday<-as.factor(data$risk_YieldToday)
data$ownership<-as.factor(data$ownership)
data$region2<- with(data, ifelse(q1.1=="South West", "South West","Other_regions")) #hice esto porque me da error porque algunas regiones que tienen pocas respuestas todos respondieron lo mismo en "NbSyes
data$region2<- with(data, ifelse(q1.1=="East Anglia", "East Anglia",data$region2))
data$region2<- with(data, ifelse(q1.1=="Scotland", "Scotland",data$region2))
data$age2<- with(data, ifelse(q1.2=="< 24"|q1.2=="25 to 34","< 35",data$q1.2))


#risk - sin agregar ingreso no sale ningun resultado significativo
modRisk1<- glm(NbSyes ~ risk_YieldToday+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))
modRisk2<- glm(NbSyes ~ risk_YieldFuture+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))
modRisk3<- glm(NbSyes ~ risk_BussiToday+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))
modRisk4<- glm(NbSyes ~ risk_BussiFuture+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))

modRisk1ind<- glm(NbSindiv ~ risk_YieldToday+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))
modRisk2ind<- glm(NbSindiv ~ risk_YieldFuture+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))
modRisk3ind<- glm(NbSindiv ~ risk_BussiToday+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))
modRisk4ind<- glm(NbSindiv ~ risk_BussiFuture+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))

modRisk1col<- glm(NbScollab ~ risk_YieldToday+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))
modRisk2col<- glm(NbScollab ~ risk_YieldFuture+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))
modRisk3col<- glm(NbScollab ~ risk_BussiToday+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))
modRisk4col<- glm(NbScollab ~ risk_BussiFuture+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))

margRisk1<- margins(modRisk1)
margRisk2<- margins(modRisk2)
margRisk3<- margins(modRisk3)
margRisk4<- margins(modRisk4)

margRisk1ind<- margins(modRisk1ind)
margRisk2ind<- margins(modRisk2ind)
margRisk3ind<- margins(modRisk3ind)
margRisk4ind<- margins(modRisk4ind)

margRisk1col<- margins(modRisk1col)
margRisk2col<- margins(modRisk2col)
margRisk3col<- margins(modRisk3col)
margRisk4col<- margins(modRisk4col)

#sin ingreso de control
modRisk2.2<- glm(NbSyes ~ risk_YieldFuture+woman+age2+region2, data=data, family=binomial (link = "logit"))
modRisk2.2ind<- glm(NbSindiv ~ risk_YieldFuture+woman+age2+region2, data=data, family=binomial (link = "logit"))
modRisk2.2col<- glm(NbScollab ~ risk_YieldFuture+woman+age2+region2, data=data, family=binomial (link = "logit"))

modRisk4.2<- glm(NbSyes ~ risk_BussiFuture+woman+age2+region2, data=data, family=binomial (link = "logit"))
modRisk4.2ind<- glm(NbSindiv ~ risk_BussiFuture+woman+age2+region2, data=data, family=binomial (link = "logit"))
modRisk4.2col<- glm(NbScollab ~ risk_BussiFuture+woman+age2+region2, data=data, family=binomial (link = "logit"))

margRisk2.2<- margins(modRisk2.2)
margRisk2.2ind<- margins(modRisk2.2ind)
margRisk2.2col<- margins(modRisk2.2col)

margRisk4.2<- margins(modRisk4.2)
margRisk4.2ind<- margins(modRisk4.2ind)
margRisk4.2col<- margins(modRisk4.2col)

#controlando por climate change events
modRisk2.3<- glm(NbSyes ~ risk_YieldFuture+woman+age2+region2+q1.5+EvFreq, data=data, family=binomial (link = "logit"))
modRisk2.3ind<- glm(NbSindiv ~ risk_YieldFuture+woman+age2+region2+q1.5+EvFreq, data=data, family=binomial (link = "logit"))
modRisk2.3col<- glm(NbScollab ~ risk_YieldFuture+woman+age2+region2+q1.5+EvFreq, data=data, family=binomial (link = "logit"))

modRisk4.3<- glm(NbSyes ~ risk_BussiFuture+woman+age2+region2+q1.5+EvFreq, data=data, family=binomial (link = "logit"))
modRisk4.3ind<- glm(NbSindiv ~ risk_BussiFuture+woman+age2+region2+q1.5+EvFreq, data=data, family=binomial (link = "logit"))
modRisk4.3col<- glm(NbScollab ~ risk_BussiFuture+woman+age2+region2+q1.5+EvFreq, data=data, family=binomial (link = "logit"))

margRisk2.3<- margins(modRisk2.3)
margRisk2.3ind<- margins(modRisk2.3ind)
margRisk2.3col<- margins(modRisk2.3col)

margRisk4.3<- margins(modRisk4.3)
margRisk4.3ind<- margins(modRisk4.3ind)
margRisk4.3col<- margins(modRisk4.3col)

cm<-c("risk_YieldFutureVery much"="Risk Yield: Very much","risk_YieldFutureTo some extent"="Risk Yield: To some extent","risk_BussiFutureVery much"="Risk Bussiness: Very much","risk_BussiFutureTo some extent"="Risk Bussiness: To some extent",
      "q1.5¬£10,000 to ¬£20,000"="10K to 20K","q1.5¬£20,000 to ¬£50,000"="20K to 50K","q1.5¬£50,000 to ¬£100,000"="50K to 100K", "q1.5¬£100,000 to ¬£200,000"="100K to 200K","q1.5> ¬£200,000"=">200K",
      "EvFreqMore than once a year"="Ev_More than once a year","EvFreqOnce a year"="Ev_Once a year","EvFreqOnce every 2-3 years"="Ev_every 2-3 years","EvFreqOnce in the last five years"="Ev_Once in five years",
      "woman"="woman","age235 to 44"="age: 35-44","age245 to 54"="age: 45-54","age255 to 64"="age: 55-64","age265 to 74"="age: 65-74","age2>75 years old"="age >75",
      "region2Other_regions"="region: Other","region2Scotland"="region: Scotland","region2South West"="region: SW")

riskListNbS<-list(margRisk2.2,margRisk2,margRisk2.3,margRisk4.2,margRisk4,margRisk4.3)
modelsummary(riskListNbS, stars = T, coef_map = cm,coef_omit=c(14:23),gof_omit = c("AIC|BIC|RMSE"),output = "riskNbS.png")

riskListNbSind<-list(margRisk2.2ind,margRisk2ind,margRisk2.3ind,margRisk4.2ind,margRisk4ind,margRisk4.3ind)
modelsummary(riskListNbSind, stars = T, coef_map = cm,coef_omit=c(14:23),gof_omit = c("AIC|BIC|RMSE"),output = "riskNbSind.png")

riskListNbScol<-list(margRisk2.2col,margRisk2col,margRisk2.3col,margRisk4.2col,margRisk4col,margRisk4.3col)
modelsummary(riskListNbScol, stars = T, coef_map = cm,coef_omit=c(14:23),gof_omit = c("AIC|BIC|RMSE"),output = "riskNbScol.png")


listriskgen<- list(margRisk4,margRisk4.3,margRisk4ind,margRisk4.3ind,margRisk4col,margRisk4.3col)
modelsummary(listriskgen,stars = T)

#how often experienced events 
data$EvFreq <- factor(data$EvFreq, levels = c("More than once a year","Once a year", "Once every 2-3 years", "Once in the last five years", "Never"))
data$EvFreq <- relevel(data$EvFreq, ref = levels(data$EvFreq)[5])

modEvent<- glm(NbSyes ~ EvFreq+woman+q1.2+region2, data=data, family=binomial (link = "logit"))
modEventind<- glm(NbSindiv ~ EvFreq+woman+q1.2+region2, data=data, family=binomial (link = "logit"))
modEventcol<- glm(NbScollab ~ EvFreq+woman+q1.2+region2, data=data, family=binomial (link = "logit"))

margEvent1<- margins(modEvent)
margEvent2<- margins(modEventind)
margEvent3<- margins(modEventcol)

#climate attribution
data$CCattr <- factor(data$CCattr, levels = c("Mainly", "Partly", "Not at all"))
data$CCattr <- relevel(data$CCattr, ref = levels(data$CCattr)[3])

modAtt<- glm(NbSyes ~ CCattr+woman+q1.2+region2, data=data, family=binomial (link = "logit"))
modAttind<- glm(NbSindiv ~ CCattr+woman+q1.2+region2, data=data, family=binomial (link = "logit"))
modAttcol<- glm(NbScollab ~ CCattr+woman+q1.2+region2, data=data, family=binomial (link = "logit"))

margAtt1<- margins(modAtt)
margAtt2<- margins(modAttind)
margAtt3<- margins(modAttcol)

#attribution*experienced events
modAttFreq1<- glm(NbSyes ~ CCattr+EvFreq+woman+q1.2+region2, data=data, family=binomial (link = "logit"))
modAttFreq2<- glm(NbSyes ~ CCattr*EvFreq+woman+q1.2+region2, data=data, family=binomial (link = "logit"))
modAttFreqind1<- glm(NbSindiv ~ CCattr+EvFreq+woman+q1.2+region2, data=data, family=binomial (link = "logit"))
modAttFreqind2<- glm(NbSindiv ~ CCattr*EvFreq+woman+q1.2+region2, data=data, family=binomial (link = "logit"))
modAttFreqcol1<- glm(NbScollab ~ CCattr+EvFreq+woman+q1.2+region2, data=data, family=binomial (link = "logit"))
modAttFreqcol2<- glm(NbScollab ~ CCattr*EvFreq+woman+q1.2+region2, data=data, family=binomial (link = "logit"))

margAttFreq1<- margins(modAttFreq1)
margAttFreq2<- margins(modAttFreq2)
margAttFreqind1<- margins(modAttFreqind1)
margAttFreqind2<- margins(modAttFreqind2)
margAttFreqcol1<- margins(modAttFreqcol1)
margAttFreqcol2<- margins(modAttFreqcol2)

#investigar como interpretar efectos marginales con variables interactivas-> desaparecen
modList<- list("(1)NbS"=margAtt1, "(2)NbS"=margEvent1, "(3)NbS"=margAttFreq1,"(4)NbSind"=margAtt2, "(5)NbSind"=margEvent2,"(6)NbSind"=margAttFreqind1, "(7)NbScol"=margAtt3, "(8)NbScol"=margEvent3,"(9)NbScol"=margAttFreqcol1) 
modelsummary(modList, statistic = "std.error", coef_omit = c(3:12),stars = T, output = "att.html")



#agregar income a todas
modEvent2<- glm(NbSyes ~ EvFreq+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))
modEventind2<- glm(NbSindiv ~ EvFreq+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))
modEventcol2<- glm(NbScollab ~ EvFreq+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))

margEvent1.2<- margins(modEvent2)
margEvent2.2<- margins(modEventind2)
margEvent3.2<- margins(modEventcol2)

modAtt2<- glm(NbSyes ~ CCattr+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))
modAttind2<- glm(NbSindiv ~ CCattr+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))
modAttcol2<- glm(NbScollab ~ CCattr+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))

margAtt1.2<- margins(modAtt2)
margAtt2.2<- margins(modAttind2)
margAtt3.2<- margins(modAttcol2)

#attribution*experienced events
modAttFreq1.2<- glm(NbSyes ~ CCattr+EvFreq+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))
modAttFreq2.2<- glm(NbSyes ~ CCattr*EvFreq+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))
modAttFreqind1.2<- glm(NbSindiv ~ CCattr+EvFreq+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))
modAttFreqind2.2<- glm(NbSindiv ~ CCattr*EvFreq+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))
modAttFreqcol1.2<- glm(NbScollab ~ CCattr+EvFreq+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))
modAttFreqcol2.2<- glm(NbScollab ~ CCattr*EvFreq+woman+age2+region2+q1.5, data=data, family=binomial (link = "logit"))

margAttFreq1.2<- margins(modAttFreq1.2)
margAttFreq2.2<- margins(modAttFreq2.2)
margAttFreqind1.2<- margins(modAttFreqind1.2)
margAttFreqind2.2<- margins(modAttFreqind2.2)
margAttFreqcol1.2<- margins(modAttFreqcol1.2)
margAttFreqcol2.2<- margins(modAttFreqcol2.2)

modListNbS<- list(margAtt1,margAtt1.2,margEvent1,margEvent1.2, margAttFreq1,margAttFreq1.2,margAttFreq2,margAttFreq2.2) 
modListNbSind<- list(margAtt2,margAtt2.2,margEvent2,margEvent2.2, margAttFreqind1,margAttFreqind1.2,margAttFreqind2,margAttFreqind2.2) 
modListNbScol<- list(margAtt3,margAtt3.2,margEvent3,margEvent3.2, margAttFreqcol1,margAttFreqcol1.2,margAttFreqcol2,margAttFreqcol2.2) 

cm2<-c("CCattrMainly"="CCattrib_Mainly","CCattrPartly"="CCattrib_Partly",
           "EvFreqMore than once a year"="Ev_More than once a year","EvFreqOnce a year"="Ev_Once a year","EvFreqOnce every 2-3 years"="Ev_every 2-3 years","EvFreqOnce in the last five years"="Ev_Once in five years",
      "q1.5¬£10,000 to ¬£20,000"="10K to 20K","q1.5¬£20,000 to ¬£50,000"="20K to 50K","q1.5¬£50,000 to ¬£100,000"="50K to 100K", "q1.5¬£100,000 to ¬£200,000"="100K to 200K","q1.5> ¬£200,000"=">200K",
      "woman"="woman","age2>75 years old"="age >75","age235 to 44"="age: 35-44","age245 to 54"="age: 45-54","age255 to 64"="age: 55-64","age265 to 74"="age: 65-74",
      "region2Other_regions"="region: Other","region2Scotland"="region: Scotland","region2South West"="region: SW")

modelsummary(modListNbS, statistic = "std.error",coef_map = cm2,stars = T, coef_omit = c(12:21),gof_omit = c("AIC|BIC|RMSE"), output = "attNbS.html")
modelsummary(modListNbSind, statistic = "std.error",coef_map = cm2,stars = T, coef_omit = c(12:21),gof_omit = c("AIC|BIC|RMSE"), output = "attNbSind.html")
modelsummary(modListNbScol, statistic = "std.error",coef_map = cm2,stars = T, coef_omit = c(12:21),gof_omit = c("AIC|BIC|RMSE"), output = "attNbScol.html")

#reportar solo con income
modListinc<- list("(1)NbS"=margAtt1.2, "(2)NbS"=margEvent1.2, "(3)NbS"=margAttFreq1.2,"(4)NbSind"=margAtt2.2, "(5)NbSind"=margEvent2.2,"(6)NbSind"=margAttFreqind1.2, "(7)NbScol"=margAtt3.2, "(8)NbScol"=margEvent3.2,"(9)NbScol"=margAttFreqcol1.2) 
modelsummary(modListinc, statistic = "std.error",coef_map = cm2,stars = T, coef_omit = c(7:20), output = "attInc.html")



#predicted probabilities
Y <- modAtt$y
table(true = Y, pred = round(fitted(modAtt))) 

Y2 <- modAttind$y
table(true = Y2, pred = round(fitted(modAttind))) 

Y3 <- modAttcol$y
table(true = Y3, pred = round(fitted(modAttcol))) 
##Agregar income a todos


#Income
modIncomeNbS<- glm(NbSyes ~ q1.5+woman+age2+region2, data=data, family=binomial (link = "logit"))
modIncomeNbSind<- glm(NbSindiv ~ q1.5+woman+age2+region2, data=data, family=binomial (link = "logit"))
modIncomeNbScol<- glm(NbScollab ~ q1.5+woman+age2+region2, data=data, family=binomial (link = "logit"))

margIncNbS<-margins(modIncomeNbS)
margIncNbSind<-margins(modIncomeNbSind)
margIncNbScol<-margins(modIncomeNbScol)

#income + % of oncome from activities other than farming
modIncomeNbS2<- glm(NbSyes ~ q1.9+woman+age2+region2, data=data, family=binomial (link = "logit"))
modIncomeNbSind2<- glm(NbSindiv ~ q1.9+woman+age2+region2, data=data, family=binomial (link = "logit"))
modIncomeNbScol2<- glm(NbScollab ~ q1.9+woman+age2+region2, data=data, family=binomial (link = "logit"))

margIncNbS2<-margins(modIncomeNbS2)
margIncNbSind2<-margins(modIncomeNbSind2)
margIncNbScol2<-margins(modIncomeNbScol2)


modIncomeNbS3<- glm(NbSyes ~ q1.5+q1.9+woman+age2+region2, data=data, family=binomial (link = "logit"))
modIncomeNbSind3<- glm(NbSindiv ~ q1.5+q1.9+woman+age2+region2, data=data, family=binomial (link = "logit"))
modIncomeNbScol3<- glm(NbScollab ~ q1.5+q1.9+woman+age2+region2, data=data, family=binomial (link = "logit"))

margIncNbS3<-margins(modIncomeNbS3)
margIncNbSind3<-margins(modIncomeNbSind3)
margIncNbScol3<-margins(modIncomeNbScol3)

#no salen bien los modelos
modIncomeNbS4<- glm(NbSyes ~ q1.5*q1.9+woman+age2+region2, data=data, family=binomial (link = "logit"))
modIncomeNbSind4<- glm(NbSindiv ~ q1.5*q1.9+woman+age2+region2, data=data, family=binomial (link = "logit"))
modIncomeNbScol4<- glm(NbScollab ~ q1.5*q1.9+woman+age2+region2, data=data, family=binomial (link = "logit"))

margIncNbS4<-margins(modIncomeNbS4)
margIncNbSind4<-margins(modIncomeNbSind4)
margIncNbScol4<-margins(modIncomeNbScol4)

IncListNbS<-list(margIncNbS, margIncNbS2, margIncNbS3,margIncNbS4) #no salen bien los modelos 4. quitarlos
cm3<-c("q1.5¬£10,000 to ¬£20,000"="10K to 20K","q1.5¬£20,000 to ¬£50,000"="20K to 50K","q1.5¬£50,000 to ¬£100,000"="50K to 100K", "q1.5¬£100,000 to ¬£200,000"="100K to 200K","q1.5> ¬£200,000"=">200K",
       "q1.910% to 20%"="OtherIncome:10to20%","q1.920% to 30%"="OtherIncome:20to30%","q1.930% to 40%"="OtherIncome:30to40%","q1.940% to 50%"="OtherIncome:40to50%", "q1.950% to 60%"="OtherIncome:50to60%","q1.9> 60%"="OtherIncome:>60%",
       "woman"="woman","age225 to 34"="age: 25-34","age235 to 44"="age: 35-44","age245 to 54"="age: 45-54","age255 to 64"="age: 55-64","age265 to 74"="age: 65-74","age2>75 years old"="age >75",
       "region2Other_regions"="region: Other","region2Scotland"="region: Scotland","region2South West"="region: SW")

IncListAll<-list("(1)NbS"=margIncNbS, "(2)NbS"=margIncNbS2, "(3)NbS"=margIncNbS3,"(4)NbSind"=margIncNbSind, "(5)NbSind"=margIncNbSind2, "(6)NbSind"=margIncNbSind3,"(7)NbScol"=margIncNbScol,"(8)NbScol"=margIncNbScol2,"(9)NbScol"=margIncNbScol3) 

modelsummary(IncListAll, statistic = "std.error",stars = T,gof_omit = c("AIC|BIC|RMSE"),coef_map = cm3,coef_omit = c(12:21), output = "IncNbS.html")

#nada sale significativo: controlar por funding
modIncomeNbS5<- glm(NbSyes ~ q1.5+fundingPub+fundingPriv+woman+age2+region2, data=data, family=binomial (link = "logit"))
modIncomeNbSind5<- glm(NbSindiv ~ q1.5+fundingPub+fundingPriv+woman+age2+region2, data=data, family=binomial (link = "logit"))
modIncomeNbScol5<- glm(NbScollab ~ q1.5+fundingPub+fundingPriv+woman+age2+region2, data=data, family=binomial (link = "logit"))

margIncNbS5<-margins(modIncomeNbS5)
margIncNbSind5<-margins(modIncomeNbSind5)
margIncNbScol5<-margins(modIncomeNbScol5)

modFundNbS<- glm(NbSyes ~ fundingPub+fundingPriv+woman+age2+region2, data=data, family=binomial (link = "logit"))
modFundNbSind<- glm(NbSindiv ~ fundingPub+fundingPriv+woman+age2+region2, data=data, family=binomial (link = "logit"))
modFundNbScol<- glm(NbScollab ~ fundingPub+fundingPriv+woman+age2+region2, data=data, family=binomial (link = "logit"))

margFundNbS<-margins(modFundNbS)
margFundNbSind<-margins(modFundNbSind)
margFundNbScol<-margins(modFundNbScol)

FundList<- list("(1)NbS"=margFundNbS,"(2)NbS"=margIncNbS,"(3)NbS"=margIncNbS5,"(4)NbSind"=margFundNbSind,"(5)NbSind"=margIncNbSind,"(6)NbSind"=margIncNbSind5,"(7)NbScol"=margFundNbScol,"(8)NbScol"=margIncNbScol,"(9)NbScol"=margIncNbScol5)
cm4<-c("fundingPriv"="PrivateFunding","fundingPub"="PublicFunding",
       "q1.5¬£10,000 to ¬£20,000"="10K to 20K","q1.5¬£20,000 to ¬£50,000"="20K to 50K","q1.5¬£50,000 to ¬£100,000"="50K to 100K", "q1.5¬£100,000 to ¬£200,000"="100K to 200K","q1.5> ¬£200,000"=">200K",
       "woman"="woman","age2>75 years old"="age >75","age235 to 44"="age: 35-44","age245 to 54"="age: 45-54","age255 to 64"="age: 55-64","age265 to 74"="age: 65-74",
       "region2Other_regions"="region: Other","region2Scotland"="region: Scotland","region2South West"="region: SW")

modelsummary(FundList, statistic = "std.error",stars = T,coef_map = cm4,coef_omit = c(8:16), output = "FundNbS.html")


##behaviour
#neighnours: 
data$neigh<- factor(data$neigh, levels = c("No","I don't know", "Yes"))
modNeighNbS<- glm(NbSyes ~ neigh+q1.5+woman+age2+region2, data=data, family=binomial (link = "logit"))
modNeighNbSind<- glm(NbSindiv ~ neigh+q1.5+woman+age2+region2, data=data, family=binomial (link = "logit"))
modNeighNbScol<- glm(NbScollab ~ neigh+q1.5+woman+age2+region2, data=data, family=binomial (link = "logit"))

margNeigh<-margins(modNeighNbS)
margNeighind<-margins(modNeighNbSind)
margNeighcol<-margins(modNeighNbScol)

cm5<-c("neighYes"="Neighbour_Yes","neighI don't know"="Nieghbour_I don't know",
       "q1.5¬£10,000 to ¬£20,000"="10K to 20K","q1.5¬£20,000 to ¬£50,000"="20K to 50K","q1.5¬£50,000 to ¬£100,000"="50K to 100K", "q1.5¬£100,000 to ¬£200,000"="100K to 200K","q1.5> ¬£200,000"=">200K",
      "age235 to 44"="age: 35-44","age245 to 54"="age: 45-54","age255 to 64"="age: 55-64","age265 to 74"="age: 65-74","age2>75 years old"="age >75",
      "woman"="woman",
       "region2Other_regions"="region: Other","region2Scotland"="region: Scotland","region2South West"="region: SW")
neighlist<-list("(1)NbS"=margNeigh,"(2)NbSind"=margNeighind,"(3)NbScol"=margNeighcol)
modelsummary(neighlist, stars = T,coef_map = cm5,coef_omit = c(3:17),output = "niegh.html")

#belongs to community partnership
modPartnerNbS<- glm(NbSyes ~ partner+q1.5+woman+age2+region2, data=data, family=binomial (link = "logit"))
modPartnerNbSind<- glm(NbSindiv ~ partner+q1.5+woman+age2+region2, data=data, family=binomial (link = "logit"))
modPartnerNbScol<- glm(NbScollab ~ partner+q1.5+woman+age2+region2, data=data, family=binomial (link = "logit"))

margPartner <-margins(modPartnerNbS)
margPartnerind <-margins(modPartnerNbSind)
margPartnercol <-margins(modPartnerNbScol)

cm6<-c("partnerYes"="PartnerYes",
       "q1.5¬£10,000 to ¬£20,000"="10K to 20K","q1.5¬£20,000 to ¬£50,000"="20K to 50K","q1.5¬£50,000 to ¬£100,000"="50K to 100K", "q1.5¬£100,000 to ¬£200,000"="100K to 200K","q1.5> ¬£200,000"=">200K",
       "age235 to 44"="age: 35-44","age245 to 54"="age: 45-54","age255 to 64"="age: 55-64","age265 to 74"="age: 65-74","age2>75 years old"="age >75",
       "woman"="woman",
       "region2Other_regions"="region: Other","region2Scotland"="region: Scotland","region2South West"="region: SW")
partnerList<-list("(1)NbS"=margPartner,"(2)NbSind"=margPartnerind,"(3)NbScol"=margPartnercol)
modelsummary(partnerList,stars = T,coef_map = cm6,coef_omit = c(2:16),output = "parnter.html")

#Uk government is promoting NbS
data$govPromo <- with(data, ifelse(q5.1_1=="1- Disagree", "Disagree",0))
data$govPromo <- with(data, ifelse(q5.1_1=="2- Neither agree nor disagree", "Neutral",data$govPromo))
data$govPromo <- with(data, ifelse(q5.1_1=="3 - Agree", "Agree",data$govPromo))

data$govPromo<- factor(data$govPromo, levels = c("Disagree","Neutral","Agree"))

govPromoNbS<- glm(NbSyes ~ govPromo+q1.5+woman+age2+region2, data=data, family=binomial (link = "logit"))
govPromoNbSind<- glm(NbSindiv ~ govPromo+q1.5+woman+age2+region2, data=data, family=binomial (link = "logit"))
govPromoNbScol<- glm(NbScollab ~ govPromo+q1.5+woman+age2+region2, data=data, family=binomial (link = "logit"))

margGovPromoNbS<- margins(govPromoNbS)
margGovPromoNbSind<- margins(govPromoNbSind)
margGovPromoNbScol<- margins(govPromoNbScol)

cm7<-c("govPromoAgree"="govPromoAgree","govPromoNeutral"="govPromoNeutral",
       "q1.5¬£10,000 to ¬£20,000"="10K to 20K","q1.5¬£20,000 to ¬£50,000"="20K to 50K","q1.5¬£50,000 to ¬£100,000"="50K to 100K", "q1.5¬£100,000 to ¬£200,000"="100K to 200K","q1.5> ¬£200,000"=">200K",
       "age235 to 44"="age: 35-44","age245 to 54"="age: 45-54","age255 to 64"="age: 55-64","age265 to 74"="age: 65-74","age2>75 years old"="age >75",
       "woman"="woman",
       "region2Other_regions"="region: Other","region2Scotland"="region: Scotland","region2South West"="region: SW")

promoList<-list("(1)NbS"=margGovPromoNbS,"(2)NbSind"=margGovPromoNbSind,"(3)NbScol"=margGovPromoNbScol)
modelsummary(promoList,stars = T,coef_map = cm7,coef_omit = c(3:17),output = "govPromo.html")

#NbS have higher costs than benefits
data$NbSHighCosts<-with(data, ifelse(q4.6_1=="1- Disagree","Disagree",0))
data$NbSHighCosts<-with(data, ifelse(q4.6_1=="2- Neither agree nor disagree","Neutral",data$NbSHighCosts))
data$NbSHighCosts<-with(data, ifelse(q4.6_1=="3- Agree","Agree",data$NbSHighCosts))

data$NbSHighCosts<- factor(data$NbSHighCosts, levels = c("Disagree", "Agree", "Neutral"))

modNbScosts<- glm(NbSyes ~ NbSHighCosts+q1.5+woman+age2+region2, data=data, family=binomial (link = "logit"))
modNbScostsind<- glm(NbSindiv ~ NbSHighCosts+q1.5+woman+age2+region2, data=data, family=binomial (link = "logit"))
modNbScostscol<- glm(NbScollab ~ NbSHighCosts+q1.5+woman+age2+region2, data=data, family=binomial (link = "logit"))

margCosts1 <- margins(modNbScosts)
margCosts2 <- margins(modNbScostsind)
margCosts3 <- margins(modNbScostscol)

data$NbSHighCosts2<-with(data, ifelse(NbSHighCosts=="Disagree",1,0))

modcostsNbS<- glm(NbSHighCosts2 ~ NbSyes +q1.5+woman+age2+region2, data=data, family=binomial (link = "logit"))
modcostsNbSind<- glm(NbSHighCosts2~ NbSindiv+q1.5+woman+age2+region2, data=data, family=binomial (link = "logit"))
modcostsNbScol<- glm(NbSHighCosts2~ NbScollab+q1.5+woman+age2+region2, data=data, family=binomial (link = "logit"))

margCosts1.2 <- margins(modcostsNbS)
margCosts2.2 <- margins(modcostsNbSind)
margCosts3.2 <- margins(modcostsNbScol)

costslist<-list(margCosts1.2,margCosts2.2,margCosts3.2)
modelsummary(costslist,stars = T,coef_omit = c(1:5,7:15),output = "Highcosts.html")


#una con los tres factores de behaviour
behavList<-promoList<-list("(1)NbS"=margNeigh,"(2)NbSind"=margNeighind,"(3)NbScol"=margNeighcol,"(4)NbS"=margPartner,"(5)NbSind"=margPartnerind,"(6)NbScol"=margPartnercol,"(7)NbS"=margGovPromoNbS,"(8)NbSind"=margGovPromoNbSind,"(9)NbScol"=margGovPromoNbScol )
modelsummary(behavList,stars = T,coef_omit = c(1:5,8:16),output = "behave.html")

behavList2<-promoList<-list("(1)NbS"=margNeigh,"(2)NbSind"=margNeighind,"(3)NbScol"=margNeighcol,"(4)NbS"=margPartner,"(5)NbSind"=margPartnerind,"(6)NbScol"=margPartnercol,"(7)NbS"=margGovPromoNbS,"(8)NbSind"=margGovPromoNbSind,"(9)NbScol"=margGovPromoNbScol,"(10)NbS"=margCosts1,"(11)NbSind"=margCosts2, "(12)NbScol"=margCosts3)
modelsummary(behavList2,stars = T,coef_omit = c(1:5,8:16),output = "behave2.html")


#age
##mismo modelo que income. Cambiar el orden 

cm8<-c("q1.225 to 34"="age: 25-34","q1.235 to 44"="age: 35-44","q1.245 to 54"="age: 45-54","q1.255 to 64"="age: 55-64","q1.265 to 74"="age: 65-74","q1.2>75 years old"="age >75",
"q1.5¬£10,000 to ¬£20,000"="10K to 20K","q1.5¬£20,000 to ¬£50,000"="20K to 50K","q1.5¬£50,000 to ¬£100,000"="50K to 100K", "q1.5¬£100,000 to ¬£200,000"="100K to 200K","q1.5> ¬£200,000"=">200K",
"woman"="woman","region2Other_regions"="region: Other","region2Scotland"="region: Scotland","region2South West"="region: SW")

agelist<-list(margIncNbS,margIncNbSind,margIncNbScol)
modelsummary(agelist,stars = T,coef_map = cm8,coef_omit = c(13:15),gof_omit = c("AIC|BIC|RMSE"),output = "age.html")
