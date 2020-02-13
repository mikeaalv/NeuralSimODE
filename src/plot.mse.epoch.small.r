# plot mse through epoch
# train mse based on the first value among all the mse for trianing set
# test mse based on average value
rm(list=ls())
options(warn=1)
options(stringsAsFactors=FALSE)
options(digits=15)
require(stringr)
require(magrittr)
# require(R.matlab)
require(ggplot2)
require(reshape2)
# require(xml2)
# require(cowplot)
# require(scales)
# require(lhs)
# require(foreach)
# require(doMC)
# require(readr)
# require(abind)
projectpath="./"
infortab=read.table(file=paste0(projectpath,"submitlist.tab"),sep="\t",header=TRUE)
namevec=infortab[,"names"]
dirres=paste0(projectpath,"result/")
dirlist=1:1#c(1,2,3,4,8)#1:9#1:6
mselist=list(epoch=c(),train=c(),test=c(),names=c())
for(dir in dirlist){
  locdir=paste0(dirres,dir,"/")
  files=list.files(locdir)
  filesoutput=files[str_which(string=files,pattern="testmodel\\.\\d+\\.out")]
  lines=readLines(paste0(locdir,filesoutput))
  train_ind=str_which(string=lines,pattern="Train Epoch:")
  test_ind=str_which(string=lines,pattern="Test set:")
  epoch_ind=sapply(seq(length(test_ind)),function(xi){
    if(xi>1){
      range=c(test_ind[xi-1],test_ind[xi])
    }else{
      range=c(0,test_ind[xi])
    }
    train_ind[train_ind<range[2]&train_ind>range[1]][1]
  })
  # trainrecord_ind=epoch_ind[seq(length(epoch_ind))%%2==0]
  lines[epoch_ind] %>% str_extract(string=.,pattern="Train Epoch:\\s+\\w+") %>%
              str_replace_all(string=,pattern="Train Epoch:\\s+",replacement="") %>%
              as.numeric(.) -> epoch_num
  lines[epoch_ind] %>% str_extract(string=.,pattern="Loss\\(per sample\\):\\s+[\\w\\.]+") %>%
              str_replace_all(string=,pattern="Loss\\(per sample\\):\\s+",replacement="") %>%
              as.numeric(.) -> losstrain
  lines[test_ind] %>% str_extract(string=.,pattern="Average loss \\(per sample\\):\\s+[\\w\\.]+") %>%
              str_replace_all(string=,pattern="Average loss \\(per sample\\):\\s+",replacement="") %>%
              as.numeric(.) -> losstest
  mselist[["epoch"]]=c(mselist[["epoch"]],epoch_num)
  mselist[["train"]]=c(mselist[["train"]],losstrain)
  mselist[["test"]]=c(mselist[["test"]],losstest)
  mselist[["names"]]=c(mselist[["names"]],rep(namevec[dir],times=length(epoch_num)))
}
msetab=as.data.frame(mselist)
colnames(msetab)=c("epoch","train","test","names")
msetablong=melt(msetab,id=c("epoch","names"))
msetablong[,"names"]=as.factor(msetablong[,"names"])
# msetablong[,"value"]=log10(msetablong[,"value"])
p<-ggplot(data=msetablong,aes(epoch,value,linetype=variable,colour=names))+
      geom_line(alpha=0.5)+
      scale_y_continuous(trans='log10',limits=c(0.01,max(msetablong[,"value"])))+
      xlab("epoch")+
      ylab("mse")+
      theme_bw()
ggsave(plot=p,file=paste0(dirres,"mse_epoch.pdf"))
##end state(epoch) statistics on mse
endwid=10
names=unique(msetab[,"names"])
summtab=as.data.frame(matrix(NA,nrow=length(names),ncol=3))
colnames(summtab)=c("names","train_mean","test_mean")
rownames(summtab)=names
for(namegroup in names){
  subtab=msetab[msetab[,"names"]==namegroup,]
  endepoch=max(subtab[,"epoch"])
  subtab2=subtab[subtab[,"epoch"]>=(endepoch-endwid),]
  summtab[namegroup,"names"]=namegroup
  summtab[namegroup,"train_mean"]=mean(subtab2[,"train"])
  summtab[namegroup,"test_mean"]=mean(subtab2[,"test"])
}
save(summtab,msetablong,p,file=paste0(dirres,"Rplot_store.RData"))
