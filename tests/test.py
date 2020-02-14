import unittest
import os, shutil
import subprocess
import rpy2.robjects as robjects
import glob
import pyreadr
import torch

test_input="./test_data/"
test_output="./output/"
sourcodedir="../src/"
test_temprun=test_output+"temprun/"##folder for the run of neuralnet
rundatadir=test_temprun+"data/"
runcodedir=test_temprun+"code/"
projdir=test_output+"project/"##the folder supposed to be on local computer and do plotting/analyzing related script
projresdir=projdir+"result/"
projresdir_1=projresdir+"1/"
projcodeeir=projdir+"code/"
codefilelist=['mlp_struc.py','plot_model_small.py','plot.mse.epoch.small.r','train_mlp_full_modified.py']
runinputlist=['sparselinearode_new.small.stepwiseadd.mat']
runoutputlist=['pickle_traindata.dat','pickle_testdata.dat','pickle_inputwrap.dat','pickle_dimdata.dat','model_best.resnetode.tar','model_best_train.resnetode.tar','checkpoint.resnetode.tar','testmodel.1.out']
runcodelist=['train_mlp_full_modified.py','mlp_struc.py']
runcodetest=['test.sh']
plotdata_py=['plotsave.dat']
plotdata_r=['Rplot_store.RData']
tempdata_py=['datatemp.dat']
smalval=0.001##for comparing values in such as mse
class NNTODETest(unittest.TestCase):
    def test_pre(self):
        '''
        directory preparation for the test
        '''
        try:
            os.mkdir(test_temprun)
            os.mkdir(rundatadir)
            os.mkdir(runcodedir)
            os.mkdir(projdir)
            os.mkdir(projresdir)
            os.mkdir(projresdir_1)
            os.mkdir(projcodeeir)
            shutil.copyfile(test_input+runinputlist,rundatadir+runinputlist)
            for codefile in runcodelist:
                shutil.copyfile(sourcodedir+codefile,runcodedir+codefile)
            shutil.copyfile(runcodetest,runcodedir+runcodetest)
            for codefile in codefilelist:
                shutil.copyfile(sourcodedir+codefile,projcodeeir+codefile)
        except:
            self.assertTrue(False)
        
        self.assertTrue(True)
        
    def test_run_train(self):
        '''
        test run of the training process
        '''
        try:
            os.chdir(runcodedir)
            with open (runcodetest,"r") as myfile:
                command=myfile.readlines()
            os.system(command)
            for outputfile in runoutputlist:
                shutil.copyfile(runcodedir+outputfile,projresdir_1+outputfile)
        except:
            self.assertTrue(False)
        
        self.assertTrue(True)
    
    def test_run_plotting(self):
        '''
        run the plotting related python and R script
        '''
        try:
            os.chdir(projcodeeir)
            import plot_model_small
            subprocess.call("Rscript --vanilla plot.mse.epoch.small.r", shell=True)
        except:
            self.assertTrue(False)
        
        self.assertTrue(True)
        
    def test_file_exist(self):
        '''
        test all output file are in the folder
        '''
        try:
            currlist=set([f for f in os.listdir('.') if re.search(r'.*\.(pdf|tar|dat|out)$',f)])
            storelist=set([f for f in os.listdir(test_input+'./pdf') if re.search(r'.*\.(pdf|tar|dat|out)$',f)])
            if currlist==storelist:
                self.assertTrue(True)
            else:
                self.assertTrue(False)
        except:
            self.assertTrue(False)
    
    def test_plot_model_small(self):
        '''
        test dimension&value of output files in all script
        '''
        try:
            ##figure the same
            with open(plotdata_py,"rb") as f1:
                newfig=pickle.load(f1)
            with open(test_input+plotdata_py,"rb") as f1:
                oldfig=pickle.load(f1)
            figequal=newfig.__eq__(oldfig)
            ##data size the same {temporary data at one time point stored}
            with open(tempdata_py,"rb") as f1:
                newdata=pickle.load(f1)
            with open(test_input+tempdata_py,"rb") as f1:
                olddata=pickle.load(f1)
            dataequal=newdata['data'].shape==olddata['data'].shape
            outputequal=newdata['output'].shape==olddata['output'].shape
            targetequal=newdata['target'].shape==olddata['target'].shape
            ninnersize_val_equal=newdata['ninnersize']==olddata['ninnersize']
            target_val_equal=newdata['target']==olddata['target']
            print("data_size %s output_size %s target_size %s ninnersize %s\n" % (newdata['data'].shape,newdata['output'].shape,newdata['target'].shape,newdata['ninnersize'],))
            if figequal and dataequal and outputequal and targetequal and ninnersize_val_equal and target_val_equal:
                self.assertTrue(True)
            else:
                self.assertTrue(False)
        except:
            self.assertTrue(False)
    
    def test_plot_mse_epoch_small(self):
        '''
        test plot&dimension of data
        '''
        try:
            newres=pyreadr.read_r(plotdata_r)
            oldres=pyreadr.read_r(test_input+plotdata_r)
            figequal=newres['p']==oldres['p']
            tabdimequal=(newres['summtab'].shape==oldres['summtab'].shape and newres['msetablong'].shape==oldres['msetablong'].shape)
            print("summtab_size %s msetablong_size %s\n" % (newres['summtab'].shape,newres['msetablong'].shape,))
            if figequal and tabdimequal:
                self.assertTrue(True)
            else:
                self.assertTrue(False)
        except:
            self.assertTrue(False)
    
    def test_train_mlp_full_modified(self):
        '''
        test value and dimension of the training script on a small run
        '''
        try:
            #dimension of output and input
            with open(runoutputlist[2],"rb") as f1:
                currstore=pickle.load(f1)
            with open(test_input+runoutputlist[2],"rb") as f1:
                prestore=pickle.load(f1)
            print("Xvarnorm_size %s ResponseVar_size %s\n" % (currstore['Xvarnorm'].shape,currstore['ResponseVar'].shape,))
            if currstore['Xvarnorm'].shape==prestore['Xvarnorm'].shape and currstore['ResponseVar'].shape==prestore['ResponseVar'].shape:
                dimequal=True
            else:
                dimequal=False
            #value of stored data
            inputwrap_true=currstore==prestore
            with open(runoutputlist[3],"rb") as f1:
                currstore=pickle.load(f1)
            with open(test_input+runoutputlist[3],"rb") as f1:
                prestore=pickle.load(f1)
            dimdict_true=currstore==prestore
            device=torch.device('cpu')
            currstore=torch.load(runoutputlist[6],map_location=device)
            prestore=torch.load(test_input+runoutputlist[6],map_location=device)
            arg_equal=currstore['args_input']==prestore['args_input']
            arch_equal=currstore['arch']==prestore['arch']
            epoch_equal=currstore['epoch']==prestore['epoch']
            if inputwrap_true and dimdict_true and arg_equal and arch_equal and epoch_equal:
                valequal=True
            else:
                valequal=False
            if (currstore['best_acc1']-prestore['best_acc1'])<smalval and (currstore['best_acctr']-prestore['best_acctr'])<smalval:
                perf_equal=True
            else:
                perf_equal=False
            curr_state_dict=currstore['state_dict']
            pre_state_dict=prestore['state_dict']
            layer_size_equal=True
            for layer in curr_state_dict.keys():
                layer_size_equal=layer_size_equal and (curr_state_dict[layer].shape==pre_state_dict[layer].shape)
            if dimequal and layer_size_equal and valequal and perf_equal:
                self.assertTrue(True)
            else:
                self.assertTrue(False)
        except:
            self.assertTrue(False)
        
    def test_sampler_function(self):
        try:
            random.seed(1)
            datasource=np.array([0,1,2,3,4,5,6,7,8,9])
            blocks=np.array([0,0,1,1,2,2,3,3,4,4])
            nblocks=2
            exp_res=[[2,3,4,5],[6,7,8,9],[0,1]]
            test_res=list(batch_sampler_block(datasource,blocks,nblock=nblocks))
            if test_res==exp_res:
                self.assertTrue(True)
            else:
                self.assertTrue(False)
        except:
            self.assertTrue(False)
    
    def test_clean(self):
        for filename in os.listdir(test_output):
            file_path=os.path.join(test_output,filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path,e))
                self.assertTrue(False)
        self.assertTrue(True)
    
if __name__ == '__main__':
    unittest.main()
        
