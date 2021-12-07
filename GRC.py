# coding=UTF-8
import scipy.optimize as opt
from  scipy.special import gamma
from scipy.stats import t,invgamma
import numpy
import math
import scipy.integrate as ipt
from statsmodels.api import OLS,add_constant
import random
import time
from random import uniform
from matplotlib.pyplot import plot,show
from csv import reader
from sys import exc_info
initial_value=[0,math.pi/4]#最老版本的迭代初值

def random_multivar(select_list):
    selected_list=[select_list.copy().index(i) for i in select_list.copy()]
    a=min(selected_list)
    b=max(selected_list)
    random_ab=random.uniform(a,b)
    cal_list=[abs(i-random_ab) for i in selected_list]
    return select_list[cal_list.index(min(cal_list))]

class GRC(object):
    def __init__(self,y_datas,coviate_datas,identify_var_index,classify_var_index,set_control_group=0,constant=True,estimated_var_nums=20):
        self.classify_var_index=classify_var_index
        self.coviate_datas=coviate_datas
        self.list_classifyvar_list=[]
        self.data = y_datas
        self.identify_var_index=identify_var_index
        self.constant=constant
        self.estimated_var_nums=estimated_var_nums
        for i in range(0,self.classify_var_index):
            temp_list=[]
            for k in range(0,len(self.data)):
                if self.coviate_datas[len(self.coviate_datas)-self.classify_var_index+i][k] in temp_list:
                    pass
                else:
                    temp_list.append(self.coviate_datas[len(self.coviate_datas)-self.classify_var_index+i][k])
            self.list_classifyvar_list.append(temp_list)
        self.index_identiyvar_list=[]
        self.classifyname_index=[]
        start_nums=[0 for i in range(0,len(self.list_classifyvar_list))]
        while start_nums[0]<=len(self.list_classifyvar_list[0])-1:
            change_signal=len(self.list_classifyvar_list)-1
            to_list=[]
            for i in range(0,len(start_nums)):
                to_list.append(self.list_classifyvar_list[i][start_nums[i]])
            self.classifyname_index.append(to_list)
            if start_nums == [len(i) - 1 for i in self.list_classifyvar_list]:
                break
            else:
                pass
            start_nums[len(self.list_classifyvar_list)-1]+=1
            while start_nums[change_signal] == len(self.list_classifyvar_list[change_signal]):
                start_nums[change_signal - 1] += 1
                start_nums[change_signal] = 0
                change_signal -= 1

        #计算给定的各个状态变量细分种类的个数
        #生成新的解释变量矩阵，除了非状态变量直接照搬之外，还加入了新的多元同时成立的虚拟变量
        self.new_xlines = self.coviate_datas.copy()
        #得到总的需要新生成的虚拟变量的个数，并且将空的列表插入新的解释变量矩阵
        nums_newadd_var=1
        for index_classifyvar_in_covvars in self.list_classifyvar_list:
            nums_newadd_var=nums_newadd_var*len(index_classifyvar_in_covvars)
        self.num_virtual_var=nums_newadd_var
        self.each_y=[]
        for i in range(0,nums_newadd_var):
            self.each_y.append([])
        for index_every_y in range(0, len(self.data)):
            temp_list_to_save_everyindex_in_classifyvar = []  # 储存每个状态变量下的该行联合虚拟变量对应的下标
            for index_to_panelvar in range(0, self.classify_var_index):
                temp_list_to_save_everyindex_in_classifyvar.append(
                    self.list_classifyvar_list[index_to_panelvar].index(
                        self.coviate_datas[len(self.coviate_datas) - self.classify_var_index + index_to_panelvar][
                            index_every_y]))
            get_index_every_y_in_line = 0
            for index_everyu_po in range(0, len(temp_list_to_save_everyindex_in_classifyvar) - 1):
                temp_value = 1
                for index_every_po_next in range(index_everyu_po + 1,
                                                 len(temp_list_to_save_everyindex_in_classifyvar)):
                    temp_value = temp_value * (len(self.list_classifyvar_list[index_every_po_next]))
                get_index_every_y_in_line += temp_list_to_save_everyindex_in_classifyvar[
                                                 index_everyu_po] * temp_value
            get_index_every_y_in_line += temp_list_to_save_everyindex_in_classifyvar[
                                             len(temp_list_to_save_everyindex_in_classifyvar) - 1]
            self.each_y[get_index_every_y_in_line].append([self.data[index_every_y],index_every_y])
        if constant==True:
            self.coviate_datas_t = numpy.array(self.coviate_datas[:len(self.coviate_datas)-self.classify_var_index]).T
            self.ols_result_origin= OLS(self.data, add_constant(self.coviate_datas_t)).fit()
            self.new_yline = []
            self.new_xlines=self.coviate_datas[:len(self.coviate_datas)-self.classify_var_index]
            self.new_each_y=self.each_y.copy()
            self.each_means=[]
            self.each_means_sigma=[]
            for index_x in range(0,len(self.each_y)):
                temp_yline=[]
                temp_xline=[]
                for index_y in range(0,len(self.each_y[index_x])):
                    constant_linear=0
                    for index_in_controlvar in range(0,len(self.coviate_datas)-self.classify_var_index-identify_var_index):
                        constant_linear+=self.coviate_datas[identify_var_index+index_in_controlvar][self.each_y[index_x][index_y][1]]*self.ols_result_origin.params[1+identify_var_index+index_in_controlvar]
                    unconstant_linear=self.ols_result_origin.params[0]
                    for index_in_uncontrolvar in range(0,identify_var_index):
                        unconstant_linear+=self.coviate_datas[index_in_uncontrolvar][self.each_y[index_x][index_y][1]]*self.ols_result_origin.params[1+index_in_uncontrolvar]

                    temp_yline.append(self.each_y[index_x][index_y][0]-constant_linear)
                    temp_xline.append(unconstant_linear)
                #self.each_means.append(numpy.mean(templist))
                ok_temp_olsresult=OLS(temp_yline,temp_xline).fit()
                self.each_means.append(ok_temp_olsresult.params[0])
                self.each_means_sigma.append(ok_temp_olsresult.HC3_se[0])

            self.each_xlines = [[] for i in range(0, len(self.each_means))]
            for i in range(0, len(self.each_means)):
                for j in range(0, len(self.coviate_datas) - self.classify_var_index+1):
                    self.each_xlines[i].append([])
                self.each_xlines[i].append([])
            for i in range(0, len(self.each_means)):
                self.each_xlines[i][0] = list(numpy.ones(len(self.each_y[i])))
                for index_x in range(0, len(self.each_y[i])):
                    for index_each_covx in range(1, len(self.coviate_datas) - self.classify_var_index+1):
                        self.each_xlines[i][index_each_covx].append(self.coviate_datas[index_each_covx-1][
                            self.each_y[i][index_x][1]])
                    self.each_xlines[i][len(self.coviate_datas) - self.classify_var_index+1].append(int(i))
            self.alphas=[[self.classifyname_index[i],self.each_means[i],self.each_means_sigma[i],len(self.each_y[i])-1] for i in range(0,len(self.each_means))]
            self.betas=[[] for i in range(self.classify_var_index)]
            self.citas = [[] for i in range(self.classify_var_index)]
            self.sigma_of_betas=[[] for i in range(self.classify_var_index)]
            for i in range(0,len(self.betas)):
                for j in range(0,len(self.list_classifyvar_list[i])-1):
                    self.betas[i].append(None)
                    self.citas[i].append(None)
                    self.sigma_of_betas[i].append(None)
            self.classify_varlists=[[] for i in range(self.classify_var_index)]
            self.classify_namelists=[[] for i in range(self.classify_var_index)]
            self.estimated_classify_varlist=[[] for i in range(self.classify_var_index)]
            self.MC_params_density()
            for i in range(0,len(self.classify_varlists)):
                for j in range(0,len(self.list_classifyvar_list[i])):
                    self.classify_varlists[i].append([])
                    self.classify_namelists[i].append(None)
                    self.estimated_classify_varlist[i].append([])
            for i in range(0,len(self.alphas)):
                for index_x in range(0,self.classify_var_index):
                    index_left=index_x-1 if index_x!=0 else -1
                    index_list=self.alphas[i][0][0:index_left+1] if index_x!=0 else []
                    for index_each_value in self.alphas[i][0][index_left+2:]:
                        index_list.append(index_each_value)
                    self.classify_varlists[index_x][self.list_classifyvar_list[index_x].index(self.alphas[i][0][index_x])].append([index_list,self.alphas[i][1],self.alphas[i][3]])
                    self.classify_namelists[index_x][self.list_classifyvar_list[index_x].index(self.alphas[i][0][index_x])]=self.alphas[i]
                    self.estimated_classify_varlist[index_x][self.list_classifyvar_list[index_x].index(self.alphas[i][0][index_x])].append([index_list,self.estimated_lambdalist_mean[i],self.estimated_lambdalist_std[i]])
            for index_each_classifyname in range(0,len(self.classify_varlists)):
                for index_each_value in range(1,len(self.classify_varlists[index_each_classifyname])):
                    temp_group1 = []
                    list_pdf=[]
                    temp_group2=[]
                    for  i in range(0,len(self.classify_varlists[index_each_classifyname][index_each_value])):
                        temp_group1.append(self.classify_varlists[index_each_classifyname][index_each_value][i][1]/self.classify_varlists[index_each_classifyname][0][i][1])
                        temp_group2.append(self.estimated_classify_varlist[index_each_classifyname][index_each_value][i][1]/self.estimated_classify_varlist[index_each_classifyname][0][i][1])
                        list_pdf.append(1/len(self.classify_varlists[index_each_classifyname][index_each_value]))
                    self.betas[index_each_classifyname][index_each_value-1]=[numpy.dot(numpy.array(temp_group1),numpy.array(list_pdf)),self.classify_namelists[index_each_classifyname][index_each_value][0][index_each_classifyname],self.classify_namelists[index_each_classifyname][0][0][index_each_classifyname]]
                    self.citas[index_each_classifyname][index_each_value-1]=[numpy.dot(numpy.array(temp_group2),numpy.array(list_pdf)),self.classify_namelists[index_each_classifyname][index_each_value][0][index_each_classifyname],self.classify_namelists[index_each_classifyname][0][0][index_each_classifyname]]#[numpy.dot(numpy.array(temp_group2),numpy.array(list_pdf)),

            print("sdfsdf")
        else:
            self.coviate_datas_t = numpy.array(self.coviate_datas[:len(self.coviate_datas) - self.classify_var_index]).T
            self.ols_result_origin = OLS(self.data, add_constant(self.coviate_datas_t)).fit()
            self.new_yline = []
            self.new_xlines = self.coviate_datas[:len(self.coviate_datas) - self.classify_var_index]
            self.new_each_y = self.each_y.copy()
            self.each_means = []
            self.each_means_sigma = []
            for index_x in range(0, len(self.each_y)):
                temp_yline = []
                temp_xline = []
                for index_y in range(0, len(self.each_y[index_x])):
                    constant_linear = 0
                    for index_in_controlvar in range(0, len(
                            self.coviate_datas) - self.classify_var_index - identify_var_index):
                        constant_linear += self.coviate_datas[identify_var_index + index_in_controlvar][
                                               self.each_y[index_x][index_y][1]] * self.ols_result_origin.params[
                                               1 + identify_var_index + index_in_controlvar]
                    unconstant_linear = self.ols_result_origin.params[0]
                    for index_in_uncontrolvar in range(0, identify_var_index):
                        unconstant_linear += self.coviate_datas[index_in_uncontrolvar][
                                                 self.each_y[index_x][index_y][1]] * self.ols_result_origin.params[
                                                 1 + index_in_uncontrolvar]

                    temp_yline.append(self.each_y[index_x][index_y][0] - constant_linear)
                    temp_xline.append(unconstant_linear)
                # self.each_means.append(numpy.mean(templist))
                ok_temp_olsresult = OLS(temp_yline, temp_xline).fit()
                self.each_means.append(ok_temp_olsresult.params[0])
                self.each_means_sigma.append(ok_temp_olsresult.HC3_se[0])

            self.each_xlines = [[] for i in range(0, len(self.each_means))]
            for i in range(0, len(self.each_means)):
                for j in range(0, len(self.coviate_datas) - self.classify_var_index + 1):
                    self.each_xlines[i].append([])
                self.each_xlines[i].append([])
            for i in range(0, len(self.each_means)):
                self.each_xlines[i][0] = list(numpy.ones(len(self.each_y[i])))
                for index_x in range(0, len(self.each_y[i])):
                    for index_each_covx in range(1, len(self.coviate_datas) - self.classify_var_index + 1):
                        self.each_xlines[i][index_each_covx].append(self.coviate_datas[index_each_covx - 1][
                                                                        self.each_y[i][index_x][1]])
                    self.each_xlines[i][len(self.coviate_datas) - self.classify_var_index + 1].append(int(i))
            self.alphas = [
                [self.classifyname_index[i], self.each_means[i], self.each_means_sigma[i], len(self.each_y[i]) - 1] for
                i in range(0, len(self.each_means))]
            self.betas = [[] for i in range(self.classify_var_index)]
            self.sigma_of_betas = [[] for i in range(self.classify_var_index)]
            for i in range(0, len(self.betas)):
                for j in range(0, len(self.list_classifyvar_list[i]) - 1):
                    self.betas[i].append(None)
                    self.sigma_of_betas[i].append(None)
            self.classify_varlists = [[] for i in range(self.classify_var_index)]
            for i in range(0, len(self.classify_varlists)):
                for j in range(0, len(self.list_classifyvar_list[i])):
                    self.classify_varlists[i].append([])
            for i in self.alphas:
                for index_x in range(0, self.classify_var_index):
                    index_left = index_x - 1 if index_x != 0 else -1
                    index_list = i[0][0:index_left + 1] if index_x != 0 else []
                    for index_each_value in i[0][index_left + 2:]:
                        index_list.append(index_each_value)
                    self.classify_varlists[index_x][self.list_classifyvar_list[index_x].index(i[0][index_x])].append(
                        [index_list, i[1], i[3]])
            self.alphas_estimated=[[self.alphas[i][0],self.estimated_lambdalist_mean[i]] for i in range(0,len(self.alphas))]
            for index_each_classifyname in range(0, len(self.classify_varlists)):
                for index_each_value in range(1, len(self.classify_varlists[index_each_classifyname])):
                    temp_group1 = []
                    list_pdf = []
                    for i in range(0, len(self.classify_varlists[index_each_classifyname][index_each_value])):
                        a = gamma(self.classify_varlists[index_each_classifyname][index_each_value][i][2] / 2 + 1 / 2)
                        b = gamma(self.classify_varlists[index_each_classifyname][index_each_value][i][2] / 2)
                        if a == float('inf') and b == float('inf'):
                            g_n = 1 / math.sqrt(
                                self.classify_varlists[index_each_classifyname][index_each_value][i][2] * math.pi)
                        else:
                            g_n = a / (b * math.sqrt(
                                self.classify_varlists[index_each_classifyname][index_each_value][i][2] * math.pi))
                        temp_group1.append(self.classify_varlists[index_each_classifyname][index_each_value][i][1] /
                                           self.classify_varlists[index_each_classifyname][0][i][1])
                        list_pdf.append(g_n / len(self.classify_varlists[index_each_classifyname][index_each_value]))
                    sum_pdf = sum(list_pdf)
                    list_pdf = [i / sum_pdf for i in list_pdf]
                    self.betas[index_each_classifyname][index_each_value - 1] = numpy.dot(numpy.array(temp_group1),
                                                                                          numpy.array(list_pdf))
                    self.sigma_of_betas[index_each_classifyname][index_each_value - 1] = math.sqrt(numpy.dot(
                        numpy.array([(i - self.betas[index_each_classifyname][index_each_value - 1]) ** 2 for i in
                                     temp_group1]), numpy.array(list_pdf) * len(temp_group1) / (len(temp_group1) - 1)))
    def P_posterior_thetas(self,whichone,linear_varlist,lambda_varlist,sigma):
        if self.each_y == [] or self.each_y == None:
            raise ValueError("The (in)dependant data should be classify first")
        else:
            lambda_pi_line=[]
            resid_except_whichone=[]
            for index_each_ylines in range(0, len(self.each_y)):
                for index_each_x in range(0, len(self.each_y[index_each_ylines])):
                    lambda_pi_line.append(lambda_varlist[self.each_xlines[index_each_ylines][6][index_each_x]] *
                                          self.each_xlines[index_each_ylines][
                                              whichone ][index_each_x])
                    notvaryingcoff_linearvalue=0
                    for i in range(self.identify_var_index+1 if self.constant==True else self.identify_var_index,len(self.each_xlines[index_each_ylines])-1):
                        notvaryingcoff_linearvalue+=self.each_xlines[index_each_ylines][i][index_each_x]*linear_varlist[i]
                    varyingcoff_linearvalue=0
                    for i in range(0,self.identify_var_index+1 if self.constant==True else self.identify_var_index):
                        if i==whichone:
                            pass
                        else:
                            varyingcoff_linearvalue+=linear_varlist[i]*self.each_xlines[index_each_ylines][i][index_each_x]*lambda_varlist[int(self.each_xlines[index_each_ylines][6][index_each_x])]
                    resid_except_whichone.append(self.each_y[index_each_ylines][index_each_x][0]-notvaryingcoff_linearvalue-varyingcoff_linearvalue)

            estimated_std=math.sqrt(sigma/numpy.dot(
                numpy.array(lambda_pi_line), numpy.array(lambda_pi_line)))
            estimated_mu =numpy.dot(numpy.array(resid_except_whichone).T, numpy.array(lambda_pi_line)) / numpy.dot(
                numpy.array(lambda_pi_line), numpy.array(lambda_pi_line))
            return random.normalvariate(estimated_mu,estimated_std)
    def P_posterior_betas(self,whichone,linear_varlist,lambda_varlist,sigma):
        if self.each_y == [] or self.each_y == None:
            raise ValueError("The (in)dependant data should be classify first")
        else:
            xi_line = []
            resid_except_whichone = []

            for index_each_ylines in range(0, len(self.each_y)):
                for index_each_x in range(0, len(self.each_y[index_each_ylines])):

                    notvaryingcoff_linearvalue = 0
                    for i in range(self.identify_var_index + 1 if self.constant == True else self.identify_var_index,
                                   len(self.each_xlines[index_each_ylines]) - 1):
                        if whichone+self.identify_var_index+1==i and self.constant==True:
                            xi_line.append(self.each_xlines[index_each_ylines][i][index_each_x])
                        elif whichone+self.identify_var_index==i and self.constant==False:
                            xi_line.append(self.each_xlines[index_each_ylines][i][index_each_x])
                        else:
                            notvaryingcoff_linearvalue += self.each_xlines[index_each_ylines][i][index_each_x] * \
                                                      linear_varlist[i]
                    varyingcoff_linearvalue = 0
                    for i in range(0,
                                   self.identify_var_index + 1 if self.constant == True else self.identify_var_index):

                        varyingcoff_linearvalue += linear_varlist[i] * self.each_xlines[index_each_ylines][i][
                                index_each_x] * lambda_varlist[
                                                           int(self.each_xlines[index_each_ylines][6][index_each_x])]
                    resid_except_whichone.append(self.each_y[index_each_ylines][index_each_x][
                                                     0] - notvaryingcoff_linearvalue - varyingcoff_linearvalue)



            estimated_std = math.sqrt(sigma/numpy.dot(
                numpy.array(xi_line), numpy.array(xi_line)))
            estimated_mu =  numpy.dot(numpy.array(resid_except_whichone), numpy.array(xi_line)) / numpy.dot(
                numpy.array(xi_line), numpy.array(xi_line))
            return random.normalvariate(estimated_mu, estimated_std)

    def P_posterior_lambdas(self,whichone,linear_varlist,lambda_varlist,sigma):
        if self.each_y == [] or self.each_y == None:
            raise ValueError("The (in)dependant data should be classify first")
        else:
            Dipi_line = []
            resid_except_whichone = []
            for index_each_ylines in range(0, len(self.each_y)):
                for index_each_x in range(0, len(self.each_y[index_each_ylines])):
                    notvaryingcoff_linearvalue = 0
                    for i in range(self.identify_var_index + 1 if self.constant == True else self.identify_var_index,
                                   len(self.each_xlines[index_each_ylines]) - 1):

                        notvaryingcoff_linearvalue += self.each_xlines[index_each_ylines][i][index_each_x] * \
                                                      linear_varlist[i]
                    varyingcoff_linearvalue = 0
                    for i in range(0,
                                   self.identify_var_index + 1 if self.constant == True else self.identify_var_index):

                        varyingcoff_linearvalue += linear_varlist[i] * self.each_xlines[index_each_ylines][i][
                                index_each_x]
                    if whichone==int(self.each_xlines[index_each_ylines][6][index_each_x]):
                        resid_except_whichone.append(self.each_y[index_each_ylines][index_each_x][
                                                     0] - notvaryingcoff_linearvalue )
                        Dipi_line.append(varyingcoff_linearvalue)
                    else:
                        pass
            estimated_mu = numpy.dot(numpy.array(resid_except_whichone), numpy.array(Dipi_line)) / numpy.dot(
                numpy.array(Dipi_line), numpy.array(Dipi_line))
            estimated_std = math.sqrt(sigma/numpy.dot(
                numpy.array(Dipi_line), numpy.array(Dipi_line)))
            return random.normalvariate(estimated_mu, estimated_std)
    def P_posterior_sigma(self,linear_varlist,lambda_varlist):
        if self.each_y == [] or self.each_y == None:
            raise ValueError("The (in)dependant data should be classify first")
        else:
            resid = []
            for index_each_ylines in range(0, len(self.each_y)):
                for index_each_x in range(0, len(self.each_y[index_each_ylines])):

                    notvaryingcoff_linearvalue = 0
                    for i in range(self.identify_var_index + 1 if self.constant == True else self.identify_var_index,
                                   len(self.each_xlines[index_each_ylines]) - 1):

                        notvaryingcoff_linearvalue += self.each_xlines[index_each_ylines][i][index_each_x] * \
                                                          linear_varlist[i]
                    varyingcoff_linearvalue = 0
                    for i in range(0,
                                   self.identify_var_index + 1 if self.constant == True else self.identify_var_index):
                        varyingcoff_linearvalue += linear_varlist[i] * self.each_xlines[index_each_ylines][i][
                            index_each_x] * lambda_varlist[
                                                       int(self.each_xlines[index_each_ylines][6][index_each_x])]
                    resid.append(self.each_y[index_each_ylines][index_each_x][
                                                     0] - notvaryingcoff_linearvalue - varyingcoff_linearvalue)
            start_points=0
            estimated_var_list=[]
            if int(len(resid)/self.estimated_var_nums)<20:
                print("Beacuse there are few samples for estiated inverse gamma parameters,it will use boostrap metohd automatially.")
                #Boostarp method to estimated parameters for inverse gamma distribution
            else:
                while start_points+self.estimated_var_nums<len(resid):
                    estimated_var_list.append(sum([i**2 for i in resid[start_points:start_points+self.estimated_var_nums]])/(self.estimated_var_nums-1))
                    start_points+=self.estimated_var_nums
                ave_var=numpy.mean(estimated_var_list)
                std_var=numpy.std(estimated_var_list)
            alpha=(ave_var**2)/(std_var**2)+2
            beta=(ave_var**3)/(std_var**2)+ave_var
            gen_sigma=1/random.gammavariate(alpha=alpha,beta=beta)
            print("Generate a sigma:",gen_sigma)
            return gen_sigma#sum([i**2 for i in resid])/(len(self.data)-1)

    def MC_params_density(self,parallelchain_nums=10,cut_point = 80,sample_size=150):
        print("Begin")
        linear_varlist = self.ols_result_origin.params.copy()
        lambda_varlist = self.each_means.copy()
        sigma = sum([i ** 2 for i in self.ols_result_origin.resid]) / self.ols_result_origin.df_resid
        estimated_linearvarlist = [[i] for i in linear_varlist.copy()]
        estimated_lambda = [[i] for i in lambda_varlist.copy()]
        estimated_sigma=[sigma]
        while min([min([len(i) for i in estimated_linearvarlist]),min([len(i) for i in estimated_lambda]),len(estimated_sigma)])<=cut_point+sample_size:
            #linearvalue varibales's postrior distribution
            print('index',min([min([len(i) for i in estimated_linearvarlist]),min([len(i) for i in estimated_lambda])]))
            for i in range(0,len(linear_varlist)):
                if i>=self.identify_var_index+1 and self.constant==True:
                    x_linear=self.P_posterior_betas(i-(self.identify_var_index+1),linear_varlist,lambda_varlist,sigma)
                    print('Lineanr',i,'||',linear_varlist[i],"->",x_linear)
                    linear_varlist[i]=x_linear
                    estimated_linearvarlist[i].append(x_linear)
                elif  i>=self.identify_var_index and self.constant==False:
                    x_linear = self.P_posterior_betas(i - (self.identify_var_index), linear_varlist, lambda_varlist,
                                                      sigma)
                    print('Lineanr',i,'||',linear_varlist[i], "->", x_linear)
                    linear_varlist[i] = x_linear
                    estimated_linearvarlist[i].append(x_linear)
                else:
                    x_linear = self.P_posterior_thetas(i, linear_varlist, lambda_varlist,
                                                      sigma)
                    print('Lineanr',i,'||',linear_varlist[i], "->", x_linear)
                    linear_varlist[i] = x_linear
                    estimated_linearvarlist[i].append(x_linear)
            #lambda variables 's postrior distribution function
            print("Linear part Finished,--->>>>>Lambda part")
            for i in range(0,len(lambda_varlist)):
                x_lambda=self.P_posterior_lambdas(i,linear_varlist,lambda_varlist,sigma)
                print('Lambda',i,'||',lambda_varlist[i], "->", x_lambda)
                lambda_varlist[i]=x_lambda
                estimated_lambda[i].append(x_lambda)
            sigma=self.P_posterior_sigma(linear_varlist,lambda_varlist)
            estimated_sigma.append(sigma)

        self.estimated_linearvarlist_mean=[numpy.mean(estimated_linearvarlist[i][len(estimated_linearvarlist[i])-1-sample_size+cut_point:len(estimated_linearvarlist[i])]) for i in range(0,len(estimated_linearvarlist))]
        self.estimated_lambdalist_mean = [numpy.mean(estimated_lambda[i][
                                                   len(estimated_lambda[i]) - 1 - sample_size+cut_point:len(
                                                       estimated_lambda[i])]) for i in
                                        range(0, len(estimated_lambda))]
        self.estimated_linearvarlist_std=[numpy.std(estimated_linearvarlist[i][len(estimated_linearvarlist[i])-1-sample_size+cut_point:len(estimated_linearvarlist[i])]) for i in range(0,len(estimated_linearvarlist))]
        self.estimated_lambdalist_std=[numpy.std(estimated_lambda[i][
                                                   len(estimated_lambda[i]) - 1 - sample_size+cut_point:len(
                                                       estimated_lambda[i])]) for i in
                                        range(0, len(estimated_lambda))]



        return [estimated_linearvarlist,estimated_lambda]
def helloworld():
    random.seed(1234+int(random.normalvariate(0,1000)))
    i = 450
    ulines=[]
    counts = 1
    x_1 = []
    x_2 = []
    x_3 = []
    x_4 = []
    x_5 = []
    x_6 = []
    x_7 = []
    x_8 = []
    y = []
    dis0_param1 = random.uniform(-100, 100)
    dis0_param2 = random.uniform(1, 100)
    dis1_param1 = random.uniform(-100, 100)
    dis1_param2 = random.uniform(1, 1000)
    dis2_param1 = random.uniform(0, 100)
    dis2_param2 = random.uniform(0, 100)
    dis2_param3 = random.uniform(-100, 100)
    dis2_param4 = random.uniform(1, 1000)
    dis3_param1 = random.uniform(0, 100)
    dis4_param1 = random.uniform(0, 10)
    dis4_param2 = random.uniform(0, 100)
    dis5_param1 = random.uniform(-1000, 1000)
    dis5_param2 = random.uniform(-1000, 1000)

    disb1_param1 = random.uniform(-100, 100)
    disb1_param2 = random.uniform(1, 1000)
    disb2_param1 = random.uniform(-100, 100)
    disb2_param2 = random.uniform(1, 1000)
    disb3_param1 = random.uniform(-100, 100)
    disb3_param2 = random.uniform(1, 1000)
    disb4_param1 = random.uniform(-100, 100)
    disb4_param2 = random.uniform(1, 1000)
    disb5_param1 = random.uniform(-100, 100)
    disb5_param2 = random.uniform(1, 1000)

    disu_param1 = random.uniform(-100, 100)
    disu_param2 = random.uniform(100, 1000)
    print("U的分布参数:%f | %f" % (disu_param1, disu_param2))

    dise_param = 100
    b0 = random.normalvariate(dis0_param1, dis0_param2)
    b1 = random.normalvariate(disb1_param1, disb1_param2)
    b2 = random.normalvariate(disb2_param1, disb2_param2)
    b3 = random.normalvariate(disb3_param1, disb3_param2)
    b4 = random.normalvariate(disb4_param1, disb4_param2)
    b5 = random.normalvariate(disb5_param1, disb5_param2)
    dict_shock_x6 = {
        0:-1,
        1: 1
    }
    dict_shock_x7 = {
        'mike': 1,
        'joe': 5,
    }
    dict_shock_x8 = {
        1997: 1,
        1998: 1.6,
        1999: 1.8,
        2000: 2,
        2001: 2.3
    }

    while counts < i:
        x1 = random.normalvariate(dis1_param1, dis1_param2)
        x2 = random.betavariate(dis2_param1, dis2_param2) * random.normalvariate(dis2_param3, dis2_param4)
        x3 = random.gammavariate(1, dis3_param1)  # 指数分布
        x4 = random.gammavariate(dis4_param1, dis4_param2)
        x5 = random.uniform(dis5_param1, dis5_param2)
        x6 = random_multivar([0, 1])
        x7 = random_multivar(['mike', 'joe'])
        x8 = random_multivar([1997, 1998, 1999, 2000, 2001])
        e = random.normalvariate(0, dise_param)
        linear_value =  b0+b1 * x1 + b2 * x2
        x_1.append(x1)
        x_2.append(x2)
        x_3.append(x3)
        x_4.append(x4)
        x_5.append(x5)
        x_6.append(x6)
        x_7.append(x7)
        x_8.append(x8)
        y.append(
            linear_value * dict_shock_x6[x6] * dict_shock_x7[x7] * dict_shock_x8[x8] + b3 * x3 + b4 * x4 + b5 * x5 +e)
        ulines.append(dict_shock_x6[x6] * dict_shock_x7[x7] * dict_shock_x8[x8])
        counts += 1

    begin_time = time.time()
    ops = GRC(y, [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8], set_control_group=[0,'mike',1997],identify_var_index=2, classify_var_index=3, constant=True)
    print("------------------------")
    print("%f|%f|%f|%f|%f|%f" % (b0, b1, b2, b3, b4, b5))
    print("%f|%f|%f|%f|%f|%f" % (ops.estimated_linearvarlist_mean[0], ops.estimated_linearvarlist_mean[1], ops.estimated_linearvarlist_mean[2], ops.estimated_linearvarlist_mean[3], ops.estimated_linearvarlist_mean[4], ops.estimated_linearvarlist_mean[5]))
    print("%f|%f|%f|%f|%f|%f" % (ops.ols_result_origin.params[0], ops.ols_result_origin.params[1], ops.ols_result_origin.params[2], ops.ols_result_origin.params[3], ops.ols_result_origin.params[4], ops.ols_result_origin.params[5]))
    print("一共用了%f秒" % (time.time() - begin_time))

    print("---------mean:",numpy.mean(ulines))
    print("------------------------------")
    new_mean_u=numpy.mean(ulines)
    mean_u=0
    for i in range(0,len(ops.each_means)):
        mean_u+=ops.each_means[i]*len(ops.each_y[i])/len(ops.data)
    each_meany=[]
    each_index_meany=[]
    for index_6 in dict_shock_x6:
        for index_7 in dict_shock_x7:
            for index_8 in dict_shock_x8:
                each_meany.append(dict_shock_x6[index_6]*dict_shock_x7[index_7]*dict_shock_x8[index_8])
                each_index_meany.append([index_6,index_7,index_8])
    for i in range(0, len(ops.estimated_lambdalist_mean)):
        print(ops.estimated_lambdalist_mean[i], '||', ops.each_means[i], "||", each_meany[each_index_meany.index(ops.alphas[i][0])])
    print("---------------------")
    for i in range(0,len(ops.estimated_linearvarlist_mean)):
        print(ops.estimated_linearvarlist_mean[i],'||',ops.ols_result_origin.params[i])
    AK_mean=[each_meany[i]/min(each_meany) for i in range(0,len(each_meany))]
    new_umeans=ops.each_means/(1/new_mean_u)
    BK_mean=[new_umeans[i]/min(new_umeans) for i in range(0,len(new_umeans))]
    time.sleep(15)
helloworld()