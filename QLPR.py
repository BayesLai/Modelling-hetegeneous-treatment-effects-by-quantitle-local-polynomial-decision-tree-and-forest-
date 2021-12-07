# coding=UTF-8
from numpy import mean,std,array,cov,linalg
from math import sqrt,pi,exp,log
from statsmodels.api import OLS,add_constant,families,GLM,Logit
import random as random
from scipy.stats import chi2,chi2_contingency,f,binom,mannwhitneyu,norm

initial_value=[0,pi/4]#最老版本的迭代初值
def random_multivar(select_list):
    selected_list=[select_list[:].index(i) for i in select_list[:]]
    a=min(selected_list)
    b=max(selected_list)
    random_ab=random.uniform(0,1)
    return select_list[int(random_ab*len(select_list))]
def BOOSTRAP_WMW_rank_test(sample_list):
    BOOSTRAP_sample_1=sample_list[0:int(len(sample_list)/2)]
    BOOSTRAP_sample_2=sample_list[int(len(sample_list)/2):]
    return mannwhitneyu(x=BOOSTRAP_sample_1,y=BOOSTRAP_sample_2)
def Likelihood_test(MEAN_list,VAR_list):
    mu_MEAN=mean(MEAN_list)
    sigma_VAR=mean(VAR_list)
    PDF_compare=[]
    for i in range(0,len(MEAN_list)):
        PDF_compare.append(log(norm.pdf(x=MEAN_list[i],loc=MEAN_list[i],scale=sqrt(VAR_list[i])))-log(norm.pdf(x=MEAN_list[i],loc=mu_MEAN,scale=sqrt(sigma_VAR))))
    LikelihoodStatistics=2*mean(PDF_compare)
    print("Likelihood Statistics",LikelihoodStatistics,"~Chi(%d)"%(2))
    return [LikelihoodStatistics,float(1)-chi2.cdf(x=LikelihoodStatistics,df=(2))]
class QLPRF(object):
    def __init__(self,y_datas,coviate_datas,identify_var_datas,treat_var_index,Response_type='normal',contineousity=[],confidence=0,constant=True):
        self.coviate_datas=coviate_datas[:]
        self.list_classifyvar_list=[]
        self.y = y_datas
        self.identify_var_datas=identify_var_datas
        self.identify_var_index=len(identify_var_datas)
        self.constant=constant
        self.treat_var_index=treat_var_index-1
        self.coviate_datas_t = array(self.coviate_datas).T
        self.sorted_identify_var = [sorted(i[:]) for i in self.identify_var_datas] if contineousity==[] else [[] for i in self.identify_var_datas]
        self.quantile_identify_var = []
        self.forest_treenum=30
        self.each_tree_sample_num=int(len(self.y)*0.3)
        self.confidence=confidence
        if contineousity==[]:
            for i in range(0, len(self.identify_var_datas)):
                self.quantile_identify_var.append(
                [float(self.sorted_identify_var[i].index(j)+1) / len(self.identify_var_datas[i]) for j in self.identify_var_datas[i]])
        else:
            for i in range(0, len(self.identify_var_datas)):
                if contineousity[i] in['yes','no']:
                    if contineousity[i]=='yes':
                        got_list=[]
                        for index in range(0,len(self.identify_var_datas[i])):
                            if self.identify_var_datas[i][index] in got_list:
                                pass
                            else:
                                got_list.append(self.identify_var_datas[i][index])
                        self.quantile_identify_var.append(
                            [(float(got_list.index(j) + 0.5) )/ (len(got_list)) for j in
                             self.identify_var_datas[i]])
                    elif contineousity[i]=='no':
                        self.quantile_identify_var.append(
                            [float(self.sorted_identify_var[i].index(j) + 1) / len(self.identify_var_datas[i]) for j in
                             self.identify_var_datas[i]])
                    else:
                        raise ValueError("Values must be in 'yes' or 'no'")
        self.Response_type=Response_type
        if self.Response_type== 'normal':
            self.ols_result_origin = OLS(self.y, add_constant(
                array(self.coviate_datas).T)).fit()
        elif self.Response_type=='binary':
            self.ols_result_origin = Logit(self.y, add_constant(
                array(self.coviate_datas).T)).fit()
        else:
            Method_dict={
                'possion':families.Poisson(),
                'binomial':families.Binomial()
            }
            GLM_model = GLM(self.y, add_constant(array(self.coviate_datas).T), family=Method_dict[self.Response_type])
            self.ols_result_origin = GLM_model.fit()
        print(self.ols_result_origin.summary())
        self.forest()
        self.DHG = self.D_homegeneous_regression()
        self.HETEROGENEOUSITY_TEST()
    def single_spilt_tree(self,y,covariates,treat_var,phi_vector,x_index,quantile_var):
        copy_covariates=covariates[:]
        initial_length=len(self.coviate_datas)+1 if self.constant==True else len(self.coviate_datas)
        phi_vector = sorted(phi_vector)
        for j in range(2, len(phi_vector) * 2):
            copy_covariates.append([])
        for i in range(len(y)):
            for j in range(1, len(phi_vector)):
                if phi_vector[j - 1] <= quantile_var[i] <= phi_vector[j]:
                    if treat_var[i] == 1:
                        copy_covariates[initial_length - 1 + j - 1].append(1)

                    else:
                        copy_covariates[initial_length - 1 + j - 1].append(0)

                else:
                    copy_covariates[initial_length - 1 + j - 1].append(0)
        for i in range(len(y)):
            for j in range(1, len(phi_vector)):
                if phi_vector[j - 1] <= self.quantile_identify_var[x_index][i] <= phi_vector[j]:
                    if treat_var[i] == 1:
                        copy_covariates[initial_length - 1 + j + len(phi_vector) - 2].append(
                            (self.quantile_identify_var[x_index][i] - (
                                        phi_vector[j - 1] + phi_vector[j]) / 2))
                    else:
                        copy_covariates[initial_length - 1 + j + len(phi_vector) - 2].append(0)
                else:
                    copy_covariates[initial_length - 1 + j + len(phi_vector) - 2].append(0)

        if self.Response_type == 'normal':
            spilt_OLS_result = OLS(y, add_constant(
                array(copy_covariates).T)).fit()
        elif self.Response_type == 'binary':
            spilt_OLS_result = Logit(y, add_constant(
                array(copy_covariates).T)).fit()
        else:
            Method_dict = {
                'possion': families.Poisson(),
                'binomial': families.Binomial()
            }
            GLM_model = GLM(self.y, add_constant(array(copy_covariates).T), family=Method_dict[self.Response_type])
            spilt_OLS_result = GLM_model.fit()
        chi2_a=float(0)
        chi2_b=float(0)
        for i in range(len(phi_vector)):
            chi2_a+=spilt_OLS_result.tvalues[initial_length+i-1 ]**2
            chi2_b+=spilt_OLS_result.tvalues[initial_length+i-1+len(phi_vector)-1 ]**2
        #print(chi2_a,chi2_b)
        return 1-f.cdf(float(chi2_a)/float(chi2_b),dfn=len(phi_vector)-1,dfd=len(phi_vector)-1)
    def get_single_spilttree_OLSparams(self,y,covariates,x_index,treat_var,phi_vector,quantile_var):
        copy_covariates = covariates[:]
        phi_vector = sorted(phi_vector)
        initial_length = len(self.coviate_datas) + 1 if self.constant == True else len(self.coviate_datas)
        for j in range(2, len(phi_vector) * 2):
            copy_covariates.append([])
        for i in range(len(y)):
            for j in range(1, len(phi_vector)):
                if phi_vector[j - 1] <= self.quantile_identify_var[x_index][i] <= phi_vector[j]:
                    if treat_var[i] == 1:
                        copy_covariates[initial_length - 1 + j - 1].append(1)

                    else:
                        copy_covariates[initial_length - 1 + j - 1].append(0)

                else:
                    copy_covariates[initial_length - 1 + j - 1].append(0)
        for i in range(len(y)):
            for j in range(1, len(phi_vector)):
                if phi_vector[j - 1] <= quantile_var[i] <= phi_vector[j]:
                    if treat_var[i] == 1:
                        copy_covariates[initial_length - 1 + j + len(phi_vector) - 2].append(
                            (self.quantile_identify_var[x_index][i] - (phi_vector[j - 1] + phi_vector[j]) / 2))
                    else:
                        copy_covariates[initial_length - 1 + j + len(phi_vector) - 2].append(0)
                else:
                    copy_covariates[initial_length - 1 + j + len(phi_vector) - 2].append(0)
        if self.Response_type == 'normal':
            spilt_OLS_result = OLS(y, add_constant(
                array(copy_covariates).T)).fit()
        elif self.Response_type == 'binary':
            spilt_OLS_result = Logit(y, add_constant(
                array(copy_covariates).T)).fit()
        else:
            Method_dict = {
                'possion': families.Poisson(),
                'binomial': families.Binomial()
            }
            GLM_model = GLM(self.y, add_constant(array(copy_covariates).T), family=Method_dict[self.Response_type])
            spilt_OLS_result = GLM_model.fit()
        sigmas=array(spilt_OLS_result.params)/array(spilt_OLS_result.tvalues)
        return [spilt_OLS_result.params[len(self.coviate_datas)+1 if self.constant==True else len(self.coviate_datas):],
        sigmas]
    def iter_casualtree(self,reg_index):
        y_boostrap=[]
        covariatesa=[[] for i in self.coviate_datas]
        identify_datas=[[] for i in self.identify_var_datas]
        quantile=[]
        index_list=[j for j in range(len(self.y))]
        for i in range(0,self.each_tree_sample_num):
            random_index=random_multivar(index_list)
            y_boostrap.append(self.y[random_index])
            for k in range(0,len(self.coviate_datas)):
                covariatesa[k].append(self.coviate_datas[k][random_index])
            for k in range(0,len(self.identify_var_datas)):
                identify_datas[k].append(self.identify_var_datas[k][random_index])
            quantile.append(self.quantile_identify_var[reg_index][random_index])
        goto_func = lambda x,k: self.single_spilt_tree(y=y_boostrap,covariates=covariatesa, x_index=k,
                                                     treat_var=covariatesa[self.treat_var_index],
                                                     phi_vector=x,quantile_var=quantile)
        n_vector=[]
        pvalue_vector=[]
        OLS_params=[]
        ends_point = int((len(self.y)) ** 0.5) if self.Response_type != 'binary' else int((len(self.y)) ** 0.25)
        print("end point",ends_point)
        for grid_search in range(2, ends_point):
            initial_phi_vector = [0]
            for i in range(grid_search):
                initial_phi_vector.append(float(i + 1) / grid_search)

            pvalue_vector.append(goto_func(initial_phi_vector, reg_index))
            n_vector.append(grid_search)
            #print("add pvalue:",pvalue_vector[len(pvalue_vector)-1],"add n:",grid_search)
        best_n_param=n_vector[pvalue_vector.index(min(pvalue_vector))]
        phi_vector=[0]
        for i in range(best_n_param):
            phi_vector.append(float(i + 1) / best_n_param)

        return [best_n_param,phi_vector]
    def forest(self):
        self.forest_params_distributions=[{'p':None,'n':None,'params':None,'M_nums':None} for i in self.identify_var_datas]
        print("Start forest work.")
        for regindex in range(len(self.identify_var_datas)):
            M_distribution=[]
            for i in range(self.forest_treenum):
                tree_result=self.iter_casualtree(reg_index=regindex)
                M_distribution.append(tree_result[0]-1)
                print("Get a tree!!!",regindex,"||",tree_result)
            mu=mean(M_distribution)
            sigma=std(M_distribution)**2
            print(regindex,"||||",mu,"||||",sigma)
            if 0<1-sigma/mu<=1:
                self.forest_params_distributions[regindex]['p']=1-sigma/mu
                self.forest_params_distributions[regindex]['n'] = int(mu/self.forest_params_distributions[regindex]['p'])
                self.forest_params_distributions[regindex]['params']={
                '1':[[0,1],[self.ols_result_origin.params[self.treat_var_index]],[self.ols_result_origin.params[self.treat_var_index]/self.ols_result_origin.tvalues[self.treat_var_index]]]
                }
                for i in range(2, self.forest_params_distributions[regindex]['n']+1):
                    initial_phi_vector = [0]
                    for ink in range(i):
                        initial_phi_vector.append(float(ink + 1) / i)
                    params_result = self.get_single_spilttree_OLSparams(y=self.y, covariates=self.coviate_datas,
                                                                        x_index=regindex,
                                                                        treat_var=self.coviate_datas[
                                                                            self.treat_var_index],
                                                                        phi_vector=initial_phi_vector,
                                                                        quantile_var=self.quantile_identify_var[
                                                                            regindex])
                    self.forest_params_distributions[regindex]['params'][str(i)] = [
                        initial_phi_vector,
                        params_result[0],
                        params_result[1]
                    ]
            else:
                self.forest_params_distributions[regindex]['p'] = None
                self.forest_params_distributions[regindex]['params'] = {
                    '1': [[0, 1], [self.ols_result_origin.params[self.treat_var_index]], [
                        self.ols_result_origin.params[self.treat_var_index] / self.ols_result_origin.tvalues[
                            self.treat_var_index]],1]
                }
                for i in M_distribution:
                    if str(i) in self.forest_params_distributions[regindex].keys():
                        self.forest_params_distributions[regindex]['params'][str(i)][3]+=1
                    else:
                        initial_phi_vector = [0]
                        for ink in range(i):
                            initial_phi_vector.append(float(ink + 1) / i)
                        params_result = self.get_single_spilttree_OLSparams(y=self.y, covariates=self.coviate_datas,
                                                                            x_index=regindex,
                                                                            treat_var=self.coviate_datas[
                                                                                self.treat_var_index],
                                                                            phi_vector=initial_phi_vector,
                                                                            quantile_var=self.quantile_identify_var[
                                                                                regindex])
                        self.forest_params_distributions[regindex]['params'][str(i)] = [
                            initial_phi_vector,
                            params_result[0],
                            params_result[1],0]


        print("finished")
        print("End forest work.")
    def D_get_mean_param(self,identify_var,quantile):
        for i in self.forest_params_distributions:
            if i == []:
                raise ValueError("应该先使用forest函数训练")
            else:
                pass
        mean_sum =float( 0)

        for i in self.forest_params_distributions[identify_var]['params']:
            if self.forest_params_distributions[identify_var]['p'] != None:
                for k in range(1, len(self.forest_params_distributions[identify_var]['params'][i][0])):
                    if self.forest_params_distributions[identify_var]['params'][i][0][k - 1] <= quantile <= \
                                self.forest_params_distributions[identify_var]['params'][i][0][k]:
                        binominal_pdf = binom.pmf(int(i) - 1,
                                                      self.forest_params_distributions[identify_var]['n'],
                                                      self.forest_params_distributions[identify_var]['p'])
                        mean_sum += binominal_pdf * \
                                        self.forest_params_distributions[identify_var]['params'][i][1][
                                            k - 1]
            else:
                for k in range(1, len(self.forest_params_distributions[identify_var]['params'][i][0])):
                    if self.forest_params_distributions[identify_var]['params'][i][0][k - 1] <= quantile <= \
                                    self.forest_params_distributions[identify_var]['params'][i][0][k]:
                        mean_sum += float(self.forest_params_distributions[identify_var]['params'][i][3]) / \
                                    (self.each_tree_sample_num+1) * \
                                            self.forest_params_distributions[identify_var]['params'][i][1][
                                                k - 1]
        return mean_sum
    def D_get_VAR_param(self,identify_var,quantile):
        for i in self.forest_params_distributions:
            if i == []:
                raise ValueError("应该先使用forest函数训练")
            else:
                pass
        VAR_sum = float(0)
        for i in self.forest_params_distributions[identify_var]['params']:
            if self.forest_params_distributions[identify_var]['p'] != None:
                for k in range(1, len(self.forest_params_distributions[identify_var]['params'][i][0])):
                    if self.forest_params_distributions[identify_var]['params'][i][0][k - 1] <= quantile <= \
                            self.forest_params_distributions[identify_var]['params'][i][0][k]:
                        binominal_pdf = binom.pmf(int(i) - 1,
                                                  self.forest_params_distributions[identify_var]['n'],
                                                  self.forest_params_distributions[identify_var]['p'])
                        VAR_sum += binominal_pdf * \
                                    self.forest_params_distributions[identify_var]['params'][i][2][
                                        k - 1]**2
            else:
                for k in range(1, len(self.forest_params_distributions[identify_var]['params'][i][0])):
                    if self.forest_params_distributions[identify_var]['params'][i][0][k - 1] <= quantile <= \
                            self.forest_params_distributions[identify_var]['params'][i][0][k]:
                        VAR_sum += float(self.forest_params_distributions[identify_var]['params'][i][3]) / \
                                   (self.each_tree_sample_num + 1) * \
                                    self.forest_params_distributions[identify_var]['params'][i][2][
                                        k - 1]**2
        return VAR_sum
    def get_mean_param(self,identify_var,quantile):
        for i in self.forest_params_distributions:
            if i == []:
                raise ValueError("应该先使用forest函数训练")
            else:
                pass
        mean_sum =float( 0)
        for i in self.forest_params_distributions[identify_var]['params']:

            for k in range(1, len(self.forest_params_distributions[identify_var]['params'][i][0])):
                if self.forest_params_distributions[identify_var]['params'][i][0][k - 1] <= quantile <= \
                        self.forest_params_distributions[identify_var]['params'][i][0][k]:
                    binominal_pdf=binom.pmf(int(i)-1,
                                            self.forest_params_distributions[identify_var]['n'],
                                            self.forest_params_distributions[identify_var]['p'])
                    mean_sum += binominal_pdf* \
                                self.forest_params_distributions[identify_var]['params'][i][1][
                                    k - 1]
        return mean_sum
    def get_VAR_param(self,identify_var,quantile):
        for i in self.forest_params_distributions:
            if i == []:
                raise ValueError("应该先使用forest函数训练")
            else:
                pass
        mean_VAR=float( 0)
        for i in self.forest_params_distributions[identify_var]['params']:
            for k in range(1, len(self.forest_params_distributions[identify_var]['params'][i][0])):
                if self.forest_params_distributions[identify_var]['params'][i][0][k - 1] <= quantile <= \
                        self.forest_params_distributions[identify_var]['params'][i][0][k]:
                    binominal_pdf=binom.pmf(int(i)-1,
                                            self.forest_params_distributions[identify_var]['n'],
                                            self.forest_params_distributions[identify_var]['p'])
                    mean_VAR += binominal_pdf* \
                                (self.forest_params_distributions[identify_var]['params'][i][2][
                                    k - 1])**2
        return mean_VAR
    def D_homegeneous_regression(self):
        for i in self.forest_params_distributions:
            if i == []:
                raise ValueError("应该先使用forest函数训练")
            else:
                pass
        self.HG_plot_lists = [[[None for j in self.y], [None for j in self.y],[None for j in self.y]] for i in self.identify_var_datas]
        new_y=[]

        for i_params_list in range(len(self.identify_var_datas)):
            for i in range(len(self.y)):
                if self.identify_var_datas[i_params_list].count(self.identify_var_datas[i_params_list][i]) > 1:
                    now_index = self.sorted_identify_var[i_params_list].index(
                        self.identify_var_datas[i_params_list][i])
                    self.HG_plot_lists[i_params_list][0][now_index] = self.identify_var_datas[i_params_list][now_index]
                    self.HG_plot_lists[i_params_list][1][now_index] = self.D_get_mean_param(identify_var=i_params_list,
                                                                                       quantile=
                                                                                       self.quantile_identify_var[
                                                                                           i_params_list][
                                                                                          now_index])
                    for repeat_index in range(
                            self.identify_var_datas[i_params_list].count(self.identify_var_datas[i_params_list][i])):
                        old_index = now_index
                        now_index += 1
                        self.HG_plot_lists[i_params_list][0][now_index] = self.identify_var_datas[i_params_list][now_index]
                        self.HG_plot_lists[i_params_list][1][now_index] = self.D_get_mean_param(identify_var=i_params_list,
                                                                                           quantile=
                                                                                           self.quantile_identify_var[
                                                                                               i_params_list][
                                                                                               old_index])
                else:
                    now_index = self.sorted_identify_var[i_params_list].index(
                        self.identify_var_datas[i_params_list][i])
                    self.HG_plot_lists[i_params_list][0][now_index] = self.identify_var_datas[i_params_list][now_index]
                    self.HG_plot_lists[i_params_list][1][now_index] = self.D_get_mean_param(identify_var=i_params_list,
                                                                                       quantile=
                                                                                       self.quantile_identify_var[
                                                                                           i_params_list][now_index])
        for i_params_list in range(len(self.identify_var_datas)):
            for i in range(len(self.y)):
                if self.identify_var_datas[i_params_list].count(self.identify_var_datas[i_params_list][i]) > 1:
                    now_index = self.sorted_identify_var[i_params_list].index(
                        self.identify_var_datas[i_params_list][i])
                    self.HG_plot_lists[i_params_list][2][now_index] =self.D_get_VAR_param(identify_var=i_params_list,
                                                                                       quantile=
                                                                                       self.quantile_identify_var[
                                                                                           i_params_list][now_index])
                    for repeat_index in range(
                            self.identify_var_datas[i_params_list].count(self.identify_var_datas[i_params_list][i])):
                        old_index = now_index
                        now_index += 1
                        self.HG_plot_lists[i_params_list][2][now_index] = self.D_get_VAR_param(identify_var=i_params_list,
                                                                                          quantile=
                                                                                          self.quantile_identify_var[
                                                                                              i_params_list][now_index])
                else:
                    now_index = self.sorted_identify_var[i_params_list].index(
                        self.identify_var_datas[i_params_list][i])
                    self.HG_plot_lists[i_params_list][2][now_index] = self.D_get_VAR_param(identify_var=i_params_list,
                                                                                      quantile=
                                                                                      self.quantile_identify_var[
                                                                                          i_params_list][now_index])
        cal_resid_treatment = []
        cal_resid = []
        for i in range(len(self.y)):
            negative_E_sum_fz=float(0)
            for i_params_list in range(len(self.identify_var_datas)):
                negative_E_sum_fz+=self.HG_plot_lists[i_params_list][1][i]
            new_y.append(self.y[i]-negative_E_sum_fz*self.coviate_datas[self.treat_var_index][i])
        self.cutparams_OLS_result = OLS(new_y, add_constant(
            array(self.coviate_datas).T)).fit()
        self.resid = self.cutparams_OLS_result.resid
        for i in range(len(self.y)):
            if self.coviate_datas[self.treat_var_index][i] == 1:
                cal_resid_treatment.append(self.resid[i] * self.resid[i])
            else:
                cal_resid.append(self.resid[i] * self.resid[i])
        self.estimated_var = abs(sum(cal_resid_treatment) / (len(cal_resid_treatment) - 1) - sum(cal_resid) / (
                    len(cal_resid) - 1) )
        print("my estimated VAR:",
              self.estimated_var)
        for i_params in range(len(self.identify_var_datas)):
            print(self.HG_plot_lists[i_params][2])
        return self.HG_plot_lists

    def HETEROGENEOUSITY_TEST(self):
        for INDEX_params in range(len(self.identify_var_datas)):
            try:
                WMW_test_VAR_p_value = BOOSTRAP_WMW_rank_test(self.HG_plot_lists[INDEX_params][2])
                pvalue=WMW_test_VAR_p_value.pvalue
            except ValueError:
                pvalue=0
            print("-"*100)
            if pvalue>=self.confidence:
                print(INDEX_params+1,"th Heterogeneous variable's VAR unsatisfy confidence conditions." )
                print("P-value:", pvalue,"while confidence level=",self.confidence)
                try:
                    WMW_test_mean_p_value = BOOSTRAP_WMW_rank_test(self.HG_plot_lists[INDEX_params][1])
                    pvalue=WMW_test_mean_p_value.pvalue
                except ValueError:
                    print("All numbers are identical in mannwhitneyu")
                    pvalue=0
                print("P-value:", pvalue)
                if pvalue>= self.confidence:
                    print(INDEX_params + 1, "th Heterogeneous variable's mean unsatisfy confidence conditions.")
                    L = Likelihood_test(self.HG_plot_lists[INDEX_params][1], self.HG_plot_lists[INDEX_params][2])
                    print("Likelihood Test pass" if L[1] < self.confidence else "Likelihood Test didn't pass")
                    print("Likelihood Statistics:", L[0], "P_value:", L[1], "while confidence level is ",
                          self.confidence)
                else:
                    print(INDEX_params + 1, "th Heterogeneous variable's mean  satisfy confidence conditions.")
            else:
                print(INDEX_params + 1, "th Heterogeneous variable's VAR  satisfy confidence conditions.")
                print("P-value:", pvalue, "while confidence level=", self.confidence)
                try:
                    WMW_test_mean_p_value = BOOSTRAP_WMW_rank_test(self.HG_plot_lists[INDEX_params][1])
                    pvalue=WMW_test_mean_p_value.pvalue
                except ValueError:
                    print("All numbers are identical in mannwhitneyu")
                    pvalue = 0
                print("P-value:", pvalue)
                if pvalue>= self.confidence:
                    print(INDEX_params + 1, "th Heterogeneous variable's mean unsatisfy confidence conditions.")
                else:
                    print(INDEX_params + 1, "th Heterogeneous variable's mean  satisfy confidence conditions.")
class QLPRT(object):
    def __init__(self,y_datas,coviate_datas,identify_var_datas,treat_var_index,Response_type='normal',confidence=0,contineousity=[],constant=True):
        self.coviate_datas=coviate_datas[:]
        self.list_classifyvar_list=[]
        self.y = y_datas
        self.identify_var_datas=identify_var_datas
        self.identify_var_index=len(identify_var_datas)
        self.constant=constant
        self.treat_var_index=treat_var_index-1
        self.coviate_datas_t = array(self.coviate_datas).T
        self.sorted_identify_var = [sorted(i[:]) for i in self.identify_var_datas] if contineousity==[] else [[] for i in self.identify_var_datas]
        self.quantile_identify_var = []
        self.forest_treenum=50
        self.each_tree_sample_num=int(len(self.y)*0.5)
        self.confidence=confidence
        if contineousity==[]:
            for i in range(0, len(self.identify_var_datas)):
                self.quantile_identify_var.append(
                [float(self.sorted_identify_var[i].index(j)+1) / len(self.identify_var_datas[i]) for j in self.identify_var_datas[i]])
        else:
            for i in range(0, len(self.identify_var_datas)):
                if contineousity[i] in['yes','no']:
                    if contineousity[i]=='yes':
                        got_list=[]
                        for index in range(0,len(self.identify_var_datas[i])):
                            if self.identify_var_datas[i][index] in got_list:
                                pass
                            else:
                                got_list.append(self.identify_var_datas[i][index])
                        self.quantile_identify_var.append(
                            [(float(got_list.index(j) + 0.5) )/ (len(got_list)) for j in
                             self.identify_var_datas[i]])
                    elif contineousity[i]=='no':
                        self.quantile_identify_var.append(
                            [float(self.sorted_identify_var[i].index(j) + 1) / len(self.identify_var_datas[i]) for j in
                             self.identify_var_datas[i]])
                    else:
                        raise ValueError("Values must be in 'yes' or 'no'")
        self.Response_type=Response_type
        if self.Response_type == 'normal':
            self.ols_result_origin = OLS(self.y, add_constant(
                array(self.coviate_datas).T)).fit()
        elif self.Response_type == 'binary':
            self.ols_result_origin = Logit(self.y, add_constant(
                array(self.coviate_datas).T)).fit()
        else:
            Method_dict = {
                'possion': families.Poisson(),
                'binomial': families.Binomial()
            }
            GLM_model = GLM(self.y, add_constant(array(self.coviate_datas).T), family=Method_dict[self.Response_type])
            self.ols_result_origin = GLM_model.fit()
        print(self.ols_result_origin.summary())
        self.population_phi_vector = [[] for i in range(0, len(self.identify_var_datas))]
        self.population_tree_OLSparams = [[] for i in range(0, len(self.identify_var_datas))]
        self.population_best_num = [None for i in range(len(self.identify_var_datas))]
        self.DHG = self.D_homegeneous_regression()
        self.HETEROGENEOUSITY_TEST()
    def population_tree(self,reg_index):
        goto_func = lambda x,k: self.single_spilt_tree(y=self.y, covariates=self.coviate_datas,
                                                     treat_var=self.coviate_datas[self.treat_var_index],
                                                     phi_vector=x,x_index=k,quantile_var=self.quantile_identify_var[reg_index])
        n_vector = []
        pvalue_vector = []
        ends_point= int((len(self.y)) ** 0.5) if self.Response_type!='binary' else  int((len(self.y)) ** 0.2)
        for grid_search in range(2, ends_point):
            initial_phi_vector = [0]
            for i in range(grid_search):
                initial_phi_vector.append(float(i + 1) / grid_search)

            pvalue_vector.append(goto_func(initial_phi_vector,reg_index))
            n_vector.append(grid_search)
        best_n_param = n_vector[pvalue_vector.index(min(pvalue_vector))]
        phi_vector = [0]
        for i in range(best_n_param):
            phi_vector.append(float(i + 1) / best_n_param)
        self.population_phi_vector[reg_index] = phi_vector
        self.population_tree_OLSparams[reg_index] = self.get_single_spilttree_OLSparams(y=self.y,
                                                                                        covariates=self.coviate_datas,
                                                                                        x_index=reg_index,
                                                                                        treat_var=
                                                                                        self.coviate_datas[
                                                                                            self.treat_var_index],
                                                                                        phi_vector=phi_vector,
                                                                                        quantile_var=self.quantile_identify_var[reg_index])


        self.population_best_num[reg_index] = best_n_param
    def single_spilt_tree(self,y,covariates,treat_var,phi_vector,x_index,quantile_var):
        copy_covariates=covariates[:]
        initial_length=len(self.coviate_datas)+1 if self.constant==True else len(self.coviate_datas)
        phi_vector = sorted(phi_vector)
        for j in range(2, len(phi_vector) * 2):
            copy_covariates.append([])

        for i in range(len(y)):
            for j in range(1, len(phi_vector)):
                if phi_vector[j - 1] <= quantile_var[i] <= phi_vector[j]:
                    if treat_var[i] == 1:
                        copy_covariates[initial_length - 1 + j - 1].append(1)

                    else:
                        copy_covariates[initial_length - 1 + j - 1].append(0)

                else:
                    copy_covariates[initial_length - 1 + j - 1].append(0)
        for i in range(len(y)):
            for j in range(1, len(phi_vector)):
                if phi_vector[j - 1] <= self.quantile_identify_var[x_index][i] <= phi_vector[j]:
                    if treat_var[i] == 1:
                        copy_covariates[initial_length - 1 + j + len(phi_vector) - 2].append(
                            (self.quantile_identify_var[x_index][i] - (
                                        phi_vector[j - 1] + phi_vector[j]) / 2))
                    else:
                        copy_covariates[initial_length - 1 + j + len(phi_vector) - 2].append(0)
                else:
                    copy_covariates[initial_length - 1 + j + len(phi_vector) - 2].append(0)
        if self.Response_type == 'normal':
            spilt_OLS_result = OLS(y, add_constant(
                array(copy_covariates).T)).fit()
        elif self.Response_type == 'binary':
            try:
                spilt_OLS_result = Logit(y, add_constant(
                array(copy_covariates).T)).fit()
            except linalg.LinAlgError:
                print("sdf")
        else:
            Method_dict = {
                'possion': families.Poisson(),
                'binomial': families.Binomial()
            }
            GLM_model = GLM(self.y, add_constant(array(copy_covariates).T), family=Method_dict[self.Response_type])
            spilt_OLS_result = GLM_model.fit()
        chi2_a=float(0)
        chi2_b=float(0)
        for i in range(len(phi_vector)):
            chi2_a+=spilt_OLS_result.tvalues[initial_length+i-1 ]**2
            chi2_b+=spilt_OLS_result.tvalues[initial_length+i-1+len(phi_vector)-1 ]**2
        #print(chi2_a,chi2_b)
        return 1-f.cdf(float(chi2_a)/float(chi2_b),dfn=len(phi_vector)-1,dfd=len(phi_vector)-1)
    def get_single_spilttree_OLSparams(self,y,covariates,x_index,treat_var,phi_vector,quantile_var):
        copy_covariates = covariates[:]
        phi_vector = sorted(phi_vector)
        initial_length = len(self.coviate_datas) + 1 if self.constant == True else len(self.coviate_datas)
        for j in range(2, len(phi_vector) * 2):
            copy_covariates.append([])
        for i in range(len(y)):
            for j in range(1, len(phi_vector)):
                if phi_vector[j - 1] <= self.quantile_identify_var[x_index][i] <= phi_vector[j]:
                    if treat_var[i] == 1:
                        copy_covariates[initial_length - 1 + j - 1].append(1)

                    else:
                        copy_covariates[initial_length - 1 + j - 1].append(0)

                else:
                    copy_covariates[initial_length - 1 + j - 1].append(0)
        for i in range(len(y)):
            for j in range(1, len(phi_vector)):
                if phi_vector[j - 1] <= quantile_var[i] <= phi_vector[j]:
                    if treat_var[i] == 1:
                        copy_covariates[initial_length - 1 + j + len(phi_vector) - 2].append(
                            (self.quantile_identify_var[x_index][i] - (phi_vector[j - 1] + phi_vector[j]) / 2))
                    else:
                        copy_covariates[initial_length - 1 + j + len(phi_vector) - 2].append(0)
                else:
                    copy_covariates[initial_length - 1 + j + len(phi_vector) - 2].append(0)
        if self.Response_type == 'normal':
            spilt_OLS_result = OLS(y, add_constant(
                array(copy_covariates).T)).fit()
        elif self.Response_type == 'binary':
            spilt_OLS_result = Logit(y, add_constant(
                array(copy_covariates).T)).fit()
        else:
            Method_dict = {
                'possion': families.Poisson(),
                'binomial': families.Binomial()
            }
            GLM_model = GLM(self.y, add_constant(array(copy_covariates).T), family=Method_dict[self.Response_type])
            spilt_OLS_result = GLM_model.fit()
        sigmas=array(spilt_OLS_result.params[len(self.coviate_datas)+1 if self.constant==True else len(self.coviate_datas):])/array(spilt_OLS_result.tvalues[len(self.coviate_datas)+1 if self.constant==True else len(self.coviate_datas):])
        sigmas_2=array(spilt_OLS_result.params[:len(self.coviate_datas)+1 if self.constant==True else len(self.coviate_datas)])/array(spilt_OLS_result.tvalues[:len(self.coviate_datas)+1 if self.constant==True else len(self.coviate_datas)])
        return [spilt_OLS_result.params[len(self.coviate_datas)+1 if self.constant==True else len(self.coviate_datas):],
        sigmas,
                spilt_OLS_result.params[:len(self.coviate_datas)+1 if self.constant==True else len(self.coviate_datas)],
                sigmas_2]
    def D_get_mean_param(self,identify_var,quantile):
        mean_sum = float(0)
        for i in range(1, len(self.population_phi_vector[identify_var])):
            if self.population_phi_vector[identify_var][i - 1] <= quantile <= self.population_phi_vector[identify_var][
                i]:
                mean_sum += self.population_tree_OLSparams[identify_var][0][i - 1]

        return mean_sum
    def D_get_VAR_param(self,identify_var,quantile):
        mean_sum = float(0)
        for i in range(1, len(self.population_phi_vector[identify_var])):
            if self.population_phi_vector[identify_var][i - 1] <= quantile <= self.population_phi_vector[identify_var][
                i]:
                mean_sum += self.population_tree_OLSparams[identify_var][1][i - 1]**2

        return mean_sum

    def D_homegeneous_regression(self):
        for j in range(0, len(self.identify_var_datas)):
            self.population_tree(j)
        self.HG_plot_lists = [[[None for j in self.y], [None for j in self.y],[None for j in self.y]] for i in self.identify_var_datas]
        new_y = []

        for i_params_list in range(len(self.identify_var_datas)):
            for i in range(len(self.y)):
                if self.identify_var_datas[i_params_list].count(self.identify_var_datas[i_params_list][i]) > 1:
                    now_index = self.sorted_identify_var[i_params_list].index(
                        self.identify_var_datas[i_params_list][i])
                    self.HG_plot_lists[i_params_list][0][now_index] = self.identify_var_datas[i_params_list][now_index]
                    self.HG_plot_lists[i_params_list][1][now_index] = self.D_get_mean_param(identify_var=i_params_list,
                                                                                       quantile=
                                                                                       self.quantile_identify_var[
                                                                                           i_params_list][
                                                                                           now_index])
                    self.HG_plot_lists[i_params_list][2][now_index] = self.D_get_VAR_param(identify_var=i_params_list,
                                                                                       quantile=
                                                                                       self.quantile_identify_var[
                                                                                           i_params_list][
                                                                                           now_index])
                    for repeat_index in range(self.identify_var_datas[i_params_list].count(self.identify_var_datas[i_params_list][i])):
                        old_index=now_index
                        now_index += 1
                        self.HG_plot_lists[i_params_list][0][now_index] = self.identify_var_datas[i_params_list][now_index]
                        self.HG_plot_lists[i_params_list][1][now_index] = self.D_get_mean_param(identify_var=i_params_list,
                                                                                           quantile=
                                                                                           self.quantile_identify_var[
                                                                                               i_params_list][
                                                                                               old_index])
                        self.HG_plot_lists[i_params_list][2][now_index] = self.D_get_VAR_param(identify_var=i_params_list,
                                                                                          quantile=
                                                                                          self.quantile_identify_var[
                                                                                              i_params_list][
                                                                                              now_index])
                else:
                    now_index = self.sorted_identify_var[i_params_list].index(
                        self.identify_var_datas[i_params_list][i])
                    self.HG_plot_lists[i_params_list][0][now_index] = self.identify_var_datas[i_params_list][now_index]
                    self.HG_plot_lists[i_params_list][1][now_index] = self.D_get_mean_param(identify_var=i_params_list,
                                                                                       quantile=
                                                                                       self.quantile_identify_var[
                                                                                           i_params_list][now_index])
                    self.HG_plot_lists[i_params_list][2][now_index] = self.D_get_VAR_param(identify_var=i_params_list,
                                                                                      quantile=
                                                                                      self.quantile_identify_var[
                                                                                          i_params_list][
                                                                                          now_index])
        cal_resid_treatment = []
        cal_resid = []
        for i in range(len(self.y)):
            negative_E_sum_fz = float(0)
            for i_params_list in range(len(self.identify_var_datas)):
                negative_E_sum_fz += self.HG_plot_lists[i_params_list][1][i]
            new_y.append(self.y[i] - negative_E_sum_fz * self.coviate_datas[self.treat_var_index][i])
        self.cutparams_OLS_result = OLS(new_y, add_constant(
            array(self.coviate_datas).T)).fit()
        self.resid = self.cutparams_OLS_result.resid
        for i in range(len(self.y)):
            if self.coviate_datas[self.treat_var_index][i] == 1:
                cal_resid_treatment.append(self.resid[i] * self.resid[i])
            else:
                cal_resid.append(self.resid[i] * self.resid[i])
        self.estimated_var=abs(sum(cal_resid_treatment) / (len(cal_resid_treatment) - 1) - sum(cal_resid) / (len(cal_resid) - 1) )
        print("my estimated VAR:",
              self.estimated_var)
        return self.HG_plot_lists
    def HETEROGENEOUSITY_TEST(self):
        for INDEX_params in range(len(self.identify_var_datas)):
            try:
                WMW_test_VAR_p_value = BOOSTRAP_WMW_rank_test(self.HG_plot_lists[INDEX_params][2])
                pvalue=WMW_test_VAR_p_value.pvalue
            except ValueError:
                pvalue=0
            print("-" * 100)
            print("P-value:",pvalue)
            if pvalue>=self.confidence:
                print(INDEX_params+1,"th Heterogeneous variable's VAR unsatisfy confidence conditions." )
                print("P-value:", pvalue, "while confidence level=", self.confidence)
                try:
                    WMW_test_mean_p_value = BOOSTRAP_WMW_rank_test(self.HG_plot_lists[INDEX_params][1])
                    pvalue=WMW_test_mean_p_value.pvalue
                except ValueError:
                    print("All numbers are identical in mannwhitneyu")
                    pvalue = 0
                if pvalue>= self.confidence:
                    print(INDEX_params + 1, "th Heterogeneous variable's mean unsatisfy confidence conditions.")
                    L=Likelihood_test(self.HG_plot_lists[INDEX_params][1], self.HG_plot_lists[INDEX_params][2])
                    print("Likelihood Test pass" if L[1]<self.confidence else "Likelihood Test didn't pass")
                    print("Likelihood Statistics:",L[0],"P_value:",L[1],"while confidence level is ",self.confidence)
                else:
                    print(INDEX_params + 1, "th Heterogeneous variable's mean  satisfy confidence conditions.")
            else:
                print(INDEX_params + 1, "th Heterogeneous variable's VAR  satisfy confidence conditions.")
                print("P-value:", pvalue, "while confidence level=", self.confidence)
                try:
                    WMW_test_mean_p_value = BOOSTRAP_WMW_rank_test(self.HG_plot_lists[INDEX_params][1])
                    pvalue=WMW_test_mean_p_value.pvalue
                except ValueError:
                    print("All numbers are identical in mannwhitneyu")
                    pvalue = 0
                if pvalue>= self.confidence:
                    print(INDEX_params + 1, "th Heterogeneous variable's mean unsatisfy confidence conditions.")
                else:
                    print(INDEX_params + 1, "th Heterogeneous variable's mean  satisfy confidence conditions.")