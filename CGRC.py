import scipy.optimize as opt
import numpy
import math
from statsmodels.api import OLS,add_constant
import random
import time
#C:lm对1假设模型使用n*lr然后考虑到样本量因素，使用F统计量，上下不同自由度  对varlist[]使用1假设，加总形成wald统计量，同样使用llf假设，比较F统计量
from csv import reader

def random_multivar(select_list):
    selected_list=[select_list.copy().index(i) for i in select_list.copy()]
    a=min(selected_list)
    b=max(selected_list)
    random_ab=random.uniform(a,b)
    cal_list=[(i-random_ab)**2 for i in selected_list]
    return select_list[cal_list.index(min(cal_list))]

class CGRC(object):
    def __init__(self,y_datas,coviate_datas,classify_var_index,constant=True):
        self.data=y_datas
        self.data_2=[]
        self.invar=[]
        self.coviate_datas=[]
        self.coviate_datas=[k for k in coviate_datas]
        self.coviate_datas_t=[]
        self.constant=constant
        self.mean_y = numpy.mean(self.data)
        self.std_y = numpy.std(self.data)
        self.hess_inverse_data=[]
        self.classify_var_index=classify_var_index
        self.coviate_datas_del=self.coviate_datas.copy()
        del self.coviate_datas_del[len(self.coviate_datas_del)-classify_var_index:len(self.coviate_datas_del)]
        self.list_classifyvar_list=[]
        self.num_virtual_var=None
        '''
        计算给定的各个状态变量细分种类的个数
        '''
        for i in range(0,self.classify_var_index):
            temp_list=[]
            for k in range(0,len(self.data)):
                if self.coviate_datas[len(self.coviate_datas)-self.classify_var_index+i][k] in temp_list:
                    pass
                else:
                    temp_list.append(self.coviate_datas[len(self.coviate_datas)-self.classify_var_index+i][k])
            self.list_classifyvar_list.append(temp_list)
        '''
        生成新的解释变量矩阵，除了非状态变量直接照搬之外，还加入了新的多元同时成立的虚拟变量
        '''

        self.new_xlines = self.coviate_datas[:len(self.coviate_datas)-self.classify_var_index].copy()
        #得到总的需要新生成的虚拟变量的个数，并且将空的列表插入新的解释变量矩阵
        nums_newadd_var=1
        for index_classifyvar_in_covvars in self.list_classifyvar_list:
            nums_newadd_var=nums_newadd_var*len(index_classifyvar_in_covvars)
        self.num_virtual_var=nums_newadd_var
        for nums_index in range(0,nums_newadd_var):
            self.new_xlines.append([])
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
                                             len(temp_list_to_save_everyindex_in_classifyvar) - 1] + 1
            each_linear_value = 1 if constant == True else 0
            for each_covvar in range(0, len(self.coviate_datas) - self.classify_var_index):
                each_linear_value += self.coviate_datas[each_covvar][index_every_y]

            self.new_xlines[
                len(self.coviate_datas) - self.classify_var_index + get_index_every_y_in_line - 1].append(
                1 )
            for index_to_add_value_left in range(0, get_index_every_y_in_line - 1):
                self.new_xlines[len(self.coviate_datas) - self.classify_var_index + index_to_add_value_left].append(
                    0)
            for index_to_add_value_right in range(get_index_every_y_in_line, nums_newadd_var):
                self.new_xlines[
                    len(self.coviate_datas) - self.classify_var_index + index_to_add_value_right].append(0)
        if constant==True:
            self.coviate_datas_t = numpy.array(self.coviate_datas_del).T
            self.coviate_datas_t = add_constant(self.coviate_datas_t)
            self.ols_result_origin= OLS(self.data, self.coviate_datas_t).fit()
            self.new_xlines_t=numpy.array(self.new_xlines).T
            self.ols_result_after=OLS(self.data,add_constant(self.new_xlines_t)).fit()
            print(self.ols_result_origin.summary())
            print(self.ols_result_after.summary())
        else:
            print('Non constant')
            self.coviate_datas_t = numpy.array(self.coviate_datas_del)
            self.coviate_datas_t = self.coviate_datas_t.T
            self.ols_result = OLS(self.data, add_constant(self.coviate_datas_t),constant=False).fit()
        self.summary={
            'Wald test':None,
            'Loglikelihood test':None,
            't test':None
        }

    def CGRC_Leastsquare_func(self,varlist):
        jac_CGRC_Leastsquare_func = numpy.zeros(len(varlist))
        #从每一行开始对对应的虚拟变量赋值
        # 求在第i(i<=数据总长度)行需要添加为1的虚拟变量的具体下标
        nums_newadd_var=1
        for index_classifyvar_in_covvars in self.list_classifyvar_list:
            nums_newadd_var=nums_newadd_var*len(index_classifyvar_in_covvars)
        total_residual=0
        for index_every_y in range(0, len(self.data)):
            temp_list_to_save_everyindex_in_classifyvar = []  # 储存每个状态变量下的该行联合虚拟变量对应的下标
            for index_to_panelvar in range(0, self.classify_var_index):
                temp_list_to_save_everyindex_in_classifyvar.append(
                    self.list_classifyvar_list[index_to_panelvar].index(
                        self.coviate_datas[len(self.coviate_datas) - self.classify_var_index + index_to_panelvar][
                            index_every_y]))
            get_index_every_y_in_line = 0#对应每行虚拟变量的坐标

            # 计算第i行的对应虚拟变量的坐标
            for index_everyu_po in range(0, len(temp_list_to_save_everyindex_in_classifyvar) - 1):
                temp_value = 1
                for index_every_po_next in range(index_everyu_po + 1,
                                                 len(temp_list_to_save_everyindex_in_classifyvar)):
                    temp_value = temp_value * (len(self.list_classifyvar_list[index_every_po_next]))
                get_index_every_y_in_line += temp_list_to_save_everyindex_in_classifyvar[
                                                 index_everyu_po] * temp_value
            get_index_every_y_in_line += temp_list_to_save_everyindex_in_classifyvar[
                                             len(temp_list_to_save_everyindex_in_classifyvar) - 1] + 1

            #计算外生解释变量与回归系数的线性组合
            linear_value=varlist[0] if self.constant==True else 0#除了虚拟变量和状态变量之外的外生线性解释变量与回归系数的线性组合
            if self.constant==True:
                for index_independant_var in range(0, len(self.coviate_datas) - self.classify_var_index):
                    linear_value += self.coviate_datas[index_independant_var][index_every_y] * varlist[index_independant_var+1]
                total_residual += (self.data[index_every_y] - varlist[len(self.coviate_datas) - self.classify_var_index + get_index_every_y_in_line - 1] * linear_value) ** 2
                jac_CGRC_Leastsquare_func[0] +=2* ((self.data[index_every_y] - varlist[
                    len(self.coviate_datas) - self.classify_var_index + get_index_every_y_in_line - 1] * linear_value)
                                                                          * (- varlist[
                            len(self.coviate_datas) - self.classify_var_index + get_index_every_y_in_line - 1]))
                for index_every_independantvar in range(0, len(self.coviate_datas) - self.classify_var_index):
                    jac_CGRC_Leastsquare_func[index_every_independantvar+1] +=2* ((self.data[index_every_y] - varlist[len(
                        self.coviate_datas) - self.classify_var_index + get_index_every_y_in_line - 1] * linear_value)
                                                                              * (- varlist[
                                len(self.coviate_datas) - self.classify_var_index + get_index_every_y_in_line - 1]) *
                                                                              self.coviate_datas[index_every_independantvar][index_every_y])
            else:
                for index_independant_var in range(0, len(self.coviate_datas) - self.classify_var_index):
                    linear_value += self.coviate_datas[index_independant_var][index_every_y] * varlist[index_independant_var]
                total_residual += (self.data[index_every_y] - varlist[len(self.coviate_datas) - self.classify_var_index + get_index_every_y_in_line - 1] * linear_value) ** 2
                for index_every_independantvar in range(0, len(self.coviate_datas) - self.classify_var_index):
                    jac_CGRC_Leastsquare_func[index_every_independantvar] += 2*((self.data[index_every_y] - varlist[len(
                        self.coviate_datas) - self.classify_var_index + get_index_every_y_in_line - 1] * linear_value)
                                                                              * (- varlist[
                                len(self.coviate_datas) - self.classify_var_index + get_index_every_y_in_line - 1]) *
                                                                              self.coviate_datas[
                                                                                  index_every_independantvar][
                                                                                  index_every_y])

            for index_every_virtualvar in range(len(self.coviate_datas) - self.classify_var_index,len(varlist)):
                jac_CGRC_Leastsquare_func[index_every_virtualvar]+=(2*((self.data[index_every_y] - varlist[len(self.coviate_datas) - self.classify_var_index + get_index_every_y_in_line - 1] * linear_value)*(-linear_value))+2*varlist[index_every_virtualvar])
        for index_every_virtualvar in range(len(self.coviate_datas) - self.classify_var_index, len(varlist)):
            total_residual += varlist[index_every_virtualvar] ** 2
        #print(total_residual)
        return total_residual,numpy.array(jac_CGRC_Leastsquare_func)

    def CGRC_Regress_LS(self):
        initial_params=list(self.ols_result_origin.params)[:len(self.coviate_datas)-self.classify_var_index+1] if self.constant==True else list(self.ols_result_origin.params)[:len(self.coviate_datas)-self.classify_var_index]
        for  i in range(0,self.num_virtual_var):
            initial_params.append(1)
        result_regress=opt.basinhopping(self.CGRC_Leastsquare_func,initial_params,disp=True,niter=20,minimizer_kwargs={'jac':True})
        print(result_regress)
        return result_regress
i=666

counts = 1
x_0 = []
x_1 = []
x_2 = []
x_3 = []
x_4 = []
x_5 = []
x_6=[]
x_7=[]
x_8=[]
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
disu_param2 = random.uniform(100,1000)
print("U的分布参数:%f | %f" %(disu_param1,disu_param2))

dise_param = 1
x0 = random.normalvariate(dis0_param1, dis0_param2)
b1 = random.normalvariate(disb1_param1, disb1_param2)
b2 = random.normalvariate(disb2_param1, disb2_param2)
b3 = random.normalvariate(disb3_param1, disb3_param2)
b4 = random.normalvariate(disb4_param1, disb4_param2)
b5 = random.normalvariate(disb5_param1, disb5_param2)
dict_shock_x6 = {
        0: 0.5,
        1: 1
    }
dict_shock_x7 = {
        'mike': 1,
        'joe': 20,
        'eyic': 30,
        'go': 40
    }
dict_shock_x8 = {
        1997:10.5,
        1998: 11.0,
        1999: 11.5,
        2000: 11.3,
        2001: 11.5
    }
while counts <i:
    x1 = random.normalvariate(dis1_param1, dis1_param2)
    x2 = random.betavariate(dis2_param1, dis2_param2) * random.normalvariate(dis2_param3, dis2_param4)
    x3 = random.gammavariate(1, dis3_param1)  # 指数分布
    x4 = random.gammavariate(dis4_param1, dis4_param2)
    x5 = random.uniform(dis5_param1, dis5_param2)
    x6=random_multivar([0,1])
    x7=random_multivar(['mike','joe','eyic','go'])
    x8=random_multivar([1997,1998,1999,2000,2001])
    e = random.normalvariate(0, dise_param)
    linear_value = x0 + b1 * x1 + b2 * x2 + b3 * x3 + b4 * x4 + b5 * x5+e
    U = random.normalvariate(disu_param1, disu_param2)
    x_1.append(x1)
    x_2.append(x2)
    x_3.append(x3)
    x_4.append(x4)
    x_5.append(x5)
    x_6.append(x6)
    x_7.append(x7)
    x_8.append(x8)
    y.append(linear_value*dict_shock_x6[x6]*dict_shock_x7[x7]*dict_shock_x8[x8])
    counts += 1
begin_time=time.time()
ops = CGRC(y, [x_1, x_2, x_3, x_4, x_5,x_6,x_7,x_8],3, constant=True)
result=ops.CGRC_Regress_LS()
print("迭代完成")
print(result.x)
print("一共用了%f秒" % (time.time()-begin_time))
print("%f|%f|%f|%f|%f|%f"%(x0,b1,b2,b3,b4,b5))
print(result.x[:len(ops.coviate_datas)-ops.classify_var_index+1]-ops.ols_result_origin.params[:len(ops.coviate_datas)-ops.classify_var_index+1])
print(result.x[:len(ops.coviate_datas)-ops.classify_var_index+1]-ops.ols_result_after.params[:len(ops.coviate_datas)-ops.classify_var_index+1])
