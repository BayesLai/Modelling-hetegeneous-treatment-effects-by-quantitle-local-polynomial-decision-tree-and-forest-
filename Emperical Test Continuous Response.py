from csv import reader,writer
from QLPR import QLPRT,QLPRF
from math import sqrt
from matplotlib.pyplot import subplot,plot,show
treat=[]
age=[]
education=[]
black=[]
hispanic=[]
married=[]
nodegree=[]
re75=[]
wage_gap=[]
upper_bound_QLPRT = []
downer_bound_QLPRT = []
Estimate_QLPRT = []
upper_bound_QLPRF = []
downer_bound_QLPRF = []
Estimate_QLPRF = []
with open("NSW-data.csv",'r')as myfile:
    csvreader=reader(myfile)
    for i in list(csvreader)[1:]:
        treat.append(int(i[1]))
        age.append(float(i[2]))
        education.append(float(i[3]))
        black.append(int(i[4]))
        hispanic.append(float(i[5]))
        married.append(float(i[6]))
        nodegree.append(float(i[7]))
        re75.append(float(i[8]))
        wage_gap.append(float(i[10]))
he_varlist=[education,age]
QLPRT_result=QLPRT(wage_gap,[treat,black,hispanic,married,nodegree],he_varlist,treat_var_index=1)
QLPRT_DHG= QLPRT_result.DHG
QLPRF_result=QLPRF(wage_gap,[treat,black,hispanic,married,nodegree],he_varlist,treat_var_index=1)
QLPRF_DHG= QLPRF_result.DHG

for i in range(len(treat)):
    Estimate_QLPRT.append(QLPRT_result.cutparams_OLS_result.params[1]+QLPRT_DHG[0][1][i])
    upper_bound_QLPRT.append(QLPRT_result.cutparams_OLS_result.params[1]+QLPRT_DHG[0][1][i]+1.96*sqrt(QLPRT_result.estimated_var))
    downer_bound_QLPRT.append(QLPRT_result.cutparams_OLS_result.params[1]+QLPRT_DHG[0][1][i]-1.96*sqrt(QLPRT_result.estimated_var))
    Estimate_QLPRF.append(QLPRF_result.cutparams_OLS_result.params[1] + QLPRF_DHG[0][1][i] )
    upper_bound_QLPRF.append(
        QLPRF_result.cutparams_OLS_result.params[1] + QLPRF_DHG[0][1][i] + 1.96 * sqrt(
            QLPRF_result.estimated_var))
    downer_bound_QLPRF.append(
        QLPRF_result.cutparams_OLS_result.params[1] + QLPRF_DHG[0][1][i]  - 1.96 * sqrt(
            QLPRF_result.estimated_var))
index_list=[i+1 for i in range(0,len(treat))]
for j in range(len(he_varlist)):

    subplot(len(he_varlist),1,j+1)
    plot(QLPRT_DHG[j][0],[i +QLPRT_result.cutparams_OLS_result.params[1] for i in QLPRT_DHG[j][1]],'ro')
    file_name = "Continuous" + "-QLPRT-" + str(j + 1)+'.csv'
    with open(file_name,'a')as myfile:
        csvwriter=writer(myfile)
        for index_inlist in range(len(QLPRT_DHG[j][0])):
            csvwriter.writerow([QLPRT_DHG[j][0][index_inlist],QLPRT_DHG[j][1][index_inlist]])
show()
for j in range(len(he_varlist)):
    subplot(len(he_varlist),1,j+1)
    plot(QLPRF_DHG[j][0],[i +QLPRF_result.cutparams_OLS_result.params[1] for i in QLPRF_DHG[j][1]],'ro')
    file_name = "Continuous" + "-QLPRF-" + str(j + 1)+'.csv'
    with open(file_name, 'a')as myfile:
        csvwriter = writer(myfile)
        for index_inlist in range(len(QLPRF_DHG[j][0])):
            csvwriter.writerow([QLPRF_DHG[j][0][index_inlist], QLPRF_DHG[j][1][index_inlist]])
show()
subplot(2,1,1)
plot(index_list,Estimate_QLPRT)
plot(index_list,upper_bound_QLPRT)
plot(index_list,downer_bound_QLPRT)
subplot(2,1,2)
plot(index_list,Estimate_QLPRF)
plot(index_list,upper_bound_QLPRF)
plot(index_list,downer_bound_QLPRF)
show()
