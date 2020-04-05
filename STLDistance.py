from STL import STLFormula
from STLpulp import *
import operator as operatorclass
import pulp as plp
import numpy as np
import math

#CONSTANTS
INDEX_X = 0
INDEX_Y = 1
M_up = 100000
M_low = 0.000001





def directed_pompeiu_hausdorff_distance(phi1,phi2,rand_area):

    dict_vars = {}
  
    #objective, maximize epsilon
    epsilon = plp.LpVariable(name='epsilon',cat='Continuous',lowBound=0)
    
    #satisfaction of phi1 and phi2
    zvar1 = plp.LpVariable('z1_'+str(id(phi1))+'_t_'+str(phi1.horizon),cat='Binary')
    zvar2 = plp.LpVariable('z2_'+str(id(phi2))+'_t_'+str(phi2.horizon),cat='Binary')
    dict_vars['z1_'+str(id(phi1))+'_t_'+str(phi1.horizon)] = zvar1
    dict_vars['z2_'+str(id(phi2))+'_t_'+str(phi2.horizon)] = zvar2

    opt_model = plp.LpProblem("MIP Model", plp.LpMaximize)
    opt_model += epsilon
  
    dict_vars['epsilon'] = epsilon

            
    #We want to optimize a signal. The lower and upperbounds are specified by the random area.
    s = plp.LpVariable.dicts("s",(range(max(phi1.horizon,phi2.horizon)+1),range(2)),rand_area[0],rand_area[1],plp.LpContinuous)    
       
       
    #recursive function
    def model_phi1(phi,t,opt_model):
        if isinstance(phi, STLFormula.TrueF):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 1
        elif isinstance(phi, STLFormula.Negation):
            model_phi1(phi.formula,t,opt_model)
            try:
                zvar1 = dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t)]
            except KeyError:
                zvar1 = plp.LpVariable('z1_'+str(id(phi.formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t)] = zvar1 
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 1-zvar1
        elif isinstance(phi, STLFormula.Predicate):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            if phi.operator == operatorclass.gt or  phi.operator == operatorclass.ge:
                opt_model += s[t][phi.pi_index_signal] - phi.mu <= M_up*zvar-M_low
                opt_model += -(s[t][phi.pi_index_signal] - phi.mu) <= M_up*(1-zvar)-M_low
            else:
                opt_model += -s[t][phi.pi_index_signal] + phi.mu <= M_up*zvar-M_low
                opt_model += -(-s[t][phi.pi_index_signal] + phi.mu) <= M_up*(1-zvar)-M_low
        elif isinstance(phi, STLFormula.Conjunction):
            model_phi1(phi.first_formula,t,opt_model)
            model_phi1(phi.second_formula,t,opt_model)
            try:
                zvar1 = dict_vars['z1_'+str(id(phi.first_formula))+'_t_'+str(t)]
            except KeyError:
                zvar1 = plp.LpVariable('z1_'+str(id(phi.first_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi.first_formula))+'_t_'+str(t)] = zvar1       
            try:
                zvar2 = dict_vars['z1_'+str(id(phi.second_formula))+'_t_'+str(t)]
            except KeyError:
                zvar2 = plp.LpVariable('z1_'+str(id(phi.second_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi.second_formula))+'_t_'+str(t)] = zvar2
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar <= zvar1
            opt_model += zvar <= zvar2
            opt_model += zvar >= 1-2+zvar1+zvar2
        elif isinstance(phi, STLFormula.Disjunction):
            model_phi1(phi.first_formula,t,opt_model)
            model_phi1(phi.second_formula,t,opt_model)
            try:
                zvar1 = dict_vars['z1_'+str(id(phi.first_formula))+'_t_'+str(t)]
            except KeyError:
                zvar1 = plp.LpVariable('z1_'+str(id(phi.first_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi.first_formula))+'_t_'+str(t)] = zvar1       
            try:
                zvar2 = dict_vars['z1_'+str(id(phi.second_formula))+'_t_'+str(t)]
            except KeyError:
                zvar2 = plp.LpVariable('z1_'+str(id(phi.second_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi.second_formula))+'_t_'+str(t)] = zvar2
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar >= zvar1
            opt_model += zvar >= zvar2
            opt_model += zvar <= zvar1+zvar2
        elif isinstance(phi,STLFormula.Always):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            for t_i in range(phi.t1,phi.t2+1):
                model_phi1(phi.formula,t_i,opt_model)
                try:
                    zvar_i = dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)]
                except KeyError:
                    zvar_i = plp.LpVariable('z1_'+str(id(phi.formula))+'_t_'+str(t_i),cat='Binary')
                    dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] = zvar_i
                opt_model += zvar <= zvar_i
            opt_model += zvar >= 1 - (phi.t2+1-phi.t1) + plp.lpSum([dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(phi.t1,phi.t2+1)])
        elif isinstance(phi,STLFormula.Eventually):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            for t_i in range(phi.t1,phi.t2+1):
                model_phi1(phi.formula,t_i,opt_model)
                try:
                    zvar_i = dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)]
                except KeyError:
                    zvar_i = plp.LpVariable('z1_'+str(id(phi.formula))+'_t_'+str(t_i),cat='Binary')
                    dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] = zvar_i
                opt_model += zvar >= zvar_i
            opt_model += zvar <= plp.lpSum([dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(phi.t1,phi.t2+1)])
    
    def model_phi2(phi,t,opt_model):
        if isinstance(phi, STLFormula.TrueF):
            try:
                zvar = dict_vars['z2_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z2_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 1
        elif isinstance(phi, STLFormula.Negation):
            model_phi2(phi.formula,t,opt_model)
            try:
                zvar1 = dict_vars['z2_'+str(id(phi.formula))+'_t_'+str(t)]
            except KeyError:
                zvar1 = plp.LpVariable('z2_'+str(id(phi.formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi.formula))+'_t_'+str(t)] = zvar1 
            try:
                zvar = dict_vars['z2_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z2_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 1-zvar1
        elif isinstance(phi, STLFormula.Predicate):
            try:
                zvar = dict_vars['z2_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z2_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi))+'_t_'+str(t)] = zvar
            if phi.operator == operatorclass.gt or phi.operator == operatorclass.ge:
                opt_model += s[t][phi.pi_index_signal] - phi.mu - epsilon <= M_up*zvar-M_low
                opt_model += -(s[t][phi.pi_index_signal] - phi.mu) - epsilon <= M_up*(1-zvar)-M_low
            else:
                opt_model += -s[t][phi.pi_index_signal] + phi.mu + epsilon <= M_up*zvar-M_low
                opt_model += -(-s[t][phi.pi_index_signal] + phi.mu) + epsilon <= M_up*(1-zvar)-M_low
        elif isinstance(phi, STLFormula.Conjunction):
            model_phi2(phi.first_formula,t,opt_model)
            model_phi2(phi.second_formula,t,opt_model)
            try:
                zvar1 = dict_vars['z2_'+str(id(phi.first_formula))+'_t_'+str(t)]
            except KeyError:
                zvar1 = plp.LpVariable('z2_'+str(id(phi.first_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi.first_formula))+'_t_'+str(t)] = zvar1       
            try:
                zvar2 = dict_vars['z2_'+str(id(phi.second_formula))+'_t_'+str(t)]
            except KeyError:
                zvar2 = plp.LpVariable('z2_'+str(id(phi.second_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi.second_formula))+'_t_'+str(t)] = zvar2
            try:
                zvar = dict_vars['z2_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z2_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar <= zvar1
            opt_model += zvar <= zvar2
            opt_model += zvar >= 1-2+zvar1+zvar2
        elif isinstance(phi, STLFormula.Disjunction):
            model_phi2(phi.first_formula,t,opt_model)
            model_phi2(phi.second_formula,t,opt_model)
            try:
                zvar1 = dict_vars['z2_'+str(id(phi.first_formula))+'_t_'+str(t)]
            except KeyError:
                zvar1 = plp.LpVariable('z2_'+str(id(phi.first_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi.first_formula))+'_t_'+str(t)] = zvar1       
            try:
                zvar2 = dict_vars['z2_'+str(id(phi.second_formula))+'_t_'+str(t)]
            except KeyError:
                zvar2 = plp.LpVariable('z2_'+str(id(phi.second_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi.second_formula))+'_t_'+str(t)] = zvar2
            try:
                zvar = dict_vars['z2_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z2_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar >= zvar1
            opt_model += zvar >= zvar2
            opt_model += zvar <= zvar1+zvar2
        elif isinstance(phi,STLFormula.Always):
            try:
                zvar = dict_vars['z2_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z2_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi))+'_t_'+str(t)] = zvar
            for t_i in range(phi.t1,phi.t2+1):
                model_phi2(phi.formula,t_i,opt_model)
                try:
                    zvar_i = dict_vars['z2_'+str(id(phi.formula))+'_t_'+str(t_i)]
                except KeyError:
                    zvar_i = plp.LpVariable('z2_'+str(id(phi.formula))+'_t_'+str(t_i),cat='Binary')
                    dict_vars['z2_'+str(id(phi.formula))+'_t_'+str(t_i)] = zvar_i
                opt_model += zvar <= zvar_i
            opt_model += zvar >= 1 - (phi.t2+1-phi.t1) + plp.lpSum([dict_vars['z2_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(phi.t1,phi.t2+1)])
        elif isinstance(phi,STLFormula.Eventually):
            try:
                zvar = dict_vars['z2_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z2_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi))+'_t_'+str(t)] = zvar
            for t_i in range(phi.t1,phi.t2+1):
                model_phi2(phi.formula,t_i,opt_model)
                try:
                    zvar_i = dict_vars['z2_'+str(id(phi.formula))+'_t_'+str(t_i)]
                except KeyError:
                    zvar_i = plp.LpVariable('z2_'+str(id(phi.formula))+'_t_'+str(t_i),cat='Binary')
                    dict_vars['z2_'+str(id(phi.formula))+'_t_'+str(t_i)] = zvar_i
                opt_model += zvar >= zvar_i
            opt_model += zvar <= plp.lpSum([dict_vars['z2_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(phi.t1,phi.t2+1)])
    
    model_phi1(phi1,phi1.horizon,opt_model)
    model_phi2(phi2,phi2.horizon,opt_model)

    zvar1 = dict_vars['z1_'+str(id(phi1))+'_t_'+str(phi1.horizon)]
    zvar2 = dict_vars['z2_'+str(id(phi2))+'_t_'+str(phi2.horizon)]
    
    opt_model += zvar1 == 1
    opt_model += zvar2 == 0
        
    opt_model.solve(plp.GUROBI_CMD(msg=False))
    
    # print( [[s[j][i].varValue for i in range(2)] for j in range(max(phi1.horizon,phi2.horizon)+1)] )

    if epsilon.varValue == None:
        return 0.0
    return epsilon.varValue





def pompeiu_hausdorff_distance(phi1,phi2,rand_area):
    return round(max(directed_pompeiu_hausdorff_distance(phi1,phi2,rand_area),directed_pompeiu_hausdorff_distance(phi2,phi1,rand_area)),5)






def test():
    #test metrics
    predicate_x_ge02 = STLFormula.Predicate('x',operatorclass.ge,0.2,INDEX_X)
    predicate_x_lt04 = STLFormula.Predicate('x',operatorclass.le,0.4,INDEX_X)
    predicate_x_lt044 = STLFormula.Predicate('x',operatorclass.le,0.44,INDEX_X)

    phi1 = STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,20)
    phi2 = STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt044),0,20)
    phi3 = STLFormula.Eventually(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,20)
    phi4 = STLFormula.Conjunction(STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,20),STLFormula.Eventually(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt044),0,20))
    phi5 = STLFormula.Conjunction(STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,10),STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt044),12,20))
    phi6 = STLFormula.Always(STLFormula.Eventually(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,4),0,16)

    # print(phi1,"\\\\")
    # print(phi2,"\\\\")
    # print(phi3,"\\\\")
    # print(phi4,"\\\\")
    # print(phi5,"\\\\")
    # print(phi6,"\\\\")

    predicate_x_gt3 = STLFormula.Predicate('x',operatorclass.gt,3,INDEX_X)
    predicate_x_le4 = STLFormula.Predicate('x',operatorclass.le,4,INDEX_X)
    predicate_y_gt2 = STLFormula.Predicate('y',operatorclass.gt,2,INDEX_Y)
    predicate_y_le3 = STLFormula.Predicate('y',operatorclass.le,3,INDEX_Y)
    eventually = STLFormula.Eventually(STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt3,predicate_x_le4) , STLFormula.Conjunction(predicate_y_gt2,predicate_y_le3) ),0,30)
    predicate_x_gt1 = STLFormula.Predicate('x',operatorclass.gt,1,INDEX_X)
    predicate_x_le2 = STLFormula.Predicate('x',operatorclass.le,2,INDEX_X)
    always = STLFormula.Always( STLFormula.Negation( STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt1,predicate_x_le2) , STLFormula.Conjunction(predicate_y_gt2,predicate_y_le3) ) ),0,30) 
    phi = STLFormula.Conjunction(eventually,always)
    phi_prime = STLFormula.Conjunction(phi,STLFormula.Always( STLFormula.Negation( STLFormula.Conjunction( STLFormula.Conjunction(STLFormula.Predicate('x',operatorclass.gt,1,INDEX_X),STLFormula.Predicate('x',operatorclass.lt,2,INDEX_X)) , STLFormula.Conjunction(STLFormula.Predicate('y',operatorclass.gt,0.5,INDEX_Y),STLFormula.Predicate('y',operatorclass.lt,1.5,INDEX_Y)) ) ),0,30) )
    phi1_nnf = STLFormula.toNegationNormalForm(phi,False)
    phi2_nnf = STLFormula.toNegationNormalForm(phi_prime,False)


    rand_area = [-2,4]

    print(pompeiu_hausdorff_distance(phi1_nnf,phi2_nnf,rand_area))
    print(pompeiu_hausdorff_distance(STLFormula.TrueF(),phi2_nnf,rand_area))

    print(directed_pompeiu_hausdorff_distance(phi2,phi1,rand_area),directed_pompeiu_hausdorff_distance(phi1,phi2,rand_area))
    print(directed_pompeiu_hausdorff_distance(phi3,phi1,rand_area),directed_pompeiu_hausdorff_distance(phi1,phi3,rand_area))
    print(directed_pompeiu_hausdorff_distance(phi1,phi1,rand_area))
    
# test()