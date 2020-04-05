from STL import STLFormula
import operator as operatorclass
import pulp as plp
import matplotlib.pyplot as plt
import numpy as np
import random


#CONSTANTS
INDEX_X = 0
INDEX_Y = 1
M = 1000
M_up = 100000
M_low = 0.000001


#Control policy, i.e. how much can move in 1 time stamp
U = 0.2


def generate_trajectory_milp(phi,start,rand_area,OPTIMIZE_ROBUSTNESS):
    
    dict_vars = {}
  
    #objective, maximize robustness
    rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(phi.horizon),cat='Continuous')
    dict_vars['r_'+str(id(phi))+'_t_'+str(phi.horizon)] = rvar
            
    #Initialize model
    if OPTIMIZE_ROBUSTNESS:
        opt_model = plp.LpProblem("MIP Model", plp.LpMaximize)
        opt_model += rvar
    else:
        opt_model = plp.LpProblem("MIP Model")
    

    #We want to optimize a signal. The lower and upperbounds are specified by the random area.
    s = plp.LpVariable.dicts("s",(range(phi.horizon+1),range(2)),rand_area[0],rand_area[1],plp.LpContinuous)

    #the start is specified
    opt_model += s[0][0] == start[0]
    opt_model += s[0][1] == start[1]
    
    #control policy
    for t in range(0,phi.horizon):
        opt_model += s[t+1][0]-s[t][0] <= random.uniform(U-0.1,U+0.1)
        opt_model += -(s[t+1][0]-s[t][0]) <= random.uniform(U-0.1,U+0.1)
        opt_model += s[t+1][1]-s[t][1] <= random.uniform(U-0.1,U+0.1)
        opt_model += -(s[t+1][1]-s[t][1]) <= random.uniform(U-0.1,U+0.1)
        
        
    #recursive function
    def model_phi(phi,t,opt_model):
        if isinstance(phi, STLFormula.Predicate):
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            if phi.operator == operatorclass.gt or  phi.operator == operatorclass.ge:
                opt_model += s[t][phi.pi_index_signal] - phi.mu == rvar
            else:
                opt_model += -s[t][phi.pi_index_signal] + phi.mu == rvar
            
        elif isinstance(phi, STLFormula.Conjunction):
            model_phi(phi.first_formula,t,opt_model)
            model_phi(phi.second_formula,t,opt_model)
            
            try:
                pvar1 = dict_vars['p_'+str(id(phi.first_formula))+'_t_'+str(t)]
            except KeyError:
                pvar1 = plp.LpVariable('p_'+str(id(phi.first_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['p_'+str(id(phi.first_formula))+'_t_'+str(t)] = pvar1       
            try:
                pvar2 = dict_vars['p_'+str(id(phi.second_formula))+'_t_'+str(t)]
            except KeyError:
                pvar2 = plp.LpVariable('p_'+str(id(phi.second_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['p_'+str(id(phi.second_formula))+'_t_'+str(t)] = pvar2
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            
            opt_model += pvar1+pvar2 == 1 #(3)
            opt_model += rvar <= dict_vars['r_'+str(id(phi.first_formula))+'_t_'+str(t)] #(4)
            opt_model += rvar <= dict_vars['r_'+str(id(phi.second_formula))+'_t_'+str(t)] #(4)
            opt_model += dict_vars['r_'+str(id(phi.first_formula))+'_t_'+str(t)] - (1 - pvar1)*M <= rvar <= dict_vars['r_'+str(id(phi.first_formula))+'_t_'+str(t)] + (1 - pvar1)*M #(5)
            opt_model += dict_vars['r_'+str(id(phi.second_formula))+'_t_'+str(t)] - (1 - pvar2)*M <= rvar <= dict_vars['r_'+str(id(phi.second_formula))+'_t_'+str(t)] + (1 - pvar2)*M #(5)
            
        elif isinstance(phi, STLFormula.Disjunction):
            model_phi(phi.first_formula,t,opt_model)
            model_phi(phi.second_formula,t,opt_model)
            
            try:
                pvar1 = dict_vars['p_'+str(id(phi.first_formula))+'_t_'+str(t)]
            except KeyError:
                pvar1 = plp.LpVariable('p_'+str(id(phi.first_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['p_'+str(id(phi.first_formula))+'_t_'+str(t)] = pvar1       
            try:
                pvar2 = dict_vars['p_'+str(id(phi.second_formula))+'_t_'+str(t)]
            except KeyError:
                pvar2 = plp.LpVariable('p_'+str(id(phi.second_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['p_'+str(id(phi.second_formula))+'_t_'+str(t)] = pvar2
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            
            opt_model += pvar1+pvar2 == 1 #(3)
            opt_model += rvar >= dict_vars['r_'+str(id(phi.first_formula))+'_t_'+str(t)] #(4)
            opt_model += rvar >= dict_vars['r_'+str(id(phi.second_formula))+'_t_'+str(t)] #(4)
            opt_model += dict_vars['r_'+str(id(phi.first_formula))+'_t_'+str(t)] - (1 - pvar1)*M <= rvar <= dict_vars['r_'+str(id(phi.first_formula))+'_t_'+str(t)] + (1 - pvar1)*M #(5)
            opt_model += dict_vars['r_'+str(id(phi.second_formula))+'_t_'+str(t)] - (1 - pvar2)*M <= rvar <= dict_vars['r_'+str(id(phi.second_formula))+'_t_'+str(t)] + (1 - pvar2)*M #(5)

        elif isinstance(phi,STLFormula.Always):
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            for t_i in range(phi.t1,phi.t2+1):
                model_phi(phi.formula,t_i,opt_model)
                     
                try:
                    pvar_i = dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t_i)]
                except KeyError:
                    pvar_i = plp.LpVariable('p_'+str(id(phi.formula))+'_t_'+str(t_i),cat='Binary')
                    dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t_i)] = pvar_i
                    
                opt_model += rvar <= dict_vars['r_'+str(id(phi.formula))+'_t_'+str(t_i)] #(4)
                opt_model += dict_vars['r_'+str(id(phi.formula))+'_t_'+str(t_i)] - (1 - pvar_i)*M <= rvar <= dict_vars['r_'+str(id(phi.formula))+'_t_'+str(t_i)] + (1 - pvar_i)*M #(5)
            opt_model += plp.lpSum([dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(phi.t1,phi.t2+1)]) == 1 #(3)
            
        elif isinstance(phi,STLFormula.Eventually):
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            for t_i in range(phi.t1,phi.t2+1):
                model_phi(phi.formula,t_i,opt_model)
                
                try:
                    pvar_i = dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t_i)]
                except KeyError:
                    pvar_i = plp.LpVariable('p_'+str(id(phi.formula))+'_t_'+str(t_i),cat='Binary')
                    dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t_i)] = pvar_i
                    
                opt_model += rvar >= dict_vars['r_'+str(id(phi.formula))+'_t_'+str(t_i)] #(4)
                opt_model += dict_vars['r_'+str(id(phi.formula))+'_t_'+str(t_i)] - (1 - pvar_i)*M <= rvar <= dict_vars['r_'+str(id(phi.formula))+'_t_'+str(t_i)] + (1 - pvar_i)*M #(5)
            opt_model += plp.lpSum([dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(phi.t1,phi.t2+1)]) == 1 #(3)
            
        elif isinstance(phi,STLFormula.Negation):
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            model_phi(phi.formula,t,opt_model)
            try:
                rvar_i = dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t)]
            except KeyError:
                rvar_i = plp.LpVariable('p_'+str(id(phi.formula))+'_t_'+str(t),cat='Binary')
                dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t)] = rvar_i
            opt_model += rvar == -rvar_i
    
    
    model_phi(phi,phi.horizon,opt_model)
    rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(phi.horizon)]
    opt_model += rvar >= 0 
    
    # opt_model.solve()
    opt_model.solve(plp.GUROBI_CMD(msg=False))

    # return np.array([[s[j][i].varValue for i in range(2)] for j in range(phi.horizon+1)])
    if s[0][0].varValue == None:
        raise Exception("")
    
    return [[s[j][i].varValue for i in range(2)] for j in range(phi.horizon+1)]
    
    
    


def generate_trajectory_boolean_milp(phi,start,rand_area):
    dict_vars = {}
  
    #satisfaction of phi1 and phi2
    zvar1 = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(phi.horizon),cat='Binary')
    dict_vars['z1_'+str(id(phi))+'_t_'+str(phi.horizon)] = zvar1

    opt_model = plp.LpProblem("MIP Model")
  
            
    #We want to optimize a signal. The lower and upperbounds are specified by the random area.
    s = plp.LpVariable.dicts("s",(range(phi.horizon+1),range(2)),rand_area[0],rand_area[1],plp.LpContinuous)
       
    #the start is specified
    opt_model += s[0][0] == start[0]
    opt_model += s[0][1] == start[1]
    
    #control policy
    for t in range(0,phi.horizon):
        opt_model += s[t+1][0]-s[t][0] <= random.uniform(U-0.1,U+0.1)
        opt_model += -(s[t+1][0]-s[t][0]) <= random.uniform(U-0.1,U+0.1)
        opt_model += s[t+1][1]-s[t][1] <= random.uniform(U-0.1,U+0.1)
        opt_model += -(s[t+1][1]-s[t][1]) <= random.uniform(U-0.1,U+0.1)   
       
    #recursive function
    def model_phi1(phi,t,opt_model):
        if isinstance(phi, STLFormula.TrueF):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 1
        if isinstance(phi, STLFormula.Predicate):
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
                    dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] = pvar_i
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
                    dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] = pvar_i
                opt_model += zvar >= zvar_i
            opt_model += zvar <= plp.lpSum([dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(phi.t1,phi.t2+1)])
    
    model_phi1(phi,phi.horizon,opt_model)
    
    opt_model += zvar1 == 1
        
    opt_model.solve(plp.GUROBI_CMD(msg=False))
    
    return [[s[j][i].varValue for i in range(2)] for j in range(phi.horizon+1)]

    
    
    
    
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


    phi8 = STLFormula.Always(STLFormula.Disjunction(STLFormula.Predicate('x',operatorclass.ge,0.4,INDEX_X),STLFormula.Predicate('y',operatorclass.le,0.2,INDEX_X)),0,20)



    predicate_x_gt3 = STLFormula.Predicate('x',operatorclass.gt,3,INDEX_X)
    predicate_x_le4 = STLFormula.Predicate('x',operatorclass.le,4,INDEX_X)
    predicate_y_gt2 = STLFormula.Predicate('y',operatorclass.gt,2,INDEX_Y)
    predicate_y_le3 = STLFormula.Predicate('y',operatorclass.le,3,INDEX_Y)
    eventually = STLFormula.Eventually(STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt3,predicate_x_le4) , STLFormula.Conjunction(predicate_y_gt2,predicate_y_le3) ),0,30)
    predicate_x_gt1 = STLFormula.Predicate('x',operatorclass.gt,1,INDEX_X)
    predicate_x_le2 = STLFormula.Predicate('x',operatorclass.le,2,INDEX_X)
    always = STLFormula.Always( STLFormula.Negation( STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt1,predicate_x_le2) , STLFormula.Conjunction(predicate_y_gt2,predicate_y_le3) ) ),0,30) 
    phi = STLFormula.Conjunction(eventually,always)
    phi2 = STLFormula.Conjunction(phi,STLFormula.Always( STLFormula.Negation( STLFormula.Conjunction( STLFormula.Conjunction(STLFormula.Predicate('x',operatorclass.gt,1,INDEX_X),STLFormula.Predicate('x',operatorclass.lt,2,INDEX_X)) , STLFormula.Conjunction(STLFormula.Predicate('y',operatorclass.gt,0.5,INDEX_Y),STLFormula.Predicate('y',operatorclass.lt,1.5,INDEX_Y)) ) ),0,30) )
    phi_nnf = STLFormula.toNegationNormalForm(phi2,False)



    phi_test = STLFormula.Always(STLFormula.Conjunction(STLFormula.Conjunction(STLFormula.Predicate('x',operatorclass.ge,-0.5,INDEX_X),STLFormula.Predicate('x',operatorclass.le,2,INDEX_X)),STLFormula.Conjunction(STLFormula.Predicate('y',operatorclass.ge,-0.5,INDEX_Y),STLFormula.Predicate('y',operatorclass.le,0.8,INDEX_Y))),0,20)



    start=[0, 0]
    rand_area=[-2, 4]


    # trajectory = generate_trajectory_milp(phi_nnf,start,rand_area,True)
    trajectory = generate_trajectory_boolean_milp(phi_nnf,start,rand_area)
    print(trajectory)
    # print(phi_nnf.tex())
    x, y = np.array(trajectory).T
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(x,y)
    ax.plot(x, y, label=r'$\rho='+str(phi_nnf.robustness(trajectory,0))+'$')
    plt.title(r"$"+phi_nnf.tex()+"$")
    plt.ylim(rand_area[0], rand_area[1])
    plt.xlim(rand_area[0], rand_area[1])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),shadow=True, ncol=2)
    plt.show()
    
    
# test()