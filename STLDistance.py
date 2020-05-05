from STL import STLFormula
import operator as operatorclass
import pulp as plp
from itertools import product, tee, accumulate, combinations



#CONSTANTS
M_up = 100000
M_low = 0.000001

#HARDCODED
#TODO: manage more dimensions
NB_DIMENSIONS = 2


def directed_pompeiu_hausdorff_distance(phi1,phi2,rand_area):
    """
        Function computing the directed pompeiu-hausdorff distance between 2 STL Formulae.
        Takes as input:
            * phi1: an STL Formula
            * phi2: an STL Formula
            * rand_area: the domain on which signals are generated. rand_area = [lb,ub] where lb is the lower bound and ub the upper bound of the domain.
        Follows the definition of Madsen et al., "Metrics  for  signal temporal logic formulae," in 2018 IEEE Conference on Decision andControl (CDC). pp. 1542–1547
        The encoding details of the MILP optimization problem follow Raman et al., "Model  predictive  control  with  signaltemporal logic specifications,” in 53rd IEEE Conference on Decision and Control. IEEE, 2014, pp. 81–87.
    """

    dict_vars = {}
  
    #objective, maximize epsilon
    epsilon = plp.LpVariable(name='epsilon',cat='Continuous',lowBound=0,upBound=rand_area[1]-rand_area[0])
    
    #satisfaction of phi1 and phi2
    zvar1 = plp.LpVariable('z1_'+str(id(phi1))+'_t_'+str(phi1.horizon),cat='Binary')
    zvar2 = plp.LpVariable('z2_'+str(id(phi2))+'_t_'+str(phi2.horizon),cat='Binary')
    dict_vars['z1_'+str(id(phi1))+'_t_'+str(phi1.horizon)] = zvar1
    dict_vars['z2_'+str(id(phi2))+'_t_'+str(phi2.horizon)] = zvar2

    opt_model = plp.LpProblem("MIP Model", plp.LpMaximize)
    opt_model += epsilon
  
    dict_vars['epsilon'] = epsilon

            
    #We want to optimize a signal. The lower and upperbounds are specified by the random area.
    s = plp.LpVariable.dicts("s",(range(max(phi1.horizon,phi2.horizon)+1),range(NB_DIMENSIONS)),rand_area[0],rand_area[1],plp.LpContinuous)    
       
       
    #recursive function
    def model_phi1(phi,t,opt_model):
        if isinstance(phi, STLFormula.TrueF):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 1
        elif isinstance(phi, STLFormula.FalseF):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 0
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
        elif isinstance(phi, STLFormula.FalseF):
            try:
                zvar = dict_vars['z2_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z2_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 0
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
                opt_model += s[t][phi.pi_index_signal] - phi.mu + epsilon <= M_up*zvar-M_low
                opt_model += -(s[t][phi.pi_index_signal] - phi.mu) - epsilon <= M_up*(1-zvar)-M_low
            else:
                opt_model += -s[t][phi.pi_index_signal] + phi.mu + epsilon <= M_up*zvar-M_low
                opt_model += -(-s[t][phi.pi_index_signal] + phi.mu) - epsilon <= M_up*(1-zvar)-M_low
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
    
    if epsilon.varValue == None:
        return 0.0
    return epsilon.varValue





def pompeiu_hausdorff_distance(phi1,phi2,rand_area):
    """
        Function computing the pompeiu-hausdorff distance between 2 STL Formulae.
        Takes as input:
            * phi1: an STL Formula
            * phi2: an STL Formula
            * rand_area: the domain on which signals are generated. rand_area = [lb,ub] where lb is the lower bound and ub the upper bound of the domain.
        Follows the definition of Madsen et al., "Metrics  for  signal temporal logic formulae," in 2018 IEEE Conference on Decision andControl (CDC). pp. 1542–1547
    """
    return round(max(directed_pompeiu_hausdorff_distance(phi1,phi2,rand_area),directed_pompeiu_hausdorff_distance(phi2,phi1,rand_area)),5)







#Definition of an iterator
def pairwise(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)



class Rectangle:
    """
    Class representing a Satisfaction box of an STL Formula
    """
    __slots__ = '__x1', '__y1', '__x2', '__y2', '__dimension'
    
    def __init__(self, x1, y1, x2, y2, dimension):
        self.__setstate__((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), dimension))

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, ', '.join(map(repr, self)))

    def __eq__(self, other):
        return self.data == other.data

    def __ne__(self, other):
        return self.data != other.data

    def __hash__(self):
        return hash(self.data)

    def __len__(self):
        return 4

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __and__(self, other):
        x1, y1, x2, y2 = max(self.x1, other.x1), max(self.y1, other.y1), \
                         min(self.x2, other.x2), min(self.y2, other.y2)
                         
        if self.dimension==other.dimension:
            samedimension = True
            dimension = self.dimension
        elif self.dimension=='ALL' or other.dimension=='ALL':
            samedimension = True
            dimension = 'ALL'
        else:
            samedimension = False
        
        if x1 <= x2 and y1 < y2 and samedimension:
            return type(self)(x1, y1, x2, y2, dimension)

    def __sub__(self, other):
        if self.dimension==other.dimension:
            samedimension = True
            dimension = self.dimension
        elif self.dimension=='ALL' or other.dimension=='ALL':
            samedimension = True
            dimension = 'ALL'
        else:
            samedimension = False
            
        intersection = self & other
        
        if intersection is None:
            yield self
        elif not samedimension:
            yield self
        else:
            x, y = {self.x1, self.x2}, {self.y1, self.y2}
            if self.x1 < other.x1 < self.x2:
                x.add(other.x1)
            if self.y1 < other.y1 < self.y2:
                y.add(other.y1)
            if self.x1 < other.x2 < self.x2:
                x.add(other.x2)
            if self.y1 < other.y2 < self.y2:
                y.add(other.y2)
            for (x1, x2), (y1, y2) in product(pairwise(sorted(x)),
                                              pairwise(sorted(y))):
                instance = type(self)(x1, y1, x2, y2, dimension)
                if instance != intersection:
                    yield instance
                    
    def __or__(self, other):
        return list(set( list(self - other) + list(other - self) ))
        
    def __add__(self,other):
        intersection = self & other
        if intersection is None:
            return [self,other]
        elif self.x1==0 and other.x1==0 and self.x2==0 and other.x2==0:
            return [self & other]
        else:
            return list(set( list(self - other) + [other] ))

    def __getstate__(self):
        return self.x1, self.y1, self.x2, self.y2, self.dimension

    def __setstate__(self, state):
        self.__x1, self.__y1, self.__x2, self.__y2, self.__dimension = state

    @property
    def x1(self):
        return self.__x1

    @property
    def y1(self):
        return self.__y1

    @property
    def x2(self):
        return self.__x2

    @property
    def y2(self):
        return self.__y2

    @property
    def dimension(self):
        return self.__dimension

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return (self.x2-self.x1) * (self.y2-self.y1)

    intersection = __and__

    difference = __sub__
    
    symmetric_difference = __or__
    
    union = __add__

    data = property(__getstate__)



class ChoiceSet:
    """
    Class representing a choice set
    """
    def __init__(self,lst):
        self.lst = lst
    
    def __str__(self):
        return str(self.lst)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self,other):
        return self.lst == other.lst
        
    def __add__(self,other):
        union = {} 
        if not other.lst:
            return other,self
        if isinstance(other,ChoiceSet):
            if not self.lst:
                return self,other
            for a,b in product(self.lst,other.lst):
                for rect in a+b:
                    union[rect] = None
            return BoxSet([]),ChoiceSet(list(union))
        else:
            if not self.lst:
                return other,self
            for a,b in product(self.lst,other.lst):
                for rect in a-b:
                    union[rect] = None
            return BoxSet([]), ChoiceSet(list(union))
            
    def __or__(self,other):
        sd = []
        if not self.lst:
            return other
        if not other.lst:
            return self
        if self.lst == other.lst:
            return ChoiceSet([])
                
        rest_of_a = []
        for a in self.lst:
            rest_of_a.append(a)
            for b in other.lst:
                toadd = []
                todel = []
                for a_prime in rest_of_a:
                    todel.append(a_prime)
                    toadd.extend(a_prime - b)
                for d in todel:
                    rest_of_a.remove(d)
                rest_of_a.extend(toadd)
        
        sd.extend(rest_of_a)   
        
        
        rest_of_b = []
        for b in other.lst:
            rest_of_b.append(b)
            for a in self.lst:
                toadd = []
                todel = []
                for b_prime in rest_of_b:
                    todel.append(b_prime)
                    toadd.extend(b_prime - a)
                for d in todel:
                    rest_of_b.remove(d)
                rest_of_b.extend(toadd)
        
        sd.extend(rest_of_b)    
        
        return ChoiceSet(list(set(sd)))



class BoxSet:
    """
    Class representing a box set
    """
    def __init__(self,lst):
        self.lst = lst
    
    def __str__(self):
        return str(self.lst)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self,other):
        return self.lst == other.lst
        
    def __add__(self,other):
        union = {}            
        if not other.lst:
            return self,other
        if isinstance(other,BoxSet):
            if not self.lst:
                return other, ChoiceSet([])
            for a,b in product(self.lst,other.lst):
                for rect in a+b:
                    union[rect] = None
            return BoxSet(list(union)),ChoiceSet([])
        else:
            if not self.lst:
                return BoxSet([]),other
            for a,b in product(self.lst,other.lst):
                for rect in b-a:
                    union[rect] = None
            return self, ChoiceSet(list(union))
            
    def __or__(self,other):
        sd = []
        if not self.lst:
            return other
        if not other.lst:
            return self
        if self.lst == other.lst:
            return BoxSet([])
                
        
        rest_of_a = []
        for a in self.lst:
            rest_of_a.append(a)
            for b in other.lst:
                toadd = []
                todel = []
                for a_prime in rest_of_a:
                    todel.append(a_prime)
                    toadd.extend(a_prime - b)
                for d in todel:
                    rest_of_a.remove(d)
                rest_of_a.extend(toadd)
        
        sd.extend(rest_of_a)   
        
        
        rest_of_b = []
        for b in other.lst:
            rest_of_b.append(b)
            for a in self.lst:
                toadd = []
                todel = []
                for b_prime in rest_of_b:
                    todel.append(b_prime)
                    toadd.extend(b_prime - a)
                for d in todel:
                    rest_of_b.remove(d)
                rest_of_b.extend(toadd)
        
        sd.extend(rest_of_b)        
        
        return BoxSet(list(set(sd)))



def symmetric_difference_distance(phi1,phi2,rand_area):
    """
        Function computing the symmetric difference distance between 2 STL Formulae.
        Takes as input:
            * phi1: an STL Formula
            * phi2: an STL Formula
            * rand_area: the domain on which signals are generated. rand_area = [lb,ub] where lb is the lower bound and ub the upper bound of the domain.
        Follows the definition of Madsen et al., "Metrics  for  signal temporal logic formulae," in 2018 IEEE Conference on Decision andControl (CDC). pp. 1542–1547
    """
    
    max_horizon = max(phi1.horizon,phi2.horizon)
      
    
    def area(boxelement):
        if isinstance(boxelement,Rectangle):
            return boxelement.area
        if isinstance(boxelement,ChoiceSet):
            if boxelement.lst:
                tmp = 0.0
                for choicebox in boxelement.lst:
                    tmp += area(choicebox)
                return tmp/len(boxelement.lst)
        if isinstance(boxelement,BoxSet):
            tmp = 0.0
            for box in boxelement.lst:
                tmp += area(box)
            return tmp
        return 0.0

    
    def rec_aos(phi,rand_area,max_horizon):
        if isinstance(phi, STLFormula.TrueF):
            return BoxSet([Rectangle(0,0,max_horizon,1,'ALL')]), ChoiceSet([])
        elif isinstance(phi, STLFormula.FalseF):
            return BoxSet([Rectangle(0,0,0,0,'ALL')]), ChoiceSet([])
        elif isinstance(phi, STLFormula.Predicate):
            if phi.operator == operatorclass.gt or phi.operator == operatorclass.ge:
                return BoxSet([Rectangle(0,phi.mu/(rand_area[1]-rand_area[0]),0,1,phi.dimension)]), ChoiceSet([])
            else:
                return BoxSet([Rectangle(0,0,0,phi.mu/(rand_area[1]-rand_area[0]),phi.dimension)]), ChoiceSet([])
        elif isinstance(phi, STLFormula.Conjunction):
            p1_box, p1_choice = rec_aos(phi.first_formula,rand_area,max_horizon)
            p2_box, p2_choice = rec_aos(phi.second_formula,rand_area,max_horizon)
            
            p_box,_ = p1_box+p2_box
            _,p_choice = p1_choice+p2_choice
            
            union_box, union_choice = p_box + p_choice

            return union_box, union_choice
            
        elif isinstance(phi, STLFormula.Disjunction):
            p1_box, p1_choice = rec_aos(phi.first_formula,rand_area,max_horizon)
            p2_box, p2_choice = rec_aos(phi.second_formula,rand_area,max_horizon)
            
            union_choice = {}
            for a,b in product(p1_box.lst,p2_box.lst):
                for rect in a+b:
                    union_choice[rect] = None
            for a,b in product(list(union_choice),p1_choice.lst):
                for rect in a+b:
                    union_choice[rect] = None
            for a,b in product(list(union_choice),p2_choice.lst):
                for rect in a+b:
                    union_choice[rect] = None
            
            return BoxSet([]), ChoiceSet(list(union_choice))
        elif isinstance(phi, STLFormula.Always):
            d_box = []
            d_choice = []
            p1_box, p1_choice = rec_aos(phi.formula,rand_area,max_horizon)
            for b in p1_box.lst:
                d_box.append(Rectangle(b.x1+phi.t1,b.y1,b.x2+phi.t2,b.y2,b.dimension))
            for b in p1_choice.lst:
                d_choice.append(Rectangle(b.x1+phi.t1,b.y1,b.x2+phi.t2,b.y2,b.dimension))
            return BoxSet(d_box), ChoiceSet(d_choice) 
        elif isinstance(phi, STLFormula.Eventually):
            p1_box, p1_choice = rec_aos(phi.formula,rand_area,max_horizon)

            union_choice = {}
            if not p1_box.lst:
                union_choice = p1_choice.lst
            elif not p1_choice.lst:
                union_choice = p1_box.lst
            else:
                for a,b in product(p1_box.lst,p1_choice.lst):
                    for rect in a+b:
                        union_choice[rect] = None
            
            d = []
            for b in union_choice:
                for i in range(phi.t1,phi.t2):
                    d.append(Rectangle(b.x1+i,b.y1,b.x2+i+1,b.y2,b.dimension))
                    
            return BoxSet([]), ChoiceSet(d)                    

    
    b1_box, b1_choice = rec_aos(phi1,rand_area,max_horizon)
    b2_box, b2_choice = rec_aos(phi2,rand_area,max_horizon)
    

    if b1_box == b2_box and b1_choice == b2_choice:
        return 0.0
    
    sd_box = b1_box | b2_box
    sd_choice = b1_choice | b2_choice
    
    bs = []
    for c1 in b1_choice.lst:
        cs = []
        for c2 in b2_choice.lst:
            cs.append(BoxSet(c1 | c2))
        bs.append(ChoiceSet(cs))
    boxchoice = ChoiceSet(bs)

    
    if not sd_box.lst:
        sd = sd_choice
    elif not sd_choice.lst:
        sd = sd_box
    else:
        boxsets = []
        for choice in sd_choice.lst:
            boxsets.append(BoxSet([choice]) | sd_box)
        sd = ChoiceSet(boxsets)
        
    sd = BoxSet([sd,boxchoice])

    return round(area(sd)/max_horizon,5)








if __name__ == '__main__':
    
    #Constant
    INDEX_X = 0
    INDEX_Y = 1
    
    #Examples of Madsen et al. (2018)
    predicate_x_ge02 = STLFormula.Predicate('x',operatorclass.ge,0.2,INDEX_X)
    predicate_x_lt04 = STLFormula.Predicate('x',operatorclass.le,0.4,INDEX_X)
    predicate_x_lt044 = STLFormula.Predicate('x',operatorclass.le,0.44,INDEX_X)

    phi1 = STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,20)
    phi2 = STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt044),0,20)
    phi3 = STLFormula.Eventually(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,20)
    phi4 = STLFormula.Conjunction(STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,20),STLFormula.Eventually(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,20))
    phi5 = STLFormula.Conjunction(STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,10),STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt044),12,20))
    phi6 = STLFormula.Always(STLFormula.Eventually(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,4),0,16)
    true = STLFormula.TrueF()


    #Reproduces the results of Madsen et al. (2018)
    rand_area = [0,1]
    
    #Symetric Difference    
    print('sd(True,phi1)',symmetric_difference_distance(true,phi1,rand_area),'(0.8)')
    print('sd(phi1,phi1)',symmetric_difference_distance(phi1,phi1,rand_area),'(0.0)')
    print('sd(phi1,phi2)',symmetric_difference_distance(phi1,phi2,rand_area),'(0.04)')
    print('sd(phi1,phi3)',symmetric_difference_distance(phi1,phi3,rand_area),'(0.19)')
    print('sd(phi1,phi4)',symmetric_difference_distance(phi1,phi4,rand_area),'(0.0)')
    print('sd(phi1,phi5)',symmetric_difference_distance(phi1,phi5,rand_area),'(0.036)')
    print('sd(phi1,phi6)',symmetric_difference_distance(phi1,phi6,rand_area),'(0.03)')

    print('')

    print('sd(True,phi2)',symmetric_difference_distance(true,phi2,rand_area),'(0.76)')
    print('sd(phi2,phi2)',symmetric_difference_distance(phi2,phi2,rand_area),'(0.0)')
    print('sd(phi2,phi3)',symmetric_difference_distance(phi2,phi3,rand_area),'(0.23)')
    print('sd(phi2,phi4)',symmetric_difference_distance(phi2,phi4,rand_area),'(0.04)')
    print('sd(phi2,phi5)',symmetric_difference_distance(phi2,phi5,rand_area),'(0.044)')
    print('sd(phi2,phi6)',symmetric_difference_distance(phi2,phi6,rand_area),'(0.07)')

    print('')

    print('sd(True,phi3)',symmetric_difference_distance(true,phi3,rand_area),'(0.99)')
    print('sd(phi3,phi3)',symmetric_difference_distance(phi3,phi3,rand_area),'(0.0)')
    print('sd(phi3,phi4)',symmetric_difference_distance(phi3,phi4,rand_area),'(0.19)')
    print('sd(phi3,phi5)',symmetric_difference_distance(phi3,phi5,rand_area),'(0.188)')
    print('sd(phi3,phi6)',symmetric_difference_distance(phi3,phi6,rand_area),'(0.16)')
    # exit()
    print('')

    print('sd(True,phi4)',symmetric_difference_distance(true,phi4,rand_area),'(0.8)')
    print('sd(phi4,phi4)',symmetric_difference_distance(phi4,phi4,rand_area),'(0.0)')
    print('sd(phi4,phi5)',symmetric_difference_distance(phi4,phi5,rand_area),'(0.036)')
    print('sd(phi4,phi6)',symmetric_difference_distance(phi4,phi6,rand_area),'(0.03)')

    print('')

    print('sd(True,phi5)',symmetric_difference_distance(true,phi5,rand_area),'(0.804)')
    print('sd(phi5,phi5)',symmetric_difference_distance(phi5,phi5,rand_area),'(0.0)')
    print('sd(phi5,phi6)',symmetric_difference_distance(phi5,phi6,rand_area),'(0.066)')

    print('')

    print('sd(True,phi6)',symmetric_difference_distance(true,phi6,rand_area),'(0.83)')
    print('sd(phi6,phi6)',symmetric_difference_distance(phi6,phi6,rand_area),'(0.0)')    
    
    
    print('')
    print('')
    # exit()
    
    #Pompeiu-Hausdorff
    print('ph(True,phi1)',pompeiu_hausdorff_distance(true,phi1,rand_area),'(0.6)')
    print('ph(phi1,phi1)',pompeiu_hausdorff_distance(phi1,phi1,rand_area),'(0.0)')
    print('ph(phi1,phi2)',pompeiu_hausdorff_distance(phi1,phi2,rand_area),'(0.04)')
    print('ph(phi1,phi3)',pompeiu_hausdorff_distance(phi1,phi3,rand_area),'(0.6)')
    print('ph(phi1,phi4)',pompeiu_hausdorff_distance(phi1,phi4,rand_area),'(0.0)')
    print('ph(phi1,phi5)',pompeiu_hausdorff_distance(phi1,phi5,rand_area),'(0.6)')
    print('ph(phi1,phi6)',pompeiu_hausdorff_distance(phi1,phi6,rand_area),'(0.6)')

    print('')

    print('ph(True,phi2)',pompeiu_hausdorff_distance(true,phi2,rand_area),'(0.56)')
    print('ph(phi2,phi2)',pompeiu_hausdorff_distance(phi2,phi2,rand_area),'(0.0)')
    print('ph(phi2,phi3)',pompeiu_hausdorff_distance(phi2,phi3,rand_area),'(0.56)')
    print('ph(phi2,phi4)',pompeiu_hausdorff_distance(phi2,phi4,rand_area),'(0.04)')
    print('ph(phi2,phi5)',pompeiu_hausdorff_distance(phi2,phi5,rand_area),'(0.56)')
    print('ph(phi2,phi6)',pompeiu_hausdorff_distance(phi2,phi6,rand_area),'(0.56)')

    print('')

    print('ph(True,phi3)',pompeiu_hausdorff_distance(true,phi3,rand_area),'(0.6)')
    print('ph(phi3,phi3)',pompeiu_hausdorff_distance(phi3,phi3,rand_area),'(0.0)')
    print('ph(phi3,phi4)',pompeiu_hausdorff_distance(phi3,phi4,rand_area),'(0.6)')
    print('ph(phi3,phi5)',pompeiu_hausdorff_distance(phi3,phi5,rand_area),'(0.6)')
    print('ph(phi3,phi6)',pompeiu_hausdorff_distance(phi3,phi6,rand_area),'(0.6)')

    print('')

    print('ph(True,phi4)',pompeiu_hausdorff_distance(true,phi4,rand_area),'(0.6)')
    print('ph(phi4,phi4)',pompeiu_hausdorff_distance(phi4,phi4,rand_area),'(0.0)')
    print('ph(phi4,phi5)',pompeiu_hausdorff_distance(phi4,phi5,rand_area),'(0.6)')
    print('ph(phi4,phi6)',pompeiu_hausdorff_distance(phi4,phi6,rand_area),'(0.6)')

    print('')

    print('ph(True,phi5)',pompeiu_hausdorff_distance(true,phi5,rand_area),'(0.6)')
    print('ph(phi5,phi5)',pompeiu_hausdorff_distance(phi5,phi5,rand_area),'(0.0)')
    print('ph(phi5,phi6)',pompeiu_hausdorff_distance(phi5,phi6,rand_area),'(0.6)')

    print('')

    print('ph(True,phi6)',pompeiu_hausdorff_distance(true,phi6,rand_area),'(0.6)')
    print('ph(phi6,phi6)',pompeiu_hausdorff_distance(phi6,phi6,rand_area),'(0.0)')
    
    
    print('')
    print('')
    
    #Example motion planning
    
    predicate_x_gt0 = STLFormula.Predicate('x',operatorclass.gt,0,INDEX_X)
    predicate_x_le1 = STLFormula.Predicate('x',operatorclass.le,1,INDEX_X)
    predicate_y_gt3 = STLFormula.Predicate('y',operatorclass.gt,3,INDEX_Y)
    predicate_y_le4 = STLFormula.Predicate('y',operatorclass.le,4,INDEX_Y)
    eventually1 = STLFormula.Eventually(STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt0,predicate_x_le1) , STLFormula.Conjunction(predicate_y_gt3,predicate_y_le4) ),10,20)
    predicate_x_gt3 = STLFormula.Predicate('x',operatorclass.gt,3,INDEX_X)
    predicate_x_le4 = STLFormula.Predicate('x',operatorclass.le,4,INDEX_X)
    predicate_y_gt0 = STLFormula.Predicate('y',operatorclass.gt,0,INDEX_Y)
    predicate_y_le1 = STLFormula.Predicate('y',operatorclass.le,1,INDEX_Y)
    eventually2 = STLFormula.Eventually(STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt3,predicate_x_le4) , STLFormula.Conjunction(predicate_y_gt0,predicate_y_le1) ),30,50)
    predicate_x_gt2 = STLFormula.Predicate('x',operatorclass.gt,2,INDEX_X)
    predicate_y_gt1 = STLFormula.Predicate('y',operatorclass.gt,1,INDEX_Y)
    predicate_x_le3 = STLFormula.Predicate('x',operatorclass.le,3,INDEX_X)
    predicate_y_le2 = STLFormula.Predicate('y',operatorclass.le,2,INDEX_Y)
    always = STLFormula.Always( STLFormula.Negation( STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt2,predicate_x_le3) , STLFormula.Conjunction(predicate_y_gt1,predicate_y_le2) ) ),0,50) 
    phi = STLFormula.Conjunction(STLFormula.Conjunction(eventually1,eventually2),always)
    phi_target = STLFormula.toNegationNormalForm(phi,False)

    phi_hypothesis = STLFormula.Conjunction(eventually1,eventually2)

    rand_area = [0,4]

    print(pompeiu_hausdorff_distance(true,phi_target,rand_area),symmetric_difference_distance(true,phi_target,rand_area),)
    print(pompeiu_hausdorff_distance(phi_hypothesis,phi_target,rand_area),symmetric_difference_distance(phi_hypothesis,phi_target,rand_area))
    print(pompeiu_hausdorff_distance(eventually1,eventually2,rand_area),symmetric_difference_distance(eventually1,eventually2,rand_area))


    
    