from STL import STLFormula
import scipy.stats
from psomax import pso_maximize
import operator as operatorclass
import math
from sympy.logic import simplify_logic, bool_map


class DTLearn:
    """
        Class representing the online Decision Tree learning algorithm of an STL Formula.
        Follows G. Bombara and C. Belta, "Online learning of temporal logic formulae for signal classification", in 2018 European Control Conference (ECC). IEEE, 2018, pp. 2057–2062
        Constructor takes as input:
            * rand_area: the domain on which signals are generated. rand_area = [lb,ub] where lb is the lower bound and ub the upper bound of the domain.
            * max_horizon: the maximum horizon of the STL Formula to learn
            * (optional) primitives: either 'MOTION_PLANNING' or 'CLASSICAL' (default set to 'CLASSICAL')
        Attributes:
            * positive_dataset: S_+ the dataset containing signals satisfying the target STL Formula
            * negative_dataset: S_- the dataset containing signals violating the target STL Formula
            * invarnode: in case the target STL Formula is partly given (prior knowledge), define an invariant node which cannot be updated 
            * dictnodes: mapping node ID -> node
            * dictnodestr:  mapping node ID -> str(node)
            * length: size of the tree
    """    


    #Incremental ID for each new node
    ID = 1


    class Node:
        """
            Class representing a non-terminal node of the decision tree.
            Follows G. Bombara and C. Belta, "Online learning of temporal logic formulae for signal classification", in 2018 European Control Conference (ECC). IEEE, 2018, pp. 2057–2062
            Constructor takes as input:
                * stl: the STL Formula used to locally separate the data in this node
                * left: left child node
                * right: right child node
            Attributes:
                * identifier: a unique identifier for the node
        """
        def __init__(self,stl,left,right):
            self.stl = stl
            self.left = left
            self.right = right
            self.identifier = 'PHI'+str(DTLearn.ID)
            DTLearn.ID += 1
        
        def __str__(self):
            return self.identifier
            
            
            
    class Leaf:
        """
            Class representing a terminal node of the decision tree.
            Follows G. Bombara and C. Belta, "Online learning of temporal logic formulae for signal classification", in 2018 European Control Conference (ECC). IEEE, 2018, pp. 2057–2062
            Constructor takes as input:
                * label: whether the leaf is an accepting or rejecting node. Has to be STLFormula.TrueF or STLFormula.FalseF
            Attributes:
                * elements: signals in the dataset being classified in this leaf
        """
        def __init__(self,label):
            self.label = label
            self.elements = []



    def __init__(self,rand_area,max_horizon,primitives='CLASSICAL'):
        self.tree = DTLearn.Leaf(STLFormula.TrueF())
        self.rand_area = rand_area
        self.max_horizon = max_horizon
        self.positive_dataset = []
        self.negative_dataset = []
        self.invarnode = None
        self.dictnodes = {}
        self.dictnodestr = {}
        self.length = 0
        self.primitives = primitives




    def tree2STL(self,node):
        """
            Recursive function for translating a node in the tree into an STL Formula, by parsing its subtree
            Follows Algorithm 2 of Bombara et al., "A decision tree approach to data classification using signal temporal logic," in Proceedings of the 19th International Conference on Hybrid Systems: Computation and Control. ACM, 2016, pp. 1–10
        """
        if isinstance(node, DTLearn.Leaf):
            if isinstance(node.label, STLFormula.TrueF):
                return STLFormula.TrueF()
            return STLFormula.FalseF()
        phi_l = STLFormula.Conjunction(node.stl,self.tree2STL(node.left))
        phi_r = STLFormula.Conjunction(STLFormula.Negation(node.stl),self.tree2STL(node.right))
        return STLFormula.Disjunction(phi_l,phi_r)
    
    
    
    
    def toSTLformula(self):
        """
            Function translating a node in the tree into an STL Formula, by parsing its subtree
        """
        return self.tree2STL(self.tree)
    
    
    
    
    def toSimply(self,node):
        """
            Recursive function for getting a string representation of an STL Formula parsable by the sympy.simplify_logic
        """
        if isinstance(node, DTLearn.Leaf):
            if isinstance(node.label, STLFormula.TrueF):
                return "True"
            return "False"
        phi_l = "And(" + node.identifier + "," + self.toSimply(node.left) + ")"
        phi_r = "And(Not(" + node.identifier + ")," + self.toSimply(node.right) + ")"
        return "Or(" + phi_l + "," + phi_r + ")"
    
    
    
    def simple_boolean(self):
        """
            Function returning the string representation of the Conjunctive Normal Form of an STL Formula
        """
        if isinstance(self.tree, DTLearn.Leaf):
            return "True"
        simplified = str(simplify_logic(expr=self.toSimply(self.tree),form='cnf'))
        simplified = simplified.replace("~","\\neg ")
        simplified = simplified.replace("|","\\vee")
        simplified = simplified.replace("&","\wedge")
        for predicate in reversed(sorted(self.dictnodestr.keys())):
            simplified = simplified.replace(predicate,self.dictnodestr[predicate])
        return simplified
    
    
    
    
    def parseTree(self,node,s,l):
        """
            Recursive function parsing the decision tree given a signal, returning the terminal node classifying the signal
        """
        if isinstance(node, DTLearn.Node):
            if node.stl.robustness(s,0) >= 0:
                L,c,phi_path = self.parseTree(node.left,s,l)
                return L,c,STLFormula.Conjunction(phi_path,node.stl)
            else:
                L,c,phi_path = self.parseTree(node.right,s,l)
                return L,c,STLFormula.Conjunction(STLFormula.Negation(phi_path),node.stl)
        if isinstance(node, DTLearn.Leaf):
            node.elements.append(s)
            return node, node.label, node.label
    
    
    
    
    def replaceNode(self,curr_node,leaf,replacement):
        """
            Recursive function to replace a terminal node by a non-terminal node containing an STL Formula locally separating the data
        """
        if isinstance(curr_node, DTLearn.Node):
            curr_node.left = self.replaceNode(curr_node.left,leaf,replacement)
            curr_node.right = self.replaceNode(curr_node.right,leaf,replacement)
            return curr_node
        if isinstance(curr_node, DTLearn.Leaf):
            if curr_node==leaf:
                return replacement
            return curr_node
    
    
    
    
    def MG(self,S,phi):
        """
            Function computing the Misclassification Gain of a dataset w.r.t. a given STL Formula
        """
        S_T = []
        S_F = []
        for s in S:
            if phi.robustness(s,0)>=0:
                S_T.append(s)
            else:
                S_F.append(s)
   
        p_T = len(S_T)/len(S)
        p_F = len(S_F)/len(S)
        
        def MR(S,phi):
            S_CP = [] 
            S_CN = []
            for s in S:
                if phi.robustness(s,0)>=0:
                    if s in self.positive_dataset:
                        S_CP.append(s)
                else:
                    if s in self.negative_dataset:
                        S_CN.append(s)
            p_CP = len(S_CP)/len(S)
            p_CN = len(S_CN)/len(S)
            return min(p_CP,p_CN)
        
        try:
            return MR(S,phi) - ( (p_T*MR(S_T,phi)) + (p_F*MR(S_F,phi)) )
        except ZeroDivisionError:
            return float("-inf")
        
        
    
    
    def locateLeaf(self,s,l):
        """
            Function returning which leaf classifies a signal.
        """
        return self.parseTree(self.tree,s,l)
    
    
    
    
    def updateLeaf(self,L,phi_path):
        """
            Function updating the decision tree.
            It finds the optimal parameters t1,t2,mu1 for each primitives defined by \mathcal{P} = \{ \diamondsuit_{[\tau_1,\tau_2]}(f(x) \sim \mu)\ \text{or}\ \Box_{[\tau_1,\tau_2]}(f(x) \sim \mu) \}
            Follows Algorithm 2 of G. Bombara and C. Belta, "Online learning of temporal logic formulae for signal classification", in 2018 European Control Conference (ECC). IEEE, 2018, pp. 2057–2062
        """
        
        S = L.elements
        
        #Constants
        INDEX_X = 0
        INDEX_Y = 1
        DELTA = 0.1
        N_MAX = 50
        epsilon = scipy.stats.norm.ppf(1-DELTA) * (1 / math.sqrt(2 * len(S)))
        
        #Define the lower and upper bounds for mu, t1, t2, respectively
        lb = [0.5, 0, self.max_horizon-1]
        ub = [3.5, self.max_horizon-1, self.max_horizon]

        #Parameters to optimize
        mu = abs(self.rand_area[1]-self.rand_area[0])/2
        t1 = 0
        t2 = self.max_horizon
                
        # Define the objective for each primitive (to be maximized)
        # \diamondsuit_{[t1,t2]}(x >= \mu)
        def weight_p1(x):
            mu,t1,t2 = x
            if isinstance(phi_path, STLFormula.TrueF):
                return self.MG(S,STLFormula.Eventually(STLFormula.Predicate('x',operatorclass.gt,mu,INDEX_X),int(round(t1)),int(round(t2))))
            return self.MG(S,STLFormula.Conjunction(phi_path,STLFormula.Eventually(STLFormula.Predicate('x',operatorclass.gt,mu,INDEX_X),int(round(t1)),int(round(t2)))))
            
        # \diamondsuit_{[t1,t2]}(x <= \mu)
        def weight_p2(x):
            mu,t1,t2 = x
            if isinstance(phi_path, STLFormula.TrueF):
                return self.MG(S,STLFormula.Eventually(STLFormula.Predicate('x',operatorclass.le,mu,INDEX_X),int(round(t1)),int(round(t2))))
            return self.MG(S,STLFormula.Conjunction(phi_path,STLFormula.Eventually(STLFormula.Predicate('x',operatorclass.le,mu,INDEX_X),int(round(t1)),int(round(t2)))))
            
        # \diamondsuit_{[t1,t2]}(y >= \mu)
        def weight_p3(x):
            mu,t1,t2 = x
            if isinstance(phi_path, STLFormula.TrueF):
                return self.MG(S,STLFormula.Eventually(STLFormula.Predicate('y',operatorclass.gt,mu,INDEX_Y),int(round(t1)),int(round(t2))))
            return self.MG(S,STLFormula.Conjunction(phi_path,STLFormula.Eventually(STLFormula.Predicate('y',operatorclass.gt,mu,INDEX_Y),int(round(t1)),int(round(t2)))))
            
        # \diamondsuit_{[t1,t2]}(y <= \mu)
        def weight_p4(x):
            mu,t1,t2 = x
            if isinstance(phi_path, STLFormula.TrueF):
                return self.MG(S,STLFormula.Eventually(STLFormula.Predicate('y',operatorclass.le,mu,INDEX_Y),int(round(t1)),int(round(t2))))
            return self.MG(S,STLFormula.Conjunction(phi_path,STLFormula.Eventually(STLFormula.Predicate('y',operatorclass.le,mu,INDEX_Y),int(round(t1)),int(round(t2)))))
            
        # \Box_{[t1,t2]}(x >= \mu)
        def weight_p5(x):
            mu,t1,t2 = x
            if isinstance(phi_path, STLFormula.TrueF):
                return self.MG(S,STLFormula.Always(STLFormula.Predicate('x',operatorclass.gt,mu,INDEX_X),int(round(t1)),int(round(t2))))
            return self.MG(S,STLFormula.Conjunction(phi_path,STLFormula.Always(STLFormula.Predicate('x',operatorclass.gt,mu,INDEX_X),int(round(t1)),int(round(t2)))))
            
        # \Box_{[t1,t2]}(x <= \mu)
        def weight_p6(x):
            mu,t1,t2 = x
            if isinstance(phi_path, STLFormula.TrueF):
                return self.MG(S,STLFormula.Always(STLFormula.Predicate('x',operatorclass.le,mu,INDEX_X),int(round(t1)),int(round(t2))))
            return self.MG(S,STLFormula.Conjunction(phi_path,STLFormula.Always(STLFormula.Predicate('x',operatorclass.le,mu,INDEX_X),int(round(t1)),int(round(t2)))))
            
        # \Box_{[t1,t2]}(y >= \mu)
        def weight_p7(x):
            mu,t1,t2 = x
            if isinstance(phi_path, STLFormula.TrueF):
                return self.MG(S,STLFormula.Always(STLFormula.Predicate('y',operatorclass.gt,mu,INDEX_Y),int(round(t1)),int(round(t2))))
            return self.MG(S,STLFormula.Conjunction(phi_path,STLFormula.Always(STLFormula.Predicate('y',operatorclass.gt,mu,INDEX_Y),int(round(t1)),int(round(t2)))))

        # \Box_{[t1,t2]}(y <= \mu)
        def weight_p8(x):
            mu,t1,t2 = x
            if isinstance(phi_path, STLFormula.TrueF):
                return self.MG(S,STLFormula.Always(STLFormula.Predicate('y',operatorclass.le,mu,INDEX_Y),int(round(t1)),int(round(t2))))
            return self.MG(S,STLFormula.Conjunction(phi_path,STLFormula.Always(STLFormula.Predicate('y',operatorclass.le,mu,INDEX_Y),int(round(t1)),int(round(t2)))))
        
        #Optimize each primitive parameter using particle swarm
        xopt_primitive_1, fopt_primitive_1 = pso_maximize(weight_p1, lb, ub, debug=False,maxiter=50)
        xopt_primitive_2, fopt_primitive_2 = pso_maximize(weight_p2, lb, ub, debug=False,maxiter=50)
        xopt_primitive_3, fopt_primitive_3 = pso_maximize(weight_p3, lb, ub, debug=False,maxiter=50)
        xopt_primitive_4, fopt_primitive_4 = pso_maximize(weight_p4, lb, ub, debug=False,maxiter=50)
        xopt_primitive_5, fopt_primitive_5 = pso_maximize(weight_p5, lb, ub, debug=False,maxiter=50)
        xopt_primitive_6, fopt_primitive_6 = pso_maximize(weight_p6, lb, ub, debug=False,maxiter=50)
        xopt_primitive_7, fopt_primitive_7 = pso_maximize(weight_p7, lb, ub, debug=False,maxiter=50)
        xopt_primitive_8, fopt_primitive_8 = pso_maximize(weight_p8, lb, ub, debug=False,maxiter=50) 
        
        #Instatiate each valued primitive
        p1 = STLFormula.Eventually(STLFormula.Predicate('x',operatorclass.gt,xopt_primitive_1[0],INDEX_X),int(round(xopt_primitive_1[1])),int(round(xopt_primitive_1[2])))
        p2 = STLFormula.Eventually(STLFormula.Predicate('x',operatorclass.le,xopt_primitive_2[0],INDEX_X),int(round(xopt_primitive_2[1])),int(round(xopt_primitive_2[2])))
        p3 = STLFormula.Eventually(STLFormula.Predicate('y',operatorclass.gt,xopt_primitive_3[0],INDEX_Y),int(round(xopt_primitive_3[1])),int(round(xopt_primitive_3[2])))
        p4 = STLFormula.Eventually(STLFormula.Predicate('y',operatorclass.le,xopt_primitive_4[0],INDEX_Y),int(round(xopt_primitive_4[1])),int(round(xopt_primitive_4[2])))
        p5 = STLFormula.Always(STLFormula.Predicate('x',operatorclass.gt,xopt_primitive_5[0],INDEX_X),int(round(xopt_primitive_5[1])),int(round(xopt_primitive_5[2])))
        p6 = STLFormula.Always(STLFormula.Predicate('x',operatorclass.le,xopt_primitive_6[0],INDEX_X),int(round(xopt_primitive_6[1])),int(round(xopt_primitive_6[2])))
        p7 = STLFormula.Always(STLFormula.Predicate('y',operatorclass.gt,xopt_primitive_7[0],INDEX_Y),int(round(xopt_primitive_7[1])),int(round(xopt_primitive_7[2])))
        p8 = STLFormula.Always(STLFormula.Predicate('y',operatorclass.le,xopt_primitive_8[0],INDEX_Y),int(round(xopt_primitive_8[1])),int(round(xopt_primitive_8[2])))
        
        #Choosing best primitive
        dicprimitves = {1:p1, 2:p2, 3:p3, 4:p4, 5:p5, 6:p6, 7:p7, 8:p8}
        sortlist = {(xopt_primitive_1[0],xopt_primitive_1[1],xopt_primitive_1[2],1):fopt_primitive_1, (xopt_primitive_2[0],xopt_primitive_2[1],xopt_primitive_2[2],2):fopt_primitive_2, (xopt_primitive_3[0],xopt_primitive_3[1],xopt_primitive_3[2],3):fopt_primitive_3, (xopt_primitive_4[0],xopt_primitive_4[1],xopt_primitive_4[2],4):fopt_primitive_4, (xopt_primitive_5[0],xopt_primitive_5[1],xopt_primitive_5[2],5):fopt_primitive_5, (xopt_primitive_6[0],xopt_primitive_6[1],xopt_primitive_6[2],6):fopt_primitive_6, (xopt_primitive_7[0],xopt_primitive_7[1],xopt_primitive_7[2],7):fopt_primitive_7, (xopt_primitive_8[0],xopt_primitive_8[1],xopt_primitive_8[2],8):fopt_primitive_8}
        best_args = sorted(sortlist.items(), key=lambda t: t[1])[7][0]
        bestprim = best_args[3]
        remaining_primitives =  list(set([1, 2, 3, 4, 5, 6, 7, 8]) - set([bestprim]))
        
        #Removing primitives if necessary by testing best primitive against the others
        toremove_primitives = []
        for i in remaining_primitives:
            if self.MG(S,dicprimitves[bestprim]) - self.MG(S,dicprimitves[i]) > epsilon:
                toremove_primitives.append(i)
                
        remaining_primitives = list(set(remaining_primitives) - set(toremove_primitives))
        
        #Decide whether to update the decision tree or not
        if len(remaining_primitives)<=1 or len(S)>N_MAX:
            createNode = True
            phi_bst = dicprimitves[bestprim]
        else:
            createNode = False
            
            
        """
            If necessary, update tree
        """
        def partition(S,phi_bst):
            S_T = []
            S_F = []
            for s in S:
                if phi_bst.robustness(s,0) < 0:
                    S_F.append(s)
                else:
                    S_T.append(s)
            return S_T, S_F 

        #Update the tree by replacing terminal node by a non terminal node containing the chose valued primitive
        if createNode:
            N = DTLearn.Node(phi_bst,DTLearn.Leaf(STLFormula.TrueF()),DTLearn.Leaf(STLFormula.FalseF()))
            self.dictnodes[N.identifier] = N
            self.dictnodestr[N.identifier] = str(N.stl)
            S_T, S_F = partition(S,phi_bst)
            N.left.elements = S_T
            N.right.elements = S_F
            self.tree = self.replaceNode(self.tree,L,N)
            self.length += 1
            




    
    def updateLeafMissionPrimitives(self,L,phi_path):
        """
            Function updating the decision tree.
            It finds the optimal parameters t1,t2,mu1,mu2,mu3,mu4 for each mission primitives defined by:
                * \mathcal{P}_{spatial} = \{ \Box_{[\tau_1,\tau_2]} \neg (x \geq \mu_1 \wedge x \leq \mu_2 \wedge y \geq \mu_3 \wedge y \leq \mu_4) \}
                * \mathcal{P}_{mission1} = \{ \diamondsuit_{[\tau_1,\tau_2]}(x \geq \mu_1 \wedge x \leq \mu_2 \wedge y \geq \mu_3 \wedge y \leq \mu_4) \}
                * \mathcal{P}_{mission2} = \{ \Box_{[\tau_1,\tau_2]}(x \geq \mu_1 \wedge x \leq \mu_2 \wedge y \geq \mu_3 \wedge y \leq \mu_4) \}
            Follows Algorithm 2 of G. Bombara and C. Belta, "Online learning of temporal logic formulae for signal classification", in 2018 European Control Conference (ECC). IEEE, 2018, pp. 2057–2062
        """
        
        S = L.elements
        
        #Constants
        INDEX_X = 0
        INDEX_Y = 1
        DELTA = 0.1
        N_MAX = 100
        epsilon = scipy.stats.norm.ppf(1-DELTA) * (1 / math.sqrt(2 * len(S)))
        
        #Define the lower and upper bounds for mu, t1, t2, respectively
        lb = [self.rand_area[0],self.rand_area[0],self.rand_area[0],self.rand_area[0], 0, self.max_horizon-1]
        ub = [self.rand_area[1],self.rand_area[1],self.rand_area[1],self.rand_area[1], self.max_horizon-1, self.max_horizon]

        #Parameters to optimize
        mu1 = abs(self.rand_area[1]-self.rand_area[0])/2
        mu2 = abs(self.rand_area[1]-self.rand_area[0])/2
        mu3 = abs(self.rand_area[1]-self.rand_area[0])/2
        mu4 = abs(self.rand_area[1]-self.rand_area[0])/2
        t1 = 0
        t2 = self.max_horizon
                
        # Define the objective for each primitive (to be maximized)
        # \Box_{[t1,t2]} \neg (x \geq mu1 \wedge x \leq mu2 \wedge y \geq mu3 \wedge y \leq mu4)
        def weight_p1(x):
            mu1,mu2,mu3,mu4,t1,t2 = x
            if isinstance(phi_path, STLFormula.TrueF):
                return self.MG(S,STLFormula.Always( STLFormula.Negation( STLFormula.Conjunction( STLFormula.Conjunction(STLFormula.Predicate('x',operatorclass.gt,mu1,INDEX_X),STLFormula.Predicate('x',operatorclass.le,mu2,INDEX_X)) , STLFormula.Conjunction(STLFormula.Predicate('y',operatorclass.gt,mu3,INDEX_Y),STLFormula.Predicate('y',operatorclass.le,mu4,INDEX_Y)) ) ),int(round(t1)),int(round(t2))))
            return self.MG(S,STLFormula.Conjunction(phi_path,STLFormula.Always( STLFormula.Negation( STLFormula.Conjunction( STLFormula.Conjunction(STLFormula.Predicate('x',operatorclass.gt,mu1,INDEX_X),STLFormula.Predicate('x',operatorclass.le,mu2,INDEX_X)) , STLFormula.Conjunction(STLFormula.Predicate('y',operatorclass.gt,mu3,INDEX_Y),STLFormula.Predicate('y',operatorclass.le,mu4,INDEX_Y)) ) ),int(round(t1)),int(round(t2)))))
        
        # \diamondsuit_{[t1,t2]} (x \geq mu1 \wedge x \leq mu2 \wedge y \geq mu3 \wedge y \leq mu4)
        def weight_p2(x):
            mu1,mu2,mu3,mu4,t1,t2 = x
            if isinstance(phi_path, STLFormula.TrueF):
                return self.MG(S,STLFormula.Eventually(STLFormula.Conjunction( STLFormula.Conjunction(STLFormula.Predicate('x',operatorclass.gt,mu1,INDEX_X),STLFormula.Predicate('x',operatorclass.le,mu2,INDEX_X)) , STLFormula.Conjunction(STLFormula.Predicate('y',operatorclass.gt,mu3,INDEX_Y),STLFormula.Predicate('y',operatorclass.le,mu4,INDEX_Y)) ),int(round(t1)),int(round(t2))))
            return self.MG(S,STLFormula.Conjunction(phi_path,STLFormula.Eventually(STLFormula.Conjunction( STLFormula.Conjunction(STLFormula.Predicate('x',operatorclass.gt,mu1,INDEX_X),STLFormula.Predicate('x',operatorclass.le,mu2,INDEX_X)) , STLFormula.Conjunction(STLFormula.Predicate('y',operatorclass.gt,mu3,INDEX_Y),STLFormula.Predicate('y',operatorclass.le,mu4,INDEX_Y)) ),int(round(t1)),int(round(t2)))))
            
        # \Box_{[t1,t2]} (x \geq mu1 \wedge x \leq mu2 \wedge y \geq mu3 \wedge y \leq mu4)
        def weight_p3(x):
            mu1,mu2,mu3,mu4,t1,t2 = x
            if isinstance(phi_path, STLFormula.TrueF):
                return self.MG(S,STLFormula.Always(STLFormula.Conjunction( STLFormula.Conjunction(STLFormula.Predicate('x',operatorclass.gt,mu1,INDEX_X),STLFormula.Predicate('x',operatorclass.le,mu2,INDEX_X)) , STLFormula.Conjunction(STLFormula.Predicate('y',operatorclass.gt,mu3,INDEX_Y),STLFormula.Predicate('y',operatorclass.le,mu4,INDEX_Y)) ),int(round(t1)),int(round(t2))))
            return self.MG(S,STLFormula.Conjunction(phi_path,STLFormula.Always(STLFormula.Conjunction( STLFormula.Conjunction(STLFormula.Predicate('x',operatorclass.gt,mu1,INDEX_X),STLFormula.Predicate('x',operatorclass.le,mu2,INDEX_X)) , STLFormula.Conjunction(STLFormula.Predicate('y',operatorclass.gt,mu3,INDEX_Y),STLFormula.Predicate('y',operatorclass.le,mu4,INDEX_Y)) ),int(round(t1)),int(round(t2)))))
            
        #Optimize each primitive parameter using particle swarm
        xopt_primitive_1, fopt_primitive_1 = pso_maximize(weight_p1, lb, ub, debug=False,maxiter=100)
        xopt_primitive_2, fopt_primitive_2 = pso_maximize(weight_p2, lb, ub, debug=False,maxiter=100)
        xopt_primitive_3, fopt_primitive_3 = pso_maximize(weight_p3, lb, ub, debug=False,maxiter=100)
        
        #Instatiate each valued primitive
        p1 = STLFormula.Always( STLFormula.Negation( STLFormula.Conjunction( STLFormula.Conjunction(STLFormula.Predicate('x',operatorclass.gt,xopt_primitive_1[0],INDEX_X),STLFormula.Predicate('x',operatorclass.le,xopt_primitive_1[1],INDEX_X)) , STLFormula.Conjunction(STLFormula.Predicate('y',operatorclass.gt,xopt_primitive_1[2],INDEX_Y),STLFormula.Predicate('y',operatorclass.le,xopt_primitive_1[3],INDEX_Y)) ) ),int(round(xopt_primitive_1[4])),int(round(xopt_primitive_1[5])))
        p2 = STLFormula.Eventually( STLFormula.Conjunction( STLFormula.Conjunction(STLFormula.Predicate('x',operatorclass.gt,xopt_primitive_2[0],INDEX_X),STLFormula.Predicate('x',operatorclass.le,xopt_primitive_2[1],INDEX_X)) , STLFormula.Conjunction(STLFormula.Predicate('y',operatorclass.gt,xopt_primitive_2[2],INDEX_Y),STLFormula.Predicate('y',operatorclass.le,xopt_primitive_2[3],INDEX_Y)) ),int(round(xopt_primitive_2[4])),int(round(xopt_primitive_2[5])))
        p3 = STLFormula.Always( STLFormula.Conjunction( STLFormula.Conjunction(STLFormula.Predicate('x',operatorclass.gt,xopt_primitive_3[0],INDEX_X),STLFormula.Predicate('x',operatorclass.le,xopt_primitive_3[1],INDEX_X)) , STLFormula.Conjunction(STLFormula.Predicate('y',operatorclass.gt,xopt_primitive_3[2],INDEX_Y),STLFormula.Predicate('y',operatorclass.le,xopt_primitive_3[3],INDEX_Y)) ),int(round(xopt_primitive_3[4])),int(round(xopt_primitive_3[5])))

        #Choosing best primitive
        dicprimitves = {1:p1, 2:p2, 3:p3}
        sortlist = {(xopt_primitive_1[0],xopt_primitive_1[1],xopt_primitive_1[2],1):fopt_primitive_1, (xopt_primitive_2[0],xopt_primitive_2[1],xopt_primitive_2[2],2):fopt_primitive_2, (xopt_primitive_3[0],xopt_primitive_3[1],xopt_primitive_3[2],3):fopt_primitive_3}
        best_args = sorted(sortlist.items(), key=lambda t: t[1])[2][0]
        bestprim = best_args[3]
        remaining_primitives =  list(set([1,2,3]) - set([bestprim]))
        
        #Removing primitives if necessary by testing best primitive against the others
        toremove_primitives = []
        for i in remaining_primitives:
            if self.MG(S,dicprimitves[bestprim]) - self.MG(S,dicprimitves[i]) > epsilon:
                toremove_primitives.append(i)
        remaining_primitives = list(set(remaining_primitives) - set(toremove_primitives))
        
        #Decide whether to update the decision tree or not
        if len(remaining_primitives)<=1 or len(S)>N_MAX:
            createNode = True
            phi_bst = dicprimitves[bestprim]
        else:
            createNode = False
            
        """
            If necessary, update tree
        """
        def partition(S,phi_bst):
            S_T = []
            S_F = []
            for s in S:
                if phi_bst.robustness(s,0) < 0:
                    S_F.append(s)
                else:
                    S_T.append(s)
            return S_T, S_F 

        #Update the tree by replacing terminal node by a non terminal node containing the chose valued primitive
        if createNode:
            N = DTLearn.Node(phi_bst,DTLearn.Leaf(STLFormula.TrueF()),DTLearn.Leaf(STLFormula.FalseF()))
            self.dictnodes[N.identifier] = N
            self.dictnodestr[N.identifier] = str(N.stl)
            S_T, S_F = partition(S,phi_bst)
            N.left.elements = S_T
            N.right.elements = S_F
            self.tree = self.replaceNode(self.tree,L,N)
            self.length += 1
               
               
    
    
    def update(self,s,l):
        """
            Function updating the decision tree when a new signal is available. 
            Takes as input:
                * s: signal
                * l: label (either STL.TrueF or STL.FalseF)
            Follows Algorithm 1 of Bombara et al., "A decision tree approach to data classification using signal temporal logic," in Proceedings of the 19th International Conference on Hybrid Systems: Computation and Control. ACM, 2016, pp. 1–10
        """ 
        
        #Add the signal to the corresponding dataset
        if isinstance(l,STLFormula.TrueF):
            self.positive_dataset.append(s)
        else:
            self.negative_dataset.append(s)
            
        #Gets leaf, class label, and path leading to the terminal node of the decision tree classifying the signal
        L,c,phi_path = self.locateLeaf(s,l)
        
        #If the leaf is not an invariant node and if the classified label mismatches the true label, update the leaf
        if not L==self.invarnode:
            if not type(l) is type(c):
                #Go for mission primitives, or classical primitives
                if self.primitives=='MOTION_PLANNING':
                    self.updateLeafMissionPrimitives(L,phi_path)
                elif self.primitives=='CLASSICAL':
                    self.updateLeaf(L,phi_path)
                else:
                    self.updateLeaf(L,phi_path)
          




if __name__ == '__main__':
    rand_area = [0,4]
    max_horizon = 19

    dtlearn = DTLearn(rand_area,max_horizon,primitives='MOTION_PLANNING')

    print("adding positive string 1")
    dtlearn.update([[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [1, 1], [1.25, 1], [1.5, 1], [1.75, 1], [2, 1], [2, 0.75], [2, 0.5], [2, 0.25], [2, 0], [2.25, 0], [2.5, 0], [2.75, 0], [3, 0], [3.25, 0.25], [3.5, 0.5], [3.75, 0.75]],STLFormula.TrueF())
    print("\nadding positive string 2")
    dtlearn.update([[0.0, 2.0], [0.25, 1.75], [0.5, 1.5], [0.75, 1.25], [1, 1], [1.25, 1], [1.5, 1], [1.75, 1], [2, 1], [2, 0.75], [2, 0.5], [2, 0.25], [2, 0], [2.25, 0], [2.5, 0], [2.75, 0], [3, 0], [3.25, 0.25], [3.1, 0.5], [2.9, 0.75]],STLFormula.TrueF())
    print("\nadding negative string 3")
    dtlearn.update([[0.0, 2.0], [0.25, 1.75], [0.5, 1.5], [0.75, 1.25], [1, 1.25], [1.25, 1.5], [1.5, 1.5], [1.75, 1.5], [2, 1.5], [2, 1.75], [2, 2], [2, 1.75], [2, 1.5], [2.25, 1.25], [2.5, 1.25], [2.75, 1.25], [3, 1.25], [3.25, 1], [3.1, 0.75], [2.9, 0.75]],STLFormula.FalseF())
    print("\nadding positive string 4")
    dtlearn.update([[2.0, 2.0], [1.75, 1.75], [1.5, 1.5], [1.25, 1.25], [1, 1], [1.25, 1], [1.5, 1], [1.75, 1], [2, 1], [2, 0.75], [2, 0.5], [2, 0.25], [2, 0], [2.25, 0], [2.5, 0], [2.75, 0], [3, 0], [3.10, 0.25], [3.0, 0.25], [2.8, 0.20]],STLFormula.TrueF())
    print("\nadding negative string 5")
    dtlearn.update([[0.0, 2.0], [0.25, 1.75], [0.5, 1.5], [0.75, 1.5], [1, 1.75], [1.25, 2], [1.5, 2.2], [1.75, 2.3], [2, 2.3], [2.2, 2.5], [2.2, 2.6], [2.2, 2.4], [2.2, 2.2], [2.25, 2.1], [2.5, 2], [2.75, 2], [3, 1.8], [3.25, 1.6], [3.5, 1.75], [3.55, 1.75]],STLFormula.FalseF())
    print("\nadding negative string 6")
    dtlearn.update([[2.0, 2.0], [1.75, 1.75], [1.5, 1.5], [1.25, 1.25], [1, 1.5], [1.25, 1.75], [1.5, 2], [1.75, 2.2], [2, 2.35], [2.2, 2.5], [2.2, 2.6], [2.2, 2.4], [2.2, 2.2], [2.25, 2.0], [2.5, 1.8], [2.75, 1.5], [3, 1.25], [3.25, 1], [3.5, 0.8], [3.55, 0.7]],STLFormula.FalseF())

    print("\nadding negative string 7")
    dtlearn.update([[0.0, 2.0], [0.25, 1.75], [0.5, 1.5], [0.75, 1.5], [1, 1.75], [1.25, 2], [1.5, 2.2], [1.75, 2.3], [2, 2.3], [2.2, 2.5], [2.2, 2.6], [2.2, 2.4], [2.2, 2.2], [2.25, 2.1], [2.5, 2], [2.75, 2], [3, 1.8], [3.25, 1.6], [3.5, 1.75], [3.55, 1.75]],STLFormula.FalseF())
    print("\nadding negative string 8")
    dtlearn.update([[2.0, 2.0], [1.75, 1.75], [1.5, 1.5], [1.25, 1.25], [1, 1.5], [1.25, 1.75], [1.5, 2], [1.75, 2.2], [2, 2.35], [2.2, 2.5], [2.2, 2.6], [2.2, 2.4], [2.2, 2.2], [2.25, 2.0], [2.5, 1.8], [2.75, 1.5], [3, 1.25], [3.25, 1], [3.5, 0.8], [3.55, 0.7]],STLFormula.FalseF())

    print("\nadding negative string 9")
    dtlearn.update([[0.0, 2.0], [0.25, 1.75], [0.5, 1.5], [0.75, 1.5], [1, 1.75], [1.25, 2], [1.5, 2.2], [1.75, 2.3], [2, 2.3], [2.2, 2.5], [2.2, 2.6], [2.2, 2.4], [2.2, 2.2], [2.25, 2.1], [2.5, 2], [2.75, 2], [3, 1.8], [3.25, 1.6], [3.5, 1.75], [3.55, 1.75]],STLFormula.FalseF())
    print("\nadding negative string 10")
    dtlearn.update([[2.0, 2.0], [1.75, 1.75], [1.5, 1.5], [1.25, 1.25], [1, 1.5], [1.25, 1.75], [1.5, 2], [1.75, 2.2], [2, 2.35], [2.2, 2.5], [2.2, 2.6], [2.2, 2.4], [2.2, 2.2], [2.25, 2.0], [2.5, 1.8], [2.75, 1.5], [3, 1.25], [3.25, 1], [3.5, 0.8], [3.55, 0.7]],STLFormula.FalseF())

    print(dtlearn.simple_boolean())

# test()