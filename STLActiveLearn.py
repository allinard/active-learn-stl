from STL import *
from STLGenerateSignal import *
from STLDistance import *
from STLDTLearn import *
import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import pickle
import time
import copy



class STLActiveLearn:
    """
        Class for Active Learning of an STL specification, given a System Under Test from which we can query information on its target specification phi_target.
        Takes as input:
            * phi_target: the target specification to learn
            * rand_area: the domain on which signals are generated. rand_area = [lb,ub] where lb is the lower bound and ub the upper bound of the domain.
            * dimensions: the dimensions on which the STL formula to learn is defined, e.g. dimensions=['x','y'].
            * start: a vector of the form [x0,y0,...] for the starting point coordinates
            * max_horizon: the maximum horizon of the STL Formula to learn
            * primitives (optional): either 'MOTION_PLANNING' or 'CLASSICAL' (default set to 'CLASSICAL')
            * signal_gen (optional): the signal generation method given an STL Formula. Either 'BOOLEAN' or 'QUANTITATIVE' or 'QUANTITATIVE_OPTIMIZE' (default set to 'QUANTITATIVE_OPTIMIZE')
            * U (optional): a basic control policy standing for how much can move in 1 time stamp, i.e. \forall t \in [0,T], |s[t]-s[t+1]| < U \pm \epsilon (default set to 0.2)
            * epsilon (optional): basic control policy parameter (default set to 0.05)
            * alpha (optional): a convergence factor which we consider the distance between the hypothesis specification and target specification good enough to terminate the algorithm (default set to 0.01)
            * beta (optional): probability of triggering either a membership query or an equivalence query (default set to 0.5)
            * gamma (optional): number of iterations without improvement (decrease of distance between phi_hypothesis and phi_target) after triggering reset of the decision tree (default set to 50)
            * MAX_IT (optional): maximum number of iterations (default set to 100)
            * phi_hypothesis (optional): the hypothesis specification (default set to True)
            * plot_activated (optional): show plot at each iteration (default set to False)
        Attributes:
            * dtlearn: decision tree for STL Formula learning 
    """    
    def __init__(self,phi_target,rand_area,dimensions,start,max_horizon,primitives='CLASSICAL',signal_gen='QUANTITATIVE_OPTIMIZE',U=0.2,epsilon=0.05,alpha=0.01,beta=0.5,gamma=50,MAX_IT=100,phi_hypothesis=STLFormula.TrueF(),plot_activated=False):

        #Instance of the DT learning algorithm
        self.dtlearn = DTLearn(rand_area,max_horizon,primitives=primitives)
        
        #For monitoring of experiments
        self.dic_results = {}

        prior_knowledge = False
        #Detect if prior knowledge on the phi_hypothesis was given
        if not isinstance(phi_hypothesis,STLFormula.TrueF):
            prior_knowledge = True
            self.invariant_phi_hypothesis = phi_hypothesis
            invarnode = DTLearn.Leaf(STLFormula.FalseF())
            truenode = DTLearn.Leaf(STLFormula.TrueF())
            self.dtlearn.tree = DTLearn.Node(self.invariant_phi_hypothesis,truenode,invarnode)
            self.dtlearn.invarnode = invarnode
            
        #Sets the hypothesis horizon to the target one, so we can generate signals properly
        phi_hypothesis.horizon = max_horizon


        NO_IMPROVE = 0
        LAST_DIST = float("inf")
        BEST_DIST = float("inf")
        BEST_HYPOTHESIS = STLFormula.TrueF()
        BEST_TREE = self.dtlearn.tree

        #START ITERATIONS
        for it in range(1,MAX_IT):

            #Decide if go for membership or equivalence query
            if random.random() > beta:
                """
                    EQUIVALENCE QUERY
                """
                print("\niteration",it, "-- Equivalence Query")
                #translate hypothesis into negation normal form
                phi_hypothesis_nnf = STLFormula.toNegationNormalForm(phi_hypothesis,False)
                #compute distance between hypothesis and target
                distance = pompeiu_hausdorff_distance(phi_hypothesis_nnf,phi_target,rand_area,dimensions)
                #generate counterexample
                if distance > alpha:
                    #compute symmetric difference of target and hypothesis
                    xor = STLFormula.Xor(phi_hypothesis_nnf,phi_target)
                    xor_nnf = STLFormula.toNegationNormalForm(xor,False)
                    #compute signal
                    try:
                        if signal_gen=='QUANTITATIVE_OPTIMIZE':
                            signal = generate_signal_milp_quantitative(xor_nnf,start,rand_area,dimensions,U,epsilon,True)
                        elif signal_gen=='QUANTITATIVE':
                            signal = generate_signal_milp_quantitative(xor_nnf,start,rand_area,dimensions,U,epsilon,False)
                        else:
                            signal = generate_signal_milp_boolean(xor_nnf,start,rand_area,dimensions,U,epsilon)
                    except Exception:
                        if signal_gen=='QUANTITATIVE_OPTIMIZE':
                            signal = generate_signal_milp_quantitative(phi_target,start,rand_area,dimensions,U,epsilon,True)
                        elif signal_gen=='QUANTITATIVE':
                            signal = generate_signal_milp_quantitative(phi_target,start,rand_area,dimensions,U,epsilon,False)
                        else:
                            signal = generate_signal_milp_boolean(phi_target,start,rand_area,dimensions,U,epsilon)
                    #let the teacher evaluate    
                    ro = round(phi_target.robustness(signal,0), 5)
                    print("robustness signal: ",ro)
                    #update hypothesis
                    #add to negative dataset
                    if ro < -0.0:
                        print("updating decision tree")
                        self.dtlearn.update(signal,STLFormula.FalseF())
                        print("updating decision tree done")
                        if plot_activated:
                            plt.plot([x for (x, y) in signal], [y for (x, y) in signal], '-r')
                            plt.grid(True)
                            plt.pause(0.01)
                    #add to positive dataset
                    else:
                        print("updating decision tree")
                        self.dtlearn.update(signal,STLFormula.TrueF())
                        print("updating decision tree done")
                        if plot_activated:
                            plt.plot([x for (x, y) in signal], [y for (x, y) in signal], '-g')
                            plt.grid(True)
                            plt.pause(0.01)
                    #retrieve hypothesis
                    phi_hypothesis = self.dtlearn.toSTLformula()
                    print("hypothesis: ",STLFormula.simplify_dtlearn(phi_hypothesis))
                    distance = pompeiu_hausdorff_distance(STLFormula.toNegationNormalForm(phi_hypothesis,False),phi_target,rand_area,dimensions)
                    print("distance: ",distance)
                #if distance < alpha, then the hypothesis and the target are considered equivalent. termination of algorithm.
                else:
                    print("solution found")
                    print(self.dtlearn.simple_boolean())
                    BEST_TREE = copy.deepcopy(self.dtlearn.tree)
                    BEST_DIST = distance
                    break
                    
                    
            else:    
                """
                    MEMBERSHIP QUERY
                """
                print("\niteration",it, "-- Membership Query")
                #translate hypothesis into negation normal form
                phi_hypothesis_nnf = STLFormula.toNegationNormalForm(phi_hypothesis,False)
                #generate signal belonging to hypothesis
                phi_hypothesis_nnf.horizon = max_horizon
                try:
                    #compute signal
                    if signal_gen=='QUANTITATIVE_OPTIMIZE':
                        if isinstance(phi_hypothesis,STLFormula.TrueF):
                            signal = generate_signal_milp_quantitative(phi_hypothesis_nnf,start,rand_area,dimensions,U,epsilon,False)
                        else:
                            signal = generate_signal_milp_quantitative(phi_hypothesis_nnf,start,rand_area,dimensions,U,epsilon,True)
                    elif signal_gen=='QUANTITATIVE':
                        signal = generate_signal_milp_quantitative(phi_hypothesis_nnf,start,rand_area,dimensions,U,epsilon,False)
                    else:
                        signal = generate_signal_milp_boolean(phi_hypothesis_nnf,start,rand_area,dimensions,U,epsilon)    
                #If the signal cannot be generated (for instance, if the constraint solver can not provide any signal satisfying the hypothesis), then reset the hypothesis to True or to prior knowledge
                except Exception:
                    if prior_knowledge:
                        invarnode = DTLearn.Leaf(STLFormula.FalseF())
                        invarnode.elements.extend(self.dtlearn.negative_dataset)
                        truenode = DTLearn.Leaf(STLFormula.TrueF())
                        truenode.elements.extend(self.dtlearn.positive_dataset)
                        self.dtlearn.tree = DTLearn.Node(self.invariant_phi_hypothesis,truenode,invarnode)
                        self.dtlearn.invarnode = invarnode
                        phi_hypothesis = self.invariant_phi_hypothesis
                    else:
                        leaf = DTLearn.Leaf(STLFormula.TrueF())
                        leaf.elements.extend(self.dtlearn.negative_dataset)
                        leaf.elements.extend(self.dtlearn.positive_dataset)
                        self.dtlearn.tree = leaf
                        phi_hypothesis = STLFormula.TrueF()
                    phi_hypothesis.horizon = max_horizon
                    NO_IMPROVE = 0
                    LAST_DIST = float("inf")
                    continue
                #let the teacher evaluate
                ro = round(phi_target.robustness(signal,0), 5)
                print("robustness signal: ",ro)
                           
               #update hypothesis
                #add to negative dataset
                if ro < -0.0:
                    print("updating decision tree")
                    self.dtlearn.update(signal,STLFormula.FalseF())
                    print("updating decision tree done")
                    if plot_activated:
                        plt.plot([x for (x, y) in signal], [y for (x, y) in signal], '-r')
                        plt.grid(True)
                        plt.pause(0.01)
                #add to positive dataset
                else:
                    print("updating decision tree")
                    self.dtlearn.update(signal,STLFormula.TrueF())
                    print("updating decision tree done")
                    if plot_activated:
                        plt.plot([x for (x, y) in signal], [y for (x, y) in signal], '-g')
                        plt.grid(True)
                        plt.pause(0.01)
                #retrieve hypothesis
                phi_hypothesis = self.dtlearn.toSTLformula()
                print("hypothesis: ",STLFormula.simplify_dtlearn(phi_hypothesis))
                distance = pompeiu_hausdorff_distance(STLFormula.toNegationNormalForm(phi_hypothesis,False),phi_target,rand_area,dimensions)
                print("distance: ",distance)


            if distance < LAST_DIST:
                NO_IMPROVE = 0
                LAST_DIST = distance
                if distance < BEST_DIST:
                    BEST_TREE = copy.deepcopy(self.dtlearn.tree)
                    BEST_DIST = distance
            elif NO_IMPROVE > 50:
                if prior_knowledge:
                    invarnode = DTLearn.Leaf(STLFormula.FalseF())
                    invarnode.elements.extend(self.dtlearn.negative_dataset)
                    truenode = DTLearn.Leaf(STLFormula.TrueF())
                    truenode.elements.extend(self.dtlearn.positive_dataset)
                    self.dtlearn.tree = DTLearn.Node(self.invariant_phi_hypothesis,truenode,invarnode)
                    self.dtlearn.invarnode = invarnode
                    phi_hypothesis = self.invariant_phi_hypothesis
                else:
                    leaf = DTLearn.Leaf(STLFormula.TrueF())
                    leaf.elements.extend(self.dtlearn.negative_dataset)
                    leaf.elements.extend(self.dtlearn.positive_dataset)
                    self.dtlearn.tree = leaf
                    phi_hypothesis = STLFormula.TrueF()
                phi_hypothesis.horizon = max_horizon
                NO_IMPROVE = 0
                LAST_DIST = float("inf")
            else:
                NO_IMPROVE += 1   
            self.dic_results[it] = (distance,str(phi_hypothesis),self.dtlearn.toSimply(self.dtlearn.tree))
                
        self.dtlearn.tree = BEST_TREE




if __name__ == '__main__':

    #Constants
    rand_area = [0,4]
    start=[0, 0]
    max_horizon = 30
    INDEX_X = 0
    INDEX_Y = 1
    dimensions = ['x','y']

    #Define STL Formulae
    predicate_x_gt3 = STLFormula.Predicate('x',operatorclass.gt,3,INDEX_X)
    predicate_x_le4 = STLFormula.Predicate('x',operatorclass.le,4,INDEX_X)
    predicate_y_gt2 = STLFormula.Predicate('y',operatorclass.gt,2,INDEX_Y)
    predicate_y_le3 = STLFormula.Predicate('y',operatorclass.le,3,INDEX_Y)
    mission = STLFormula.Always(STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt3,predicate_x_le4) , STLFormula.Conjunction(predicate_y_gt2,predicate_y_le3) ),25,30)
    predicate_x_gt1 = STLFormula.Predicate('x',operatorclass.gt,1,INDEX_X)
    predicate_y_gt1 = STLFormula.Predicate('y',operatorclass.gt,1,INDEX_Y)
    predicate_x_le2 = STLFormula.Predicate('x',operatorclass.le,2,INDEX_X)
    predicate_y_le3 = STLFormula.Predicate('y',operatorclass.le,3,INDEX_Y)
    spatialpref = STLFormula.Always( STLFormula.Negation( STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt1,predicate_x_le2) , STLFormula.Conjunction(predicate_y_gt1,predicate_y_le3) ) ),0,30) 
    phi = STLFormula.Conjunction(mission,spatialpref)
    #Define the target formula to learn
    phi_target = STLFormula.toNegationNormalForm(phi,False)


    #For the plots
    codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
         ]
    verts1_1 = [
        (1., 1.), # left, bottom
        (1., 3.), # left, top
        (2., 3.), # right, top
        (2., 1.), # right, bottom
        (0., 0.), # ignored
        ]
    path1_1 = Path(verts1_1, codes)
    patch1_1 = patches.PathPatch(path1_1, facecolor='mistyrose',lw=0)
    verts1_2 = [
        (3., 2.), # left, bottom
        (3., 3.), # left, top
        (4., 3.), # right, top
        (4., 2.), # right, bottom
        (0., 0.), # ignored
        ]
    path1_2 = Path(verts1_2, codes)
    patch1_2 = patches.PathPatch(path1_2, facecolor='honeydew',lw=0)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.tight_layout()
    ax.add_patch(patch1_1)
    ax.add_patch(patch1_2)
    plt.gcf().canvas.mpl_connect('key_release_event',
                                 lambda event: [exit(0) if event.key == 'escape' else None])
    plt.axis([rand_area[0]-0.2, rand_area[1]+0.2, rand_area[0]-0.2, rand_area[1]+0.2])
    plt.grid(True)
    plt.pause(0.01)
    
    
    #Instatiate the Active Learning algorithm
    active_learn = STLActiveLearn(phi_target,rand_area,dimensions,start,max_horizon,primitives='CLASSICAL',signal_gen='QUANTITATIVE_OPTIMIZE',U=0.2,epsilon=0.05,alpha=0.01,beta=0.5,gamma=50,MAX_IT=200,phi_hypothesis=STLFormula.TrueF(),plot_activated=True)
    
    print("\n\n Done ")
    
    print(active_learn.dtlearn.simple_boolean())
    print('distance between target and retrieved hypothesis',pompeiu_hausdorff_distance(phi_target,STLFormula.toNegationNormalForm(active_learn.dtlearn.toSTLformula(),False),rand_area,dimensions))

    exit()
