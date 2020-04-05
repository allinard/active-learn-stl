from STL import *
from STLpulp import *
from STLDistance import *
from STLdtlearn import *
import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import pickle
import time

#Constants
rand_area = [0,4]
start=[0, 0]
max_horizon = 30
MAX_IT = 1000

INDEX_X = 0
INDEX_Y = 1

"""
EXAMPLE 1
"""
predicate_x_gt3 = STLFormula.Predicate('x',operatorclass.gt,3,INDEX_X)
predicate_x_le4 = STLFormula.Predicate('x',operatorclass.le,4,INDEX_X)
predicate_y_gt2 = STLFormula.Predicate('y',operatorclass.gt,2,INDEX_Y)
predicate_y_le3 = STLFormula.Predicate('y',operatorclass.le,3,INDEX_Y)
eventually = STLFormula.Always(STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt3,predicate_x_le4) , STLFormula.Conjunction(predicate_y_gt2,predicate_y_le3) ),25,30)
predicate_x_gt1 = STLFormula.Predicate('x',operatorclass.gt,1,INDEX_X)
predicate_y_gt1 = STLFormula.Predicate('y',operatorclass.gt,1,INDEX_Y)
predicate_x_le2 = STLFormula.Predicate('x',operatorclass.le,2,INDEX_X)
predicate_y_le3 = STLFormula.Predicate('y',operatorclass.le,3,INDEX_Y)
always = STLFormula.Always( STLFormula.Negation( STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt1,predicate_x_le2) , STLFormula.Conjunction(predicate_y_gt1,predicate_y_le3) ) ),0,30) 
phi = STLFormula.Conjunction(eventually,always)
phi_target = STLFormula.toNegationNormalForm(phi,False)




"""
EXAMPLE 2
"""
# predicate_x_gt3 = STLFormula.Predicate('x',operatorclass.gt,3,INDEX_X)
# predicate_x_le4 = STLFormula.Predicate('x',operatorclass.le,4,INDEX_X)
# predicate_y_gt1 = STLFormula.Predicate('y',operatorclass.gt,1,INDEX_Y)
# predicate_y_le3 = STLFormula.Predicate('y',operatorclass.le,3,INDEX_Y)
# eventually = STLFormula.Eventually(STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt3,predicate_x_le4) , STLFormula.Conjunction(predicate_y_gt1,predicate_y_le3) ),0,30)
# predicate_x_gt1 = STLFormula.Predicate('x',operatorclass.gt,1,INDEX_X)
# predicate_y_gt05 = STLFormula.Predicate('y',operatorclass.gt,0.5,INDEX_Y)
# predicate_x_le2 = STLFormula.Predicate('x',operatorclass.le,2,INDEX_X)
# predicate_y_le35 = STLFormula.Predicate('y',operatorclass.le,3.5,INDEX_Y)
# always = STLFormula.Always( STLFormula.Negation( STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt1,predicate_x_le2) , STLFormula.Conjunction(predicate_y_gt05,predicate_y_le35) ) ),0,30) 
# phi = STLFormula.Conjunction(eventually,always)
# phi_target = STLFormula.toNegationNormalForm(phi,False)





"""
EXAMPLE 3
"""
# predicate_x_gt2 = STLFormula.Predicate('x',operatorclass.gt,2,INDEX_X)
# predicate_x_le3 = STLFormula.Predicate('x',operatorclass.le,3,INDEX_X)
# predicate_y_gt3 = STLFormula.Predicate('y',operatorclass.gt,3,INDEX_Y)
# predicate_y_le4 = STLFormula.Predicate('y',operatorclass.le,4,INDEX_Y)
# eventually1 = STLFormula.Eventually(STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt2,predicate_x_le3) , STLFormula.Conjunction(predicate_y_gt3,predicate_y_le4) ),0,30)
# predicate_x_gt3 = STLFormula.Predicate('x',operatorclass.gt,3,INDEX_X)
# predicate_x_le4 = STLFormula.Predicate('x',operatorclass.le,4,INDEX_X)
# predicate_y_gt2 = STLFormula.Predicate('y',operatorclass.gt,2,INDEX_Y)
# predicate_y_le3 = STLFormula.Predicate('y',operatorclass.le,3,INDEX_Y)
# eventually2 = STLFormula.Eventually(STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt3,predicate_x_le4) , STLFormula.Conjunction(predicate_y_gt2,predicate_y_le3) ),0,30)
# predicate_x_gt1 = STLFormula.Predicate('x',operatorclass.gt,1,INDEX_X)
# predicate_y_gt1 = STLFormula.Predicate('y',operatorclass.gt,1,INDEX_Y)
# predicate_x_le2 = STLFormula.Predicate('x',operatorclass.le,2,INDEX_X)
# predicate_y_le2 = STLFormula.Predicate('y',operatorclass.le,2,INDEX_Y)
# always = STLFormula.Always( STLFormula.Negation( STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt1,predicate_x_le2) , STLFormula.Conjunction(predicate_y_gt1,predicate_y_le2) ) ),0,30) 
# phi = STLFormula.Conjunction(STLFormula.Disjunction(eventually1,eventually2),always)
# phi_target = STLFormula.toNegationNormalForm(phi,False)




"""
EXAMPLE 4
"""
# max_horizon = 50
# predicate_x_gt0 = STLFormula.Predicate('x',operatorclass.gt,0,INDEX_X)
# predicate_x_le1 = STLFormula.Predicate('x',operatorclass.le,1,INDEX_X)
# predicate_y_gt3 = STLFormula.Predicate('y',operatorclass.gt,3,INDEX_Y)
# predicate_y_le4 = STLFormula.Predicate('y',operatorclass.le,4,INDEX_Y)
# eventually1 = STLFormula.Eventually(STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt0,predicate_x_le1) , STLFormula.Conjunction(predicate_y_gt3,predicate_y_le4) ),10,20)
# predicate_x_gt3 = STLFormula.Predicate('x',operatorclass.gt,3,INDEX_X)
# predicate_x_le4 = STLFormula.Predicate('x',operatorclass.le,4,INDEX_X)
# predicate_y_gt0 = STLFormula.Predicate('y',operatorclass.gt,0,INDEX_Y)
# predicate_y_le1 = STLFormula.Predicate('y',operatorclass.le,1,INDEX_Y)
# eventually2 = STLFormula.Eventually(STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt3,predicate_x_le4) , STLFormula.Conjunction(predicate_y_gt0,predicate_y_le1) ),30,50)
# predicate_x_gt2 = STLFormula.Predicate('x',operatorclass.gt,2,INDEX_X)
# predicate_y_gt1 = STLFormula.Predicate('y',operatorclass.gt,1,INDEX_Y)
# predicate_x_le3 = STLFormula.Predicate('x',operatorclass.le,3,INDEX_X)
# predicate_y_le2 = STLFormula.Predicate('y',operatorclass.le,2,INDEX_Y)
# always = STLFormula.Always( STLFormula.Negation( STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt2,predicate_x_le3) , STLFormula.Conjunction(predicate_y_gt1,predicate_y_le2) ) ),0,30) 
# phi = STLFormula.Conjunction(STLFormula.Disjunction(eventually1,eventually2),always)
# phi_target = STLFormula.toNegationNormalForm(phi,False)







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
verts2_1 = [
    (1., 0.5), # left, bottom
    (1., 3.5), # left, top
    (2., 3.5), # right, top
    (2., 0.5), # right, bottom
    (0., 0.), # ignored
    ]
path2_1 = Path(verts2_1, codes)
patch2_1 = patches.PathPatch(path2_1, facecolor='mistyrose',lw=0)
verts2_2 = [
    (3., 1.), # left, bottom
    (3., 3.), # left, top
    (4., 3.), # right, top
    (4., 1.), # right, bottom
    (0., 0.), # ignored
    ]
path2_2 = Path(verts2_2, codes)
patch2_2 = patches.PathPatch(path2_2, facecolor='honeydew',lw=0)
verts3_1 = [
    (1., 1.), # left, bottom
    (1., 2.), # left, top
    (2., 2.), # right, top
    (2., 1.), # right, bottom
    (0., 0.), # ignored
    ]
path3_1 = Path(verts3_1, codes)
patch3_1 = patches.PathPatch(path3_1, facecolor='mistyrose',lw=0)
verts3_2 = [
    (2., 3.), # left, bottom
    (2., 4.), # left, top
    (3., 4.), # right, top
    (3., 3.), # right, bottom
    (0., 0.), # ignored
    ]
path3_2 = Path(verts3_2, codes)
patch3_2 = patches.PathPatch(path3_2, facecolor='honeydew',lw=0)
verts3_3 = [
    (3., 2.), # left, bottom
    (3., 3.), # left, top
    (4., 3.), # right, top
    (4., 2.), # right, bottom
    (0., 0.), # ignored
    ]
path3_3 = Path(verts3_3, codes)
patch3_3 = patches.PathPatch(path3_3, facecolor='honeydew',lw=0)
verts4_1 = [
    (0., 3.), # left, bottom
    (0., 4.), # left, top
    (1., 4.), # right, top
    (1., 3.), # right, bottom
    (0., 0.), # ignored
    ]
path4_1 = Path(verts4_1, codes)
patch4_1 = patches.PathPatch(path4_1, facecolor='honeydew',lw=0)
verts4_2 = [
    (2., 1.), # left, bottom
    (2., 2.), # left, top
    (3., 2.), # right, top
    (3., 1.), # right, bottom
    (0., 0.), # ignored
    ]
path4_2 = Path(verts4_2, codes)
patch4_2 = patches.PathPatch(path4_2, facecolor='mistyrose',lw=0)
verts4_3 = [
    (3., 0.), # left, bottom
    (3., 1.), # left, top
    (4., 1.), # right, top
    (4., 0.), # right, bottom
    (0., 0.), # ignored
    ]
path4_3 = Path(verts4_3, codes)
patch4_3 = patches.PathPatch(path4_3, facecolor='honeydew',lw=0)




"""
#plot
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.tight_layout()
ax.add_patch(patch1_1)
ax.add_patch(patch1_2)
# ax.add_patch(patch4_3)
plt.gcf().canvas.mpl_connect('key_release_event',
                             lambda event: [exit(0) if event.key == 'escape' else None])
plt.axis([rand_area[0]-0.2, rand_area[1]+0.2, rand_area[0]-0.2, rand_area[1]+0.2])
plt.grid(True)
plt.pause(0.01)
"""

        
        
#Instance of the DT learning algorithm
dtlearn = DTLearn(rand_area,max_horizon)

#Start hypothesis: True
phi_hypothesis = STLFormula.TrueF()
phi_hypothesis.horizon = max_horizon


#if eventually already known
# dtlearn.tree = DTLearn.Node(STLFormula.Eventually(predicate_x_gt3,0,30),DTLearn.Node(STLFormula.Eventually(predicate_x_le4,0,30),DTLearn.Node(STLFormula.Eventually(predicate_y_gt2,0,30),DTLearn.Node(STLFormula.Eventually(predicate_y_le3,0,30),DTLearn.Leaf(STLFormula.TrueF()),DTLearn.Leaf(STLFormula.FalseF())),DTLearn.Leaf(STLFormula.FalseF())),DTLearn.Leaf(STLFormula.FalseF())),DTLearn.Leaf(STLFormula.FalseF()))
# phi_hypothesis = eventually
# phi_hypothesis.horizon = max_horizon

# pos = open("pos.pickle",'rb')
# neg = open("neg.pickle",'rb')
# dtlearn.positive_dataset = pickle.load(pos)
# dtlearn.negative_dataset = pickle.load(neg)
# pos.close()
# neg.close()

# invarnode = DTLearn.Leaf(STLFormula.FalseF())
# # invarnode.elements = dtlearn.negative_dataset
# truenode = DTLearn.Leaf(STLFormula.TrueF())
# # truenode.elements = dtlearn.positive_dataset
# dtlearn.tree = DTLearn.Node(eventually,truenode,invarnode)
# dtlearn.invarnode = invarnode

# phi_hypothesis = eventually
# phi_hypothesis.horizon = max_horizon



dic_results = {}

begintime = time.time()


NO_IMPROVE = 0
LAST_DIST = float("inf")
BEST_HYPOTHESIS = STLFormula.TrueF()
BEST_TREE = dtlearn.tree


for it in range(1,MAX_IT):
   
   
    #Decide if go for membership or equivalence query
    if random.random() > 0.6:
        """
            EQUIVALENCE QUERY
        """
       
        print("\niteration",it, "-- Equivalence Query")
        
        #translate hypothesis into negation normal form
        phi_hypothesis_nnf = STLFormula.toNegationNormalForm(STLFormula.simplify_dtlearn(phi_hypothesis),False)
        # print("phi_hypothesis_nnf",phi_hypothesis_nnf)
        #compute distance between hypothesis and target
        if isinstance(phi_hypothesis_nnf,STLFormula.TrueF):
            distance = 4.0
        else:
            distance = pompeiu_hausdorff_distance(phi_hypothesis_nnf,phi_target,rand_area)
        # print("distance: ",distance)
        #generate counterexample
        if distance > 0.0001:
            #compute symmetric difference of target and hypothesis
            xor = STLFormula.Xor(phi_hypothesis_nnf,phi_target)
            #compute trajectory
            if random.random() > 0.5:
                trajectory = generate_trajectory_boolean_milp(xor,start,rand_area)
            else:
                trajectory = generate_trajectory_milp(xor,start,rand_area,True)
            #let the teacher evaluate    
            ro = round(phi_target.robustness(trajectory,0), 5)
            print("ro",ro)
            
            #update hypothesis
            if ro < -0.0:
                # print("add to negatives")
                print("update")
                dtlearn.update(trajectory,STLFormula.FalseF())
                print("update")
                # plt.plot([x for (x, y) in trajectory], [y for (x, y) in trajectory], '-r')
                # plt.grid(True)
                # plt.pause(0.01)
            else:
                # print("add to positives")
                print("update")
                dtlearn.update(trajectory,STLFormula.TrueF())
                print("update")
                # plt.plot([x for (x, y) in trajectory], [y for (x, y) in trajectory], '-g')
                # plt.grid(True)
                # plt.pause(0.01)
            phi_hypothesis = dtlearn.toSTLformula()
            print("hypothesis",STLFormula.simplify_dtlearn(phi_hypothesis).tex())
            # print("simplebool",dtlearn.simple_boolean())
            if isinstance(phi_hypothesis,STLFormula.TrueF):
                distance = 4.0
            else:
                distance = pompeiu_hausdorff_distance(STLFormula.toNegationNormalForm(STLFormula.simplify_dtlearn(phi_hypothesis),False),phi_target,rand_area)
            print("distance: ",distance)
        #if distance is 0, then the hypothesis and the target are equivalent. termination of algorithm.
        else:
            print("solution found")
            print(STLFormula.simplify_dtlearn(phi_hypothesis).tex())
            break
        
    else:    
        """
            MEMBERSHIP QUERY
        """
        
        print("\niteration",it, "-- Membership Query")
        
        #translate hypothesis into negation normal form
        phi_hypothesis_nnf = STLFormula.toNegationNormalForm(STLFormula.simplify_dtlearn(phi_hypothesis),False)
        # print("phi_hypothesis_nnf",phi_hypothesis_nnf)
        #generate trajectory belonging to hypothesis
        phi_hypothesis_nnf.horizon = max_horizon
        try:
            if isinstance(phi_hypothesis_nnf,STLFormula.TrueF):
                trajectory = generate_trajectory_boolean_milp(phi_hypothesis_nnf,start,rand_area)
            else:
                trajectory = generate_trajectory_milp(phi_hypothesis_nnf,start,rand_area,True)
        except Exception:
            leaf = DTLearn.Leaf(STLFormula.TrueF())
            leaf.elements.extend(dtlearn.negative_dataset)
            leaf.elements.extend(dtlearn.positive_dataset)
            dtlearn.tree = leaf
            phi_hypothesis = STLFormula.TrueF()
            phi_hypothesis.horizon = max_horizon
            NO_IMPROVE = 0
            LAST_DIST = float("inf")
            continue
        # trajectory = generate_trajectory_boolean_milp(phi_hypothesis_nnf,start,rand_area)
        #let the teacher evaluate
        ro = round(phi_target.robustness(trajectory,0), 5)
        # print("ro",ro)
                   
        #update hypothesis
        if ro<0.0:
            # print("add to negatives")
            print("update")
            dtlearn.update(trajectory,STLFormula.FalseF())
            print("update")
            # plt.plot([x for (x, y) in trajectory], [y for (x, y) in trajectory], '-r')
            # plt.grid(True)
            # plt.pause(0.01)
        else:
            # print("add to positives")
            print("update")
            dtlearn.update(trajectory,STLFormula.TrueF())
            print("update")
            # plt.plot([x for (x, y) in trajectory], [y for (x, y) in trajectory], '-g')
            # plt.grid(True)
            # plt.pause(0.01)
        phi_hypothesis = dtlearn.toSTLformula()
        print("hypothesis",STLFormula.simplify_dtlearn(phi_hypothesis).tex())
        # print("simplebool",dtlearn.simple_boolean())
        if isinstance(phi_hypothesis,STLFormula.TrueF):
            distance = 4.0
        else:
            distance = pompeiu_hausdorff_distance(STLFormula.toNegationNormalForm(STLFormula.simplify_dtlearn(phi_hypothesis),False),phi_target,rand_area)
        print("distance: ",distance)

    
    if distance < LAST_DIST:
        NO_IMPROVE = 0
        LAST_DIST = distance
    # elif NO_IMPROVE > 50 or distance > LAST_DIST:
    elif NO_IMPROVE > 50:
        leaf = DTLearn.Leaf(STLFormula.TrueF())
        leaf.elements.extend(dtlearn.negative_dataset)
        leaf.elements.extend(dtlearn.positive_dataset)
        dtlearn.tree = leaf
        phi_hypothesis = STLFormula.TrueF()
        phi_hypothesis.horizon = max_horizon
        NO_IMPROVE = 0
        LAST_DIST = float("inf")
    else:
        NO_IMPROVE += 1   

    print(len(dtlearn.positive_dataset),len(dtlearn.negative_dataset))
    dic_results[it] = (distance,phi_hypothesis.tex(),dtlearn.toSimply(dtlearn.tree))


endtime = time.time()

pos = open("pos.pickle",'wb')
neg = open("neg.pickle",'wb')
dic = open("dic.pickle",'wb')
dic2 = open("dic2.pickle",'wb')
tim = open("tim.pickle",'wb')
pickle.dump(dtlearn.positive_dataset,pos)
pickle.dump(dtlearn.negative_dataset,neg)
pickle.dump(dic_results,dic)
pickle.dump(dtlearn.dictnodestr,dic2)
pickle.dump(endtime-begintime,tim)
pos.close()
neg.close()
dic.close()
dic2.close()
tim.close()
# plt.savefig('4.pdf')