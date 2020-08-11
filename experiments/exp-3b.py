import sys
sys.path.insert(0,'..')
from STL import *
from STLGenerateSignal import *
from STLDistance import *
from STLDTLearn import *
from STLActiveLearn import *
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import pickle
import time



if __name__ == '__main__':

    #Constants
    rand_area = [0,4]
    dimensions = ['x','y']
    start=[0, 0]
    max_horizon = 30
    INDEX_X = 0
    INDEX_Y = 1

    #Define STL Formulae
    predicate_x_gt2 = STLFormula.Predicate('x',operatorclass.gt,2,INDEX_X)
    predicate_x_le3 = STLFormula.Predicate('x',operatorclass.le,3,INDEX_X)
    predicate_y_gt3 = STLFormula.Predicate('y',operatorclass.gt,3,INDEX_Y)
    predicate_y_le4 = STLFormula.Predicate('y',operatorclass.le,4,INDEX_Y)
    mission1 = STLFormula.Eventually(STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt2,predicate_x_le3) , STLFormula.Conjunction(predicate_y_gt3,predicate_y_le4) ),0,30)
    predicate_x_gt3 = STLFormula.Predicate('x',operatorclass.gt,3,INDEX_X)
    predicate_x_le4 = STLFormula.Predicate('x',operatorclass.le,4,INDEX_X)
    predicate_y_gt2 = STLFormula.Predicate('y',operatorclass.gt,2,INDEX_Y)
    predicate_y_le3 = STLFormula.Predicate('y',operatorclass.le,3,INDEX_Y)
    mission2 = STLFormula.Eventually(STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt3,predicate_x_le4) , STLFormula.Conjunction(predicate_y_gt2,predicate_y_le3) ),0,30)
    predicate_x_gt1 = STLFormula.Predicate('x',operatorclass.gt,1,INDEX_X)
    predicate_y_gt1 = STLFormula.Predicate('y',operatorclass.gt,1,INDEX_Y)
    predicate_x_le2 = STLFormula.Predicate('x',operatorclass.le,2,INDEX_X)
    predicate_y_le2 = STLFormula.Predicate('y',operatorclass.le,2,INDEX_Y)
    spatialpref = STLFormula.Always( STLFormula.Negation( STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt1,predicate_x_le2) , STLFormula.Conjunction(predicate_y_gt1,predicate_y_le2) ) ),0,30) 
    phi = STLFormula.Conjunction(STLFormula.Disjunction(mission1,mission2),spatialpref)
    phi_target = STLFormula.toNegationNormalForm(phi,False)


    #For the plots
    codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
         ]
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
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.tight_layout()
    ax.add_patch(patch3_1)
    ax.add_patch(patch3_2)
    ax.add_patch(patch3_3)
    plt.gcf().canvas.mpl_connect('key_release_event',
                                 lambda event: [exit(0) if event.key == 'escape' else None])
    plt.axis([rand_area[0]-0.2, rand_area[1]+0.2, rand_area[0]-0.2, rand_area[1]+0.2])
    plt.grid(True)
    plt.pause(0.01)
    
    
    #Instatiate the Active Learning algorithm
    begintime = time.time()
    # active_learn = STLActiveLearn(phi_target,rand_area,start,max_horizon,primitives='CLASSICAL',signal_gen='QUANTITATIVE_OPTIMIZE',U=0.2,epsilon=0.05,alpha=0.01,beta=0.5,gamma=50,MAX_IT=1000,phi_hypothesis=STLFormula.TrueF(),plot_activated=True)
    active_learn = STLActiveLearn(phi_target,rand_area,dimensions,start,max_horizon,primitives='CLASSICAL',signal_gen='QUANTITATIVE_OPTIMIZE',U=0.2,epsilon=0.05,alpha=0.01,beta=0.5,gamma=50,MAX_IT=1000,phi_hypothesis=STLFormula.Disjunction(mission1,mission2),plot_activated=True)
    # active_learn = STLActiveLearn(phi_target,rand_area,start,max_horizon,primitives='MOTION_PLANNING',signal_gen='QUANTITATIVE_OPTIMIZE',U=0.2,epsilon=0.05,alpha=0.01,beta=0.5,gamma=50,MAX_IT=1000,phi_hypothesis=STLFormula.TrueF(),plot_activated=True)
    
    print("\n\n Done ")
    
    print(active_learn.dtlearn.simple_boolean())
    print('distance between target and retrieved hypothesis',pompeiu_hausdorff_distance(phi_target,STLFormula.toNegationNormalForm(active_learn.dtlearn.toSTLformula(),False),rand_area,dimensions))
    endtime = time.time()
    pos = open("pos.pickle",'wb')
    neg = open("neg.pickle",'wb')
    dic = open("dic.pickle",'wb')
    dic2 = open("dic2.pickle",'wb')
    tim = open("tim.pickle",'wb')
    pickle.dump(active_learn.dtlearn.positive_dataset,pos)
    pickle.dump(active_learn.dtlearn.negative_dataset,neg)
    pickle.dump(active_learn.dic_results,dic)
    pickle.dump(active_learn.dtlearn.dictnodestr,dic2)
    pickle.dump(endtime-begintime,tim)
    pos.close()
    neg.close()
    dic.close()
    dic2.close()
    tim.close()
    exit()
