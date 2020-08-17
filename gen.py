#from utils import *
from keras.datasets import mnist
from keras.layers import Input
import math
# importing cv2  
import cv2 
  
# importing os module   
import os 


import matplotlib.pyplot as plt
from scipy.misc import imsave

import random
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.models import Model
from Model1 import Model1
from Model2 import Model2
from Model3 import Model3

# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(_, _), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

#normalize data
x_test = x_test.astype('float32')
x_test /= 255

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

model1 = Model1(input_tensor=input_tensor)
model2 = Model2(input_tensor=input_tensor)
model3 = Model3(input_tensor=input_tensor)

def parent_generator(seed, num_parents = 100, num_parent_mutations = 50):
    #input : seed  (28，28，1）
    #       num_parents   父亲数量
    #       num_parent_mutations   修改多少像素
    #return: generated_parent_solutions (num_parents, 28,28,1)
    # global variables : num_parents,
    generated_parent_solutions = np.empty((num_parents,28,28,1))
    
    # 修改像素的范围
    pixil_range = []
    for a in range(0,101,1):
        a = a/100.0
        pixil_range.append(a)
    
    for i in range(0,num_parents,1):
        seed_copy = seed.copy() #(28,28,1)
        mutation_positions = random.sample(range(0,28*28),num_parent_mutations)
        for j in mutation_positions:
            #create a float number to represent the change of pixl
            seed_copy[j//28][j%28][0] = random.sample(pixil_range,1)[0]  
        generated_parent_solutions[i,...] = seed_copy
    return generated_parent_solutions #(num_parents, 28,28,1)

def offspring_generator(parent_solutions, p_crossover, p_mutation, p_deletion, original_img, num_mutation_pixil = 50,num_deletion_pixil = 50):
    # input: parent_solutions(#parent_solutions, 28,28,1)
    #        original_img （28，28，1）
    # return offspring_solutions (#parent_solutions, 28,28,1)
    
    #初始化offspring_solutions
    offspring_solutions = np.empty((parent_solutions.shape[0],28,28,1))
    num_crossover = int(len(parent_solutions)*p_crossover)
    num_mutations = int(len(parent_solutions)*p_mutation)
    num_deletion = parent_solutions.shape[0] - num_crossover - num_mutations
    
    
    crossover_solutions = np.empty((num_crossover,28,28,1))
    mutation_solutions = np.empty((num_mutations,28,28,1))
    deleted_solutions = np.empty((num_deletion,28,28,1))
    
    
    for i in range(num_crossover):
        
        parent1, parent2 = random.sample(list(parent_solutions),2) #both(28,28,1) 
        crossover_solutions[i,...] = crossover_generator(parent1,parent2)
        #crossover_solutions.append(crossover_generator(parent1,parent2)) 
    
    for i in range(num_mutations):
        parent = random.choice(parent_solutions) #(28,28,1)
        #mutation_solutions.append(mutation_generator(parent))
        mutation_solutions[i,...] = mutation_generator(parent, num_mutation_pixil)
    
    for i in range(num_deletion):
        parent = random.choice(parent_solutions) #(28,28,1)
        deleted_solutions[i,...] = deletion_generator(parent, original_img, num_deletion_pixil)#------------------------------------
        
    offspring_solutions = np.concatenate((crossover_solutions, mutation_solutions, deleted_solutions), axis = 0)
    
        
        
        
    return offspring_solutions


def crossover_generator(parent1, parent2):
    #input: both(28,28,1)
    #output: (28,28,1)
    # parent1 head， parent2 tail
    
    #reshape parent1&parent2
    parent1 = parent1.reshape(28*28)
    parent2 = parent2.reshape(28*28)
    
    #create a new crossover_solution
    crossover_solution = np.empty(28*28)
    for i in range(28*28):
        if i < int(28*28/2):
            crossover_solution[i] = parent1[i]
        else:
            crossover_solution[i] = parent2[i]
    return crossover_solution.reshape((28,28,1))
    

def mutation_generator(parent, num_mutations = 10):
    #input: parent of (28,28,1)
    #return: mutated img (28,28,1)
    # mutate the parent img by num_mutations pixls
    
    
    # 修改像素的范围
    pixil_range = []
    for a in range(0,101,1):
        a = a/100.0
        pixil_range.append(a)
    #修改的位置
    mutation_positions = random.sample(range(0,28*28,1), num_mutations)
    
    #reshape 数据
    parent_copy = parent.copy() #(28,28,1)
    
    #修改数据
    for i in mutation_positions:
        parent_copy[i//28][i%28][0] = random.sample(pixil_range,1)[0]
    
    return parent_copy
    
def deletion_generator(parent, original_solution, num_deletion = 10):
    #input: parent (28,28,1)
    #      original_solution (28,28,1)
    #return: one deletion_solution (28,28,1)
    #delete some pixl changes 
    
    #create deletion set
    deletion_set = []
    for i in range(28*28):
        if parent[i//28][i%28][0] != original_solution[i//28][i%28][0]:
            deletion_set.append(i)
    
    #get deletion positions
    if len(deletion_set)>=num_deletion:
        deletion_positions = random.sample(deletion_set, num_deletion)
    else:
        deletion_positions = deletion_set
    
    #delete
    for i in deletion_positions:
        parent[i//28][i%28][0] = original_solution[i//28][i%28][0]
    
    return parent #(28,28,1)
# 将解转化成object值
def ObjMapping(solutions, model1, model2, model3, original_solution, threshold = 0.5):
    # compute neuron coverage, differences of behariors, mutation times to form the multi-objective functions
    # the input is in shape (solution.shape[0],28,28,1) of solutions
    # the input of original_solution is (28,28,1)
    # return [(index, object1, object2, object3)] of len #solutions
    
    # create mapped_solutions with length equal to solution.shape[0]
    # stores a list (index, object1, object2, object3)
    #mapped_solutions = []
    mapped_solutions = np.empty((solutions.shape[0],3))
    
    # compute each object values
    for i, solution in enumerate(solutions):
        #(28,28,1)
        
        #expand the solution with one dim and then compute the coverage
        #默认计算Model1的
        neuron_coverage = compute_neuron_coverage(solution, model1, threshold )
        # compute the divergence
        #divergence = compute_divergence_sensei(solution,model1,model2,model3) #--------------------------------
        divergence = compute_divergence(solution,model1,model2,model3)
        
        #compute the #mutations
        num_mutations = int(compute_mutations(solution, original_solution))
        
        mapped_solutions[i][0] = -neuron_coverage
        #mapped_solutions[i][1] = -divergence  #----------------------------------------------------------------
        mapped_solutions[i][1] = -divergence
        mapped_solutions[i][2] = num_mutations
        
    
    return mapped_solutions
def compute_mutations(new_solution, original_solution):
    # input both in (28,28,1)
    new_solution_copy = new_solution.reshape(28*28)
    new_original_solution = original_solution.reshape(28*28)
    num_mutations = 0
    for i in range(0,len(new_solution_copy),1):
        if new_solution_copy[i] != new_original_solution[i]:
            num_mutations = num_mutations +1
    return num_mutations

#计算divergence
def compute_divergence(input_data, model1, model2, model3):
    # 计算三个模型的预测差异
    # 输入（28，28，1）
    # 输出差异值， int类型
    expanded_input = np.empty((1,28,28,1))
    expanded_input[0,...] = input_data
    label1, label2, label3 = model1.predict(expanded_input)[0], model2.predict(expanded_input)[0], model3.predict(expanded_input)[0]
    divergence = 0
    for i in range(0,10,1):
        d12 = abs(label1[i]-label2[i])
        d23 = abs(label2[i]-label3[i])
        d13 = abs(label1[i]-label3[i])
        divergence = divergence + d12+d23+d13
    return divergence

#计算divergence
def compute_divergence_sensei(input_data, model1, model2, model3):
    # 计算三个模型的预测差异
    # 输入（28，28，1）
    # 输出差异值， int类型
    expanded_input = np.empty((1,28,28,1))
    expanded_input[0,...] = input_data
    label1, label2, label3 = model1.predict(expanded_input)[0], model2.predict(expanded_input)[0], model3.predict(expanded_input)[0]
    
    cos_theta = np.sum((label1-label2)*(label1-label3))/(d(label1,label2)*d(label1,label3))
    
    divergence = ((d(label1,label2)*d(label1,label3))*math.sqrt(1-math.pow(cos_theta,2)))/2.0
    
    return divergence

def d(a,b):
    #compute Euclidean distance
    return math.sqrt(np.square(a+b).sum())


#计算coverage
def compute_neuron_coverage(input_data, model, threshold = 0):
    # input (28,28,1)
    
    #expande the input_data
    expanded_input_data = np.empty((1,28,28,1))
    expanded_input_data[0,...] = input_data
    #create a dictionary
    model_layer_dict = defaultdict(bool)
    #intialize the dict
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False
            
    layer_names = [layer.name for layer in model.layers if 'flatten' not in layer.name and 'input' not in layer.name]

    #update table
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(expanded_input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])       
        for num_neuron in range(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True
    
    #compute coverage
    covered_neurons = len([v for v in model_layer_dict.values() if v])

    total_neurons = len(model_layer_dict)

    #return covered_neurons, total_neurons, covered_neurons / float(total_neurons)
    return covered_neurons / float(total_neurons)
                
def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled

def dominance(solution_object1, solution_object2):
    #input : two solutions with object , both shape (3)
    #return: true if solution1 dominates solution2 解1比解2牛逼
    #       false if solution1 not dominates solution 2 解1不比解2牛逼，不暗示解2比解1牛逼
    #minimize!!!!
    notbad = False
    if (solution_object1[0]<=solution_object2[0])&(solution_object1[1]<=solution_object2[1])&(solution_object1[2]<=solution_object2[2]):
        notbad = True
    better = False
    if notbad:
        if (solution_object1[0]<solution_object2[0])|(solution_object1[1]<solution_object2[1])|(solution_object1[2]<solution_object2[2]):
            better = True
    return better
        
def non_dominant_sort(mapped_solutions):
    #input: mapped_solutions (#solutions, 3)
    #return: (index of solution, rank) of len(#mapped_solutions)
    
    length = mapped_solutions.shape[0]
    n = np.zeros((length), dtype = np.int)  #比 i解牛逼的解数
    s = []
    P = []
    rank = [] # the one I need to return
    
    for i in range(length):
        s.append([])
    #create the two entities
    for i in range(length):
        for j in range(length):
            if i == j:
                continue
            else:
                if dominance(mapped_solutions[j],mapped_solutions[i]):
                    n[i] = n[i]+1
                if dominance(mapped_solutions[i],mapped_solutions[j]):
                    s[i].append(j)
    #两件事，P, 创建返回的数据
    for i in range(length):
        if n[i] == 0:
            P.append(i)
                
    for i in range(length):
        if n[i] == 0:
            rank.append((i,0))
            n[i] = n[i]-1
    #开始排序
    iteration = 1
    while len(P) != 0:
        Q = []
        for i in P:
            for j in s[i]:
                n[j] = n[j] -1
                if n[j] == 0:
                    if j not in Q:
                        Q.append(j)
        for i in Q:
            rank.append((i,iteration))
            n[i] = n[i]-1
        P = Q
        iteration = iteration+1
    
    return rank
def crowd_distance_sort(sub_mapped_solutions, sub_solutions, num_needed_solutions):
    #input: sub_mapped_solutions (#sub_mapped_solutions, 3) list
    #       sub_solutions (#subsolutions, 28,28,1) np array
    #return: solutions with num_needed_solutions number
    #默认sub_mapped_solutions和sub_solutions是一一对应的
    
    
    crowd_distance_table = np.zeros((sub_solutions.shape[0], 3),dtype = float) #与输入意义对应
    #avaraged_crowd = np.zeros((sub_solutions.shape[0], 1),dtype = float)
    selected_solutions = np.empty((num_needed_solutions,28,28,1)) #输出的解
    
    index = []
    index_range = range(sub_solutions.shape[0])
    for i in index_range:
        index.append(i)
    
    for i in range(3):
        obj_sorted_index = np.argsort(sub_mapped_solutions[...,i])
        crowd_distance_table[obj_sorted_index[0]][i] = float('inf')
        crowd_distance_table[obj_sorted_index[-1]][i] = float('inf')
        max_obj = np.max(sub_mapped_solutions[...,i])
        min_obj = np.min(sub_mapped_solutions[...,i])
        scale = max_obj-min_obj
        if scale == 0.0:
            scale = 1.0
        for j in index[1:-1]:
            crowd_distance_table[(obj_sorted_index[j]),i] = (crowd_distance_table[(obj_sorted_index[j]),i])+( (sub_mapped_solutions[(obj_sorted_index[j+1]),i])-(sub_mapped_solutions[(obj_sorted_index[j-1]),i]))/(scale)
            #print(crowd_distance_table[obj_sorted_index[j]][i])
    
    averaged_crowd = (crowd_distance_table[...,0]+crowd_distance_table[...,1]+crowd_distance_table[...,2])/3
        
    sorted_crowd_index = np.argsort(averaged_crowd)
    list1 = []
    for i in sorted_crowd_index[::-1]:
        list1.append(i)
    
    for i in range(num_needed_solutions):
        selected_solutions[i,...] = sub_solutions[list1[i],...]
    return selected_solutions

def evolution(parent_solutions, original_img,label, iteration):
    #input: parent solutions
    #return : optimal solutions
    
    needed_solution_number = num_parents
     
    for generation in range(num_evolutions):
        offspring_solutions = offspring_generator(parent_solutions, p_crossover, p_mutation, p_deletion, original_img,10,10)
        combined_solutions =  np.concatenate((parent_solutions, offspring_solutions), axis=0, out=None)
        mapped_solutions = ObjMapping(combined_solutions, model1, model2, model3, original_img, threshold)
        
        
        index_for_parents = 0
        
        
        rank = non_dominant_sort(mapped_solutions)
        
        
        for i in range(rank[-1][1]+1):
            candidates = [a[0] for a in rank if a[1] == i]
            num_candidates = len(candidates)
            #够
            if needed_solution_number >= num_candidates:
                for j in candidates:
                    parent_solutions[index_for_parents,...] = combined_solutions[j]
                    index_for_parents = index_for_parents +1
                    needed_solution_number = needed_solution_number-1
            else:
                #sub_solutions
                sub_solutions = np.empty((num_candidates,28,28,1))
                index_for_subsolutions = 0
                for q in candidates:
                    sub_solutions[index_for_subsolutions,...] = combined_solutions[q]
                    index_for_subsolutions = index_for_subsolutions+1
                 
                #sub_mapped_solutions
                sub_mapped_solutions = np.empty((num_candidates,3))
                index_sub_mapped = 0
                for q in candidates:    
                    sub_mapped_solutions[index_sub_mapped,...] = mapped_solutions[q]
                    index_sub_mapped = index_sub_mapped+1
                
                selected_solutions = crowd_distance_sort(sub_mapped_solutions, sub_solutions, needed_solution_number)
                for selected_solution in selected_solutions:
                    parent_solutions[index_for_parents,...] = selected_solution
                    index_for_parents = index_for_parents+1
                break
    
    optiaml_solutions = parent_solutions
    optimal_mapped = ObjMapping(optiaml_solutions, model1, model2, model3, original_img, threshold)
    rank = non_dominant_sort(optimal_mapped)
    
    optimal_solutions_index = [a[0] for a in rank if a[1] == 0]
    for i in optimal_solutions_index:
        if optimal_mapped[i][2] == 0:
            continue
        else:
            
            img_to_save = np.empty((1,28,28,1))
            img_to_save[0,...] = optiaml_solutions[i]
            img_to_save = deprocess_image(img_to_save)
            filename = './results/0/'+str(iteration)+'_'+ str(i)+'_'+str(-optimal_mapped[i,0])+'_'+str(-optimal_mapped[i,1])+'_'+str(optimal_mapped[i,2])+'_'+str(label)+'.png'
            cv2.imwrite(filename, img_to_save) 
            print('已保存',iteration,i)

def deprocess_image(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2])  # original shape (1,img_rows, img_cols,1)


# some global variables
threshold = 0.5 # threshold for neuron coverage
num_parents = 100 # # of parent generation
num_parent_mutations = 20 # max # of pixils mutations in the parent generation
num_evolutions = 20 # number of the itertions of EA
p_crossover = 0.7 #crossover 的概率
p_mutation = 0.25 # 修改的概率
p_deletion = 1-p_crossover-p_mutation #消除修改的概率
num_mutation_pixil = 10 #一次突变的像素数
num_deletion_pixil = 5

for i in range(2):
    seed_index = random.randint(0,x_test.shape[0]-1)
    print(seed_index)
    original_img = x_test[seed_index]
    original_label = y_test[seed_index]
    #plt.imshow(original_img.reshape(28*28).reshape((28,28)), cmap = plt.cm.binary)
    #plt.show()
    print(original_label)
    parent_solutions = parent_generator(original_img, num_parents, num_parent_mutations)
    #print(parent_solutions.shape)
    evolution(parent_solutions, original_img,original_label,i)