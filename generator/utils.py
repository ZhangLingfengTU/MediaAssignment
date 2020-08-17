
import random
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.models import Model
from keras.datasets import mnist
from keras.layers import Input
from Model1 import Model1
from Model2 import Model2
from Model3 import Model3



# 将解转化成object值
def ObjMapping(solutions, model1, model2, model3, original_solution):
    # compute neuron coverage, differences of behariors, mutation times to form the multi-objective functions
    # the input is in shape (solution.shape[0],28,28,1) of solutions
    # the input of original_solution is (1,28,28,1)
    # return [(index, object1, object2, object3)] of len #solutions
    
    # create mapped_solutions with length equal to solution.shape[0]
    # stores a list (index, object1, object2, object3)
    #mapped_solutions = []
    mapped_solutions = np.empty((solutions.shape[0],3))
    
    # compute each object values
    expanded_solution_copy = np.empty((1, img_rows, img_cols, 1))
    for i, solution in enumerate(solutions):
        #(28,28,1)
        
        #expand the solution with one dim and then compute the coverage
        #默认计算Model1的
        expanded_solution_copy[0,...] = solution.copy()
        neuron_coverage = compute_neuron_coverage(expanded_solution_copy, model1 )
        # compute the divergence
        divergence = compute_divergence(expanded_solution_copy,model1,model2,model3)
        
        #compute the #mutations
        num_mutations = compute_mutations(expanded_solution_copy, original_solution)
        
        mapped_solutions[i][0] = neuron_coverage
        mapped_solutions[i][1] = divergence
        mapped_solutions[i][2] = num_mutations
        
    
    return mapped_solutions

#计算coverage
def compute_neuron_coverage(input_data, model, threshold = 0):
    # input (1,28,28,1)
    
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
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

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


#计算divergence
def compute_divergence(input_data, model1, model2, model3):
    # 计算三个模型的预测差异
    # 输入（1，28，28，1）
    # 输出差异值， int类型
    label1, label2, label3 = model1.predict(input_data)[0], model2.predict(input_data)[0], model3.predict(input_data)[0]
    divergence = 0
    for i in range(0,10,1):
        d12 = abs(label1[i]-label2[i])
        d23 = abs(label2[i]-label3[i])
        d13 = abs(label1[i]-label3[i])
        divergence = divergence + d12+d23+d13
    return divergence

#计算修改了几个像素点
def compute_mutations(new_solution, original_solution):
    # input both in (1,28,28,1)
    new_solution_copy = new_solution[0].reshape(28*28)
    new_original_solution = original_solution[0].reshape(28*28)
    num_mutations = 0
    for i in range(0,len(new_solution_copy),1):
        if new_solution_copy[i] != new_original_solution[i]:
            num_mutations = num_mutations +1
    return num_mutations