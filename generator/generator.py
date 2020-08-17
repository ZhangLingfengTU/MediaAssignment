from utils import *

# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(_, _), (x_test, _) = mnist.load_data()

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

#normalize data
x_test = x_test.astype('float32')
x_test /= 255

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# some global variables
threshold = 0 # threshold for neuron coverage
num_parents = 50 # # of parent generation
num_parent_mutations = 100 # max # of pixils mutations in the parent generation
num_evolutions = 20 # number of the itertions of EA
p_crossover = 0.7
p_mutation = 0.2
p_deletion = 1-p_crossover-p_mutation





def generator():
    #input: num_evolutions
    #
    #rreturn: 
    #global variables: 
    

    # 1. generate the seed for mutations
    gen_img = random.choice(x_test) # of shape (28,28,1)
    #   expand the gen_img into (1,28,28,1)
    expanded_gen_img = np.empty((1,img_rows,img_cols,1))
    expanded_gen_img[0,...] = gen_img.copy #(1,28,28,1)

    # 2. get the original img
    original_img = expanded_gen_img.copy()

    #3. generate parent solutions
    parent_solutions = parent_generator(expanded_gen_img) #(num_parents, 28,28,1)

    #4. generate offspring solutions
    offspring_solutions = offspring_generator(parent_solutions, p_crossover, p_mutation, p_deletion)



    return 



def parent_generator(seed, num_parents = 50, num_parent_mutations = 100):
    #input : seed （1，28，28，1）
    #       num_parents   父亲数量
    #       num_parent_mutations   修改多少像素
    #return: generated_parent_solutions (num_parents, 28,28,1)
    # global variables : num_parents,
    generated_parent_solutions = []
    
    # 修改像素的范围
    pixil_range = []
    for a in range(0,101,1):
        a = a/100.0
        pixil_range.append(a)
    
    for i in range(0,num_parents,1):
        seed_copy = seed.copy() #(1,28,28,1)
        mutation_positions = random.sample(range(0,28*28),num_parent_mutations)
        for j in mutation_positions:
            #create a float number to represent the change of pixl
            seed_copy[0][j//28][j%28][0] = random.sample(pixil_range,1)[0]  
        generated_parent_solutions.append(seed_copy)
    return generated_parent_solutions

def offspring_generator(parent_solutions, p_crossover, p_mutation, p_deletion):


    return offspring_solutions