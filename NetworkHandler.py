#####################################
#                                   #
#           NETWORK HANDLER         #
#                                   #
#####################################

"""
Script for handling networks and visualization
-set Global values for desired networks and bool vals
    for visualization and output
"""

import Grader_0thLevel_255x255 as NN
import random_tensor as rt
import tensorflow as tf
import tensor_vis as tv
from PIL import Image
import random, os, gc, pickle
import numpy as np
import matplotlib.pyplot as plt

#sess = tf.Session()

# MEAN and STD for normalizing H&E slides
# training data should already be normalized, but it doesn't
#   hurt to keep track of these vals 
MEAN = 160.24859619140625 
STD = 47.6563720703125
"""
GRADE_ONE_COUNT = 100
GRADE_TWO_COUNT = 63
GRADE_THREE_COUNT = 142
"""
GRADE_ONE_COUNT = 154551
GRADE_TWO_COUNT = 370567
GRADE_THREE_COUNT = 236675

PATCH_SIZE = "255" #The input dimensions for the network
SLIDE_LEVEL = "0"   # the slide level the network has been trained for
NET_ID = "5" # determines appended ID of desired network

#DATA_PATH = "D:/Grading_slides/{}_level/{}x{}/".format(SLIDE_LEVEL, PATCH_SIZE, PATCH_SIZE)
DATA_PATH = "Grading_slides/0_level/255x255/H&ENormed/train/"
#DATA_PATH = "Grading_slides/0_level/255x255/H&ENormed/test_set/"
EVAL_PATH = "Grading_slides/0_level/255x255/H&ENormed/eval/"
#EVAL_PATH = DATA_PATH
MODEL_PATH = "Grading_slides/{}_level/{}_grading_net_{}/".format(SLIDE_LEVEL, PATCH_SIZE,
                                                                    NET_ID)
                      
VIZ_DEST = "Grading_slides/{}_level/{}_grading_net_{}_viz/".format(SLIDE_LEVEL, PATCH_SIZE,
                                                                    NET_ID)
PREDICT_DEST = "Grading_slides/{}_level/{}_grading_net_{}_predictions/".format(SLIDE_LEVEL, PATCH_SIZE,
                                                                    NET_ID)
EVAL_RESULTS_DEST = "Grading_slides/{}_level/{}_grading_net_{}_eval_results.txt".format(SLIDE_LEVEL, PATCH_SIZE,
                                                                    NET_ID)

ALREADY_TRAINED_DEST = "Grading_slides/{}_level/{}_grading_net_{}_already_trained.pickle".format(SLIDE_LEVEL, PATCH_SIZE,
                                                                    NET_ID)
CONFUSION_DEST = "Grading_slides/{}_level/{}_grading_net_{}_confusion_matrix.npy".format(SLIDE_LEVEL, PATCH_SIZE,
                                                                    NET_ID)
DATA_CONFUSION_DEST = "Grading_slides/{}_level/{}_grading_net_{}_data_confusion_matrix.npy".format(SLIDE_LEVEL, PATCH_SIZE,
                                                                    NET_ID)
if not os.path.exists(VIZ_DEST):
    os.makedirs(VIZ_DEST)

if not os.path.exists(PREDICT_DEST):
    os.makedirs(PREDICT_DEST)

if not os.path.exists(ALREADY_TRAINED_DEST):
    #trained_file = open(ALREADY_TRAINED_DEST, "w")
    TRAINED_DICT = {'fully_trained':set({}), 'grade_one':GRADE_ONE_COUNT, 'grade_two':GRADE_TWO_COUNT, 'grade_three':GRADE_THREE_COUNT, 'epoch' : 0}
else:
    print("loading pickle")
    trained_file = open(ALREADY_TRAINED_DEST, "rb")
    TRAINED_DICT = pickle.load(trained_file)
    trained_file.close()

if not os.path.exists(CONFUSION_DEST):
    #create numpy matrix
    CONFUSION_MATRIX = np.zeros((3,3))
else:
     #load numpy matrix
    CONFUSION_MATRIX = np.load(CONFUSION_DEST)

if not os.path.exists(DATA_CONFUSION_DEST):
    #create numpy matrix
    DATA_CONFUSION_MATRIX = np.zeros((3,3))
else:
     #load numpy matrix
    DATA_CONFUSION_MATRIX = np.load(DATA_CONFUSION_DEST)

tf.logging.set_verbosity(tf.logging.INFO)

NETWORK = tf.estimator.Estimator(
    model_fn= NN.cnn_model_fn, model_dir = MODEL_PATH)
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors = tensors_to_log, every_n_iter = 50)
#
# Load Data from images
# PARAMETERS
#   png_src, string: folder location of data to be loaded
#   slide_count, int: number of slides to load for each grade
#   patch_count, int: number of patches to load from each slide
#
#GLOBAL VAR
#   TRAINED_DICT, dictionary: dictionary of patches that the network has already been trained on
#                     dictionary has format of {['folder_name']:{patch1, patch2, . .. patchN}}
# RETURNS
#   (data_set, label_set), tuple: numpy arrays of mean normalized images
#                           and respective labels
#
def load_from_im_track_train(png_src, slide_count, patch_count, randomize = False):

    data_set = []
    label_set = []

    # for each grade we want to get a random amount patches depending on the following formula
    # (R * f)/S * T
    # where R = a random number [0.0, 1.0), f is the ratio of patches in grade x  to the total patch count
    # S is the sum of the numberators (R * f), and T is the total amount of desired patches
    grades = ['grade_one', 'grade_two', 'grade_three']
    grade_total = TRAINED_DICT['grade_one'] + TRAINED_DICT['grade_two'] + TRAINED_DICT['grade_three']
    ratio_one = TRAINED_DICT['grade_one'] / grade_total
    ratio_two = TRAINED_DICT['grade_two'] / grade_total
    ratio_three = TRAINED_DICT['grade_three'] / grade_total
    patch_total = 3 * slide_count * patch_count
    grade_limits = [random.random() * ratio_one,random.random() * ratio_two,random.random() * ratio_three]
    limit_sum = grade_limits[0] + grade_limits[1] + grade_limits[2]
    grade_limits[0] = (grade_limits[0]/limit_sum) * patch_count
    grade_limits[1] = (grade_limits[1]/limit_sum) * patch_count
    grade_limits[2] = (grade_limits[2]/limit_sum) * patch_count
    print("patches totals\ngrade one: {}\tgrade two: {}\tgrade three: {}".format(grade_limits[0], grade_limits[1], grade_limits[2]))
    #print("slide count: {}".format(slide_count))

    # for each grade folder, f, in png_src (train or eval)
    for f in os.listdir(png_src):
        grade = int(f.strip("_alt.png"))
        s_list = set(os.listdir(png_src + f))
        s_list = list(s_list.difference(TRAINED_DICT['fully_trained']))
        random.shuffle(s_list)
        slide_counter = 0
        # for each slide, s, in s_list (list of slides for grade folder, f
        if TRAINED_DICT[grades[grade]] > 0 and len(s_list) > 0:
            for s in s_list:
                # s isn't in TRAINED_DICT yet, create a set for it
                if s not in TRAINED_DICT:
                    TRAINED_DICT[s] = set({})
                i_list = set(os.listdir("{}{}/{}".format(png_src, f, s)))
                i_list = list(i_list.difference(TRAINED_DICT[s]))
                # if i_list is greater than 0, there are still patches left to
                # train on
                if len(i_list) > 0:
                    random.shuffle(i_list)
                    im_counter = 0
                    for i in i_list:
                        # add current patch to list of loaded patches
                        TRAINED_DICT[s].add(i)
                        # decrement current grade's patch total by one
                        TRAINED_DICT[grades[grade]] -= 1
                        im = Image.open("{}{}/{}/{}".format(png_src, f, s, i))
                        if randomize:
                           im = randomize_image(im)
                        data_set.append(np.asarray(im))
                        label_set.append(grade)
                        im.close()
                        im_counter += 1
                        if im_counter > grade_limits[grade] or TRAINED_DICT[grades[grade]] < 1:
                        #if im_counter >  int(grade_limits[grade_counter] / slide_count):
                            break
                else:
                    TRAINED_DICT['fully_trained'].add(s)
                slide_counter += 1
                if slide_counter == slide_count:
                    break
        else:
            print("\nLooks like grade {} has no slides left: {}\n".format(grade, TRAINED_DICT[grades[grade]]))
            if TRAINED_DICT[grades[grade]] < 150:
                TRAINED_DICT[grades[grade]] = 0
            else:
                print("/nthat's weird! S_list is {} and TRAINED_DICT[{}] is {}/n".format(s_list, grades[grade], TRAINED_DICT[grades[grade]]))
                usr_cont = input("set current trained_dict? (y/n)")
                if usr_cont == 'y':
                    TRAINED_DICT[grades[grade]] = 0
        #grade_counter += 1
    #print(TRAINED_DICT) 
    data_set = np.asarray(data_set, dtype = "float32")
    label_set = np.asarray(label_set, dtype = "int32")

    return (data_set, label_set)

def load_from_im(png_src, slide_count, patch_count, randomize = False):
    #f_list = random.shuffle(list(os.listdir(png_src)))
    #s_folder = int(random.random() * (len(f_list) - set_count))
    data_set = []
    label_set = []
    for f in os.listdir(png_src):
        grade = int(f.strip("_alt.png"))
        s_list = list(os.listdir(png_src + f))
        random.shuffle(s_list)
        if (slide_count > len(s_list)):
            slide_count = len(s_list)
        for s in range(slide_count):
            i_list = list(os.listdir("{}{}/{}".format(png_src, f, s_list[s])))
            random.shuffle(i_list)
            if patch_count > len(i_list):
                patch_count = len(i_list)
            for i in range(patch_count):
                im = Image.open("{}{}/{}/{}".format(png_src, f, s_list[s], i_list[i]))
                if randomize:
                   im = randomize_image(im)
                data_set.append(np.asarray(im))
                label_set.append(grade)
                im.close()
    data_set = np.asarray(data_set, dtype = "float32")
    label_set = np.asarray(label_set, dtype = "int32")

    return (data_set, label_set)

def check_nan(png_src):
    f_list = random.shuffle(list(os.listdir(png_src)))
    data_set = []
    label_set = []
    for f in os.listdir(png_src):
        grade = int(f.strip("_alt.png"))
        s_list = list(os.listdir(png_src + f))
        for s in s_list:
            print("checking : {}{}/{}".format(png_src, f, s))
            i_list = list(os.listdir("{}{}/{}".format(png_src, f, s)))
            for i in i_list:
                im = Image.open("{}{}/{}/{}".format(png_src, f, s, i))
                im_arr = (np.asarray(im, dtype="float32") - MEAN) / STD
                im.close()
                if (np.any(np.isnan(im_arr))):
                    print("{}{}/{}/{}   has nan".format(png_src, f, s, i))

#
#   TRAIN
#   trains a network
#
def train(train_data, train_labels, batch_size = 50,
          shuffle = True, train_steps = None):
    if len(train_labels) < batch_size:
        batch_size = len(train_labels)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x":train_data},
        y = train_labels,
        batch_size = batch_size,
        num_epochs = 1,
        shuffle = shuffle)
    NETWORK.train(
        input_fn = train_input_fn,
        steps = train_steps,
        hooks = [logging_hook])
    

#
#   EVALUATE
#   evaluates a network for accuracy and loss
#
def evaluate(eval_data, eval_labels):
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=True)
    #return NETWORK.evaluate(input_fn=eval_input_fn, steps = steps)
    return NETWORK.evaluate(input_fn=eval_input_fn)

#
#   PREDICT
#   returns predictions for a data set
def predict(data, labels, batch_size = 50, shuffle = True,
             num_epochs = 1, steps = 1):
    # Evaluate the model and print results
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": data},
        y=labels,
        batch_size = batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle)
    return NETWORK.predict(input_fn=pred_input_fn)

def retrain(error_data, error_labels, batch_size = 10, retrain_steps = 1, shuffle = True):
     train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x":error_data[:, :, :, 0:3]},
        y = error_labels,
        batch_size = batch_size,
        num_epochs = None,
        shuffle = shuffle)
     NETWORK.train(
         input_fn = train_input_fn,
         steps = retrain_steps,
         hooks = [logging_hook])
    

#
#   VISUALIZE
#   method for visualizing model kernels and saving them out
def visualize(window_viz, save_viz, save_dest = None):
    if not(window_viz or save_viz):
        print("Whoops, visualize() is doing nothing at all. Check bool vals.")
    elif save_viz and not save_dest:
        print("Whoops, you want to save out but there's no destination")
    else:
        checkpoint = NETWORK.latest_checkpoint().lstrip(MODEL_PATH)
        for i in NETWORK.get_variable_names():
            if ("kernel" in i) and ("conv2d" in i):
                print(i.strip("/kernel"))
                im = tv.plot_conv_weights(NETWORK.get_variable_value(i))
                if window_viz:
                    im.show()
                if save_viz:
                    im.savefig("{}{}_{}.png".format(save_dest, checkpoint, i.strip("/kernel")))

#
#   RANDOMIZE_DATA
#   adds noise to data according to the standard deviation
#
#   params
#   data, numpy array: array of image data to be randomized
#   returns
#   data, numpy array: randomized array of data

def randomize_data(data):
    rand_plus = np.random.uniform(-1.0, 1.0, data.shape) #array of random values [-1.0, 1.0), same shape as image data set
    std_fill = np.ndarray(data.shape, dtype = "float32")    # array filled with the image population's Standard Deviation
    std_fill.fill(STD)
    data += std_fill * rand_plus #multiply the standard deviation by random vals in the range [1.0, 1.0) and add to original image data
    data[data<0] = 0    # image values to [0, 255]
    data[data>255] = 255
    return data

#
# RANDOMIZE_IMAGE
# randomly flip an image horizontally and/or vertically, rotate 90/180/270 degrees
#
# PARAMS
# im, Image: the image to be randomized
# RETURNS
# im, Image: the randomized image

def randomize_image(im):
    horizontal = random.random()
    vertical = random.random()
    rotate = random.random()
    if horizontal > 0.5:
        # flip horizontally
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if vertical > 0.5:
        # flip vertically
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    if rotate >= 0.0 and rotate < 0.25:
        im = im.rotate(90)
    if rotate >= 0.25 and rotate < 0.5:
        im = im.rotate(180)
    if rotate >= 0.5 and rotate < 0.75:
        im = im.rotate(270)
    return im

def epoch_done():
    epoch = TRAINED_DICT['epoch']
    # append output that new epoch is starting to prec_rec and eval
    # too much work right now to actually print out the proper epoch but it's
    # easy enough to manually adjust after the fact
    prec_rec_txt = open("{}/precision_recall.txt".format(PREDICT_DEST), "a")
    prec_rec_txt.write("\nEpoch {}\n*********\n".format(epoch))
    prec_rec_txt.close()
    eval_txt = open(EVAL_RESULTS_DEST, "a")
    eval_txt.write("\nEpoch {}\n*********\n".format(epoch))
    eval_txt.close()
    # back up current iteration of confusion matrix 
    this_confusion_dest = "Grading_slides/{}_level/{}_grading_net_{}_epoch_{}_confusion_matrix.npy".format(SLIDE_LEVEL, PATCH_SIZE,
                                                                                                                  NET_ID, epoch)

    np.save(this_confusion_dest, CONFUSION_MATRIX)

    this_confusion_dest = "Grading_slides/{}_level/{}_grading_net_{}_epoch_{}_data_confusion_matrix.npy".format(SLIDE_LEVEL, PATCH_SIZE,
                                                                                                                  NET_ID, epoch)
    np.save(this_confusion_dest, DATA_CONFUSION_MATRIX)
    # reset TRAINED_DICT and load fresh data
    #print("\nbefore clear:\n{}\n".format(TRAINED_DICT))    
    TRAINED_DICT.clear()
    #print("\nafter clear:\n{}\n".format(TRAINED_DICT))
    TRAINED_DICT['fully_trained'] = set({})
    TRAINED_DICT['grade_one'] = GRADE_ONE_COUNT
    TRAINED_DICT['grade_two'] = GRADE_TWO_COUNT
    TRAINED_DICT['grade_three'] = GRADE_THREE_COUNT
    TRAINED_DICT['epoch'] = epoch + 1

    print("\nPickling TRAINED_DICT\nSlides Completed: {}\n".format(len(TRAINED_DICT['fully_trained'])))
    trained_file = open(ALREADY_TRAINED_DEST, 'wb')
    pickle.dump(TRAINED_DICT, trained_file, protocol = pickle.HIGHEST_PROTOCOL)
    trained_file.close()


def prediction_output(pred_run_limit, folder_count, patch_count, Random_Data, save_pred_im = False,
                      update_confusion = False):
    #precision and recall, pr[x][y]
    # x is grade, y is true positive (0), false positive (0), false negative(0)
    
    flipper = 0
    while flipper < 2:
        if flipper == 1:
            pr_header = "TRAINING DATA RESULTS\n"
            predict_path = DATA_PATH
        else:
            pr_header = "VALIDATION DATA RESULTS\n"
            predict_path = EVAL_PATH
        pred_run = 0
        print("predicting for {}".format(pr_header))
        while pred_run < pred_run_limit:
            print("\nStarting Prediction Run #{}\n".format(pred_run))
            predict_data, predict_labels = load_from_im(predict_path,
                                                        folder_count,
                                                        patch_count,
                                                        randomize = Random_Data)
            predict_data = (predict_data - MEAN) / STD
            pred_results = predict(predict_data, predict_labels, batch_size = 50)

            preds = []
            error_labels = []
            for p in enumerate(pred_results):
                #print(p)
                conf = p[1]['probabilities'][p[1]['classes']]
                label = predict_labels[p[0]]
                guess = p[1]['classes']
                if save_pred_im:
                    predict_data = (predict_data * STD) + MEAN
                    predict_data = np.asarray(predict_data, dtype = "uint8")
                    im = Image.fromarray(predict_data[p[0]])
                    im_save_path = "{}/{}/{}_{}_{}.png".format(PREDICT_DEST, label,
                                                               guess, conf, p[0])
                    im.save(im_save_path, "PNG")
                #if guess == label:
                    #confidences[label][0].append(conf)
                    #prec_rec[label][0] += 1.0
                #else:
                    #confidences[label][1].append(conf)
                    #prec_rec[guess][1] += 1.0
                    #prec_rec[label][2] += 1.0
                if (predict_path == EVAL_PATH) and update_confusion:
                    CONFUSION_MATRIX[label, guess] += 1
                elif (predict_path == DATA_PATH) and update_confusion:
                    DATA_CONFUSION_MATRIX[label, guess] += 1
            del(predict_data, predict_labels, pred_results)
            pred_run += 1
        flipper += 1
    #print("\nPickling TRAINED_DICT\nSlides Completed: {}\n".format(len(TRAINED_DICT['fully_trained'])))
    #trained_file = open(ALREADY_TRAINED_DEST, 'wb')
    #pickle.dump(TRAINED_DICT, trained_file, protocol = pickle.HIGHEST_PROTOCOL)
    #trained_file.close()
    if (update_confusion):
        np.save(CONFUSION_DEST, CONFUSION_MATRIX)
    """
    prec_rec_txt = open("{}/precision_recall.txt".format(PREDICT_DEST), "a")
    prec_rec_txt.write(pr_header)
    label = 1
    for grade in prec_rec:
        try:
            precision = grade[0] / (grade[0] + grade[1])
            recall = grade[0] / (grade[0] + grade[2])
            prec_rec_txt.write("{}\tprecision {}\trecall: {}\taccuracy: {}\n".format(label,precision,
                                                                                     recall))
        except:
            prec_rec_txt.write("Failed to calculate precision/recall\n")
        label += 1
    prec_rec_txt.write("\n")
    prec_rec_txt.close()
    """

def eval_output(eval_steps, eval_folder_count, eval_patch_count, Random_Data):
    eval_txt = open(EVAL_RESULTS_DEST, "a")
    eval_size = eval_steps * eval_folder_count * eval_patch_count
    acc_sum = 0
    loss_sum = 0
    results = []
    for i in range(eval_steps):
        try:
            eval_data, eval_labels = load_from_im(EVAL_PATH, eval_folder_count, eval_patch_count)
                    #if Random_Data:
                    #    eval_data = randomize_data(eval_data)
            eval_data = (eval_data - MEAN) / STD
            results.append(evaluate(eval_data, eval_labels))
            acc_sum += results[i]["accuracy"]
            loss_sum += results[i]["loss"]
        except:
            print("crashed on results . . . at {}".format(i))
    eval_txt.write("Global Step: {}\tAccuracy: {}\tLoss: {}\tEval Size: {}\n".format(results[0]["global_step"],
                                                                                     (acc_sum/eval_steps),
                                                                                     (loss_sum/eval_steps),
                                                                                     eval_size))
    eval_txt.close()
    
##################
#                #
#  MAIN METHOD   #
#                #
##################
def main(unused_argv):
    print(TRAINED_DICT)
    # Train Settings
    Train_Model = True
    train_steps = 25
    folder_count = 15
    patch_count = 20
    batch_size = 50

    #Predict Settings
    Predict_Model = False
    #predict_steps = 100
    #predict_path = DATA_PATH
    save_pred_im = False
    pred_run_limit = 50
    nth_pred = 500
    
    # Eval Settings
    Eval_Model = False
    nth_eval = 500
    eval_steps = 100
    eval_folder_count = 5
    eval_patch_count = 5

    # Visualize Model
    Viz_Model = False
    viz_step = 500 # save out viz every viz_step'th

    # Randomize Settings
    Random_Data = True
    # load data
    
    Done = False
    run = 0
    run_limit = 1
    eval_sums = []
    train_file = 0
    increment = True
    #while run < run_limit:
    run_prediction = False
    run_eval = False
    new_epoch = False
    while not Done:
        if Predict_Model and (run % nth_pred == 0):
            run_prediction = True
        if Eval_Model and (run % nth_eval == 0):
            run_eval = True
        print("\n***************\nEntering Run #{}\n***************\n".format(run))
        if (TRAINED_DICT['grade_one'] + TRAINED_DICT['grade_two'] + TRAINED_DICT['grade_three'])  <= 0:
            print("Epoch {}  Completed".format(TRAINED_DICT['epoch']))
            new_epoch = True
            run_prediction = True
            update_confusion = True
            run_eval = True
                #epoch_done()
                #train_data, train_labels = load_from_im_track_train(DATA_PATH, folder_count, patch_count, randomize = Random_Data)
            
            
        print("one left: {}\ttwo left : {}\tthree left: {}".format(TRAINED_DICT['grade_one'],
                                                                   TRAINED_DICT['grade_two'],
                                                                   TRAINED_DICT['grade_three']))
            # really we just needed to load them to check whether the epoch was done
            # so delete them for now and they'll get reloaded at the actual train time
            #del(train_data, train_labels)

        if run_prediction:
            print("\nmaking prediction\n")
            prediction_output(pred_run_limit, folder_count, patch_count, Random_Data,
                              update_confusion = update_confusion)
            gc.collect()
            run_prediction = False
            update_confusion = False
            
        if Viz_Model and (run % viz_step == 0) and run != 0:
            visualize(False, True, VIZ_DEST)
        
    
        #eval_index = 0
        if run_eval:
            print("\nmaking eval\n")
            eval_output(eval_steps, eval_folder_count, eval_patch_count, Random_Data)
            #input("continue?")
            run_eval = False
        if new_epoch:
            print("\nnew epoch!\n")
            print("you need to update the training rate!")
            epoch_done()
            new_epoch = False
            Done = True
        if (Train_Model):
            train_data, train_labels = load_from_im_track_train(DATA_PATH, folder_count, patch_count, randomize = Random_Data)
            counters = [0, 0, 0]
            for i in train_labels:
                counters[i] += 1
            print("\nlabel counts: {}\n".format(counters))
            print("\ntrain data: {}\ttrain labels: {}\n".format(train_data.shape, train_labels.shape))
            train_data = (train_data - MEAN) / STD
            if len(train_data) > 0:
                train(train_data,
                      train_labels,
                      batch_size = batch_size,
                      train_steps = train_steps)
            del(train_data, train_labels)
        if run % 5 == 0:
            # back up already saved
            print("\nPickling TRAINED_DICT\nSlides Completed: {}\n".format(len(TRAINED_DICT['fully_trained'])))
            trained_file = open(ALREADY_TRAINED_DEST, 'wb')
            pickle.dump(TRAINED_DICT, trained_file, protocol = pickle.HIGHEST_PROTOCOL)
            trained_file.close()
            #np.save(CONFUSION_DEST, CONFUSION_MATRIX)

            #
        run += 1
    #sess.close()
                        
if __name__ == "__main__":
    tf.app.run()
