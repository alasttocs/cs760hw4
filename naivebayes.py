import os
import re
import shutil
import sys
import numpy as np
import math
import pandas as pd


def bag(file):
    global bag_dict
    global files_dict
    f = open(file, "r")
    data = f.read()
    f.close()

    # get the first character of the file
    lang_index = file.find("/") + 1
    first_char = file[lang_index]
    files_dict[first_char] += 1
    # print(file)
    for char in data:
        if char == " ":
            bag_dict[first_char][26] += 1
        elif char.isalpha():
            bag_dict[first_char][ord(char) - 97] += 1
    # for key in bag_dict.keys():
    #     print(f"{key}: {bag_dict[key]}\n")
    # print(f"files: {files_dict}\n")
    return


def get_priors(smoothing_factor):
    global files_dict
    prior = {}
    total = sum(files_dict.values())
    for key in files_dict.keys():
        prior[key] = (files_dict[key] + smoothing_factor) / \
            (total + smoothing_factor * 3)
    return prior


def get_class_conditionals(smoothing_factor):
    global bag_dict
    global files_dict
    class_conditionals = {"e": np.zeros(
        27), "j": np.zeros(27), "s": np.zeros(27)}
    total = sum(bag_dict.values())
    for key in bag_dict.keys():
        class_conditionals[key] = (
            bag_dict[key] + smoothing_factor)/(total + smoothing_factor * 27)
    return class_conditionals


def get_prob(x, class_conditional):
    prob = {"e": 0, "j": 0, "s": 0}
    for key in class_conditional.keys():
        for i in range(27):
            prob[key] += x[i] * math.log(class_conditional[key][i])
    return prob


def get_posterior(prob, priors):
    posterior = {"e": 0, "j": 0, "s": 0}
    for key in prob.keys():
        posterior[key] = prob[key] + math.log(priors[key])
    return posterior


bag_dict = {"e": np.zeros(27), "j": np.zeros(27), "s": np.zeros(27)}
files_dict = {"e": 0, "j": 0, "s": 0}
prior_dict = {"e": 0, "j": 0, "s": 0}


# q3
q1_path = "q1/"
if not os.path.exists(q1_path):
    os.makedirs(q1_path)
count = 0
for file in os.listdir("languageID"):
    if re.match(r'[e,j,s]\d\Wtxt', file):
        shutil.copy(f"languageID/{file}", q1_path)
        count += 1
for file in os.listdir(q1_path):
    bag(f"{q1_path}{file}")
priors = get_priors(.5)
print(priors)
class_conditionals = get_class_conditionals(.5)
for key in class_conditionals.keys():
    print(f"{key}: {class_conditionals[key]}\n")

# p4
f = open("languageID/e10.txt", "r")
data = f.read()
f.close()
file_bag = np.zeros(27)
for char in data:
    if char == " ":
        file_bag[26] += 1
    elif char.isalpha():
        file_bag[ord(char) - 97] += 1
print(f"x = {file_bag}\n")

# p5
prob = get_prob(file_bag, class_conditionals)
print(f"prob = {prob}\n")

# p6
posterior = get_posterior(prob, priors)
print(f"posterior = {posterior}\n")

max_val = max(posterior.values())
for key in posterior.keys():
    if posterior[key] == max_val:
        print(f"Predicted Class: {key}\n")
        break
# p7
count = 0
q7_path = "q7/"
if not os.path.exists(q7_path):
    os.makedirs(q7_path)
for file in os.listdir("languageID"):
    if re.match(r'[e,j,s]\d\d\Wtxt', file):
        shutil.copy(f"languageID/{file}", q7_path)
confusion_matrix = prior_dict = {
    "e": np.zeros(3), "j": np.zeros(3), "s": np.zeros(3)}
for file in os.listdir(q7_path):
    f = open(f"q7/{file}", "r")
    data = f.read()
    f.close()
    file_bag = np.zeros(27)
    lang_index = file.find("/") + 1
    first_char = file[lang_index]
    for char in data:
        if char == " ":
            file_bag[26] += 1
        elif char.isalpha():
            file_bag[ord(char) - 97] += 1
    print(f"file: {file}")
    print(f"x = {file_bag}\n")
    prob = get_prob(file_bag, class_conditionals)
    print(f"prob = {prob}\n")
    posterior = get_posterior(prob, priors)
    print(f"posterior = {posterior}\n")
    max_val = max(posterior.values())
    for key in posterior.keys():
        if posterior[key] == max_val:
            print(f"Predicted Class: {key}\n")
            if key == "e":
                confusion_matrix[first_char][0] += 1
            elif key == "j":
                confusion_matrix[first_char][1] += 1
            else:
                confusion_matrix[first_char][2] += 1
            break
print(confusion_matrix)

df = pd.DataFrame(confusion_matrix, index=[
                  'e', 'j', 's'], columns=['e', 'j', 's'])
print(df)
