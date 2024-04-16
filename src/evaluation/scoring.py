import sys
import os
# import os.path
from importlib.util import spec_from_file_location, module_from_spec
import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
from summary import full_evaluation
import ml_collections
import yaml

# input_dir = sys.argv[1]
# output_dir = sys.argv[2]

input_dir = "../input"
output_dir = "../output"

config_dir = './config.yaml'
with open(config_dir) as file:
    config = ml_collections.ConfigDict(yaml.safe_load(file))

torch.manual_seed(config.seed)

# input_dir = "../input"
# output_dir = "../output"
submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')

if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a file to store scores
    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'w')

    # Load test data
    truth_file = os.path.join(truth_dir, "truth_crisis.pkl")
    if not os.path.isfile(truth_file):
        raise Exception('Data not supplied')
    with open(truth_file, "rb") as f:
        truth_crisis = torch.tensor(pickle.load(f)).float().to(config.device)

    truth_file = os.path.join(truth_dir, "truth_regular.pkl")
    if not os.path.isfile(truth_file):
        raise Exception('Data not supplied')
    with open(truth_file, "rb") as f:
        truth_regular = torch.tensor(pickle.load(f)).float().to(config.device)

    # Load fake data
    fake_file = os.path.join(submit_dir, "fake_crisis.pkl")
    if not os.path.isfile(fake_file):
        raise Exception('Data not supplied, make sure the zip file contains the file named fake_data.pkl')
    with open(fake_file, "rb") as f:
        fake_crisis = torch.tensor(pickle.load(f)).float().to(config.device)

    fake_file = os.path.join(submit_dir, "fake_regular.pkl")
    if not os.path.isfile(fake_file):
        raise Exception('Data not supplied, make sure the zip file contains the file named fake_data.pkl')
    with open(fake_file, "rb") as f:
        fake_regular = torch.tensor(pickle.load(f)).float().to(config.device)


    # Do final evaluation
    res_dict = full_evaluation(fake_crisis, truth_crisis, config)

    print(res_dict)

    # Write the result
    output_file.write("hist_mean_regular:%0.5f" % res_dict["hist_loss_mean"])
    output_file.write("\n")
    output_file.write("corr_mean_regular:%0.5f" % res_dict["cross_corr_mean"])
    output_file.write("\n")
    output_file.write("acf_mean_regular:%0.5f" % res_dict["acf_loss_mean"])
    output_file.write("\n")
    output_file.write("var_mean_regular:%0.5f" % res_dict["var_mean"])
    output_file.write("\n")
    output_file.write("es_mean_regular:%0.5f" % res_dict["es_mean"])
    output_file.write("\n")

    print(fake_regular.shape, truth_regular.shape)
    res_dict = full_evaluation(fake_regular, truth_regular, config)

    # Write the result
    output_file.write("hist_mean_crisis:%0.5f" % res_dict["hist_loss_mean"])
    output_file.write("\n")
    output_file.write("corr_mean_crisis:%0.5f" % res_dict["cross_corr_mean"])
    output_file.write("\n")
    output_file.write("acf_mean_crisis:%0.5f" % res_dict["acf_loss_mean"])
    output_file.write("\n")
    output_file.write("var_mean_crisis:%0.5f" % res_dict["var_mean"])
    output_file.write("\n")
    output_file.write("es_mean_crisis:%0.5f" % res_dict["es_mean"])
    output_file.write("\n")

    output_file.close()