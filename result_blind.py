"""
    Module to create output project file and train final model with blind data
"""
import csv 
from grid_search import final_model
from utility import read_blind_data

blind_index, blind_data = read_blind_data("dataset/ML-CUP20-TS.csv")
 
def create_output_file(file_path, blind_predicted):
    with open(file_path, 'w') as result_file:
            writer = csv.writer(result_file)
            writer.writerow(['# Marco Natali, William Simoni'])
            writer.writerow(['# Scarsenal'])
            writer.writerow(['# ML-CUP20'])
            writer.writerow(['# 31/12/2020'])
            
            for index, elem in zip(blind_index, blind_data):
                writer.writerow([
                    str(index),
                    elem[0],
                    elem[1],
                ])

create_output_file("Scarsenal_ML-CUP20-TS.csv", final_model().predict(blind_data))