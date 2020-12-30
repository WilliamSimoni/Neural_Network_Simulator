"""
    Module to create output project file and train final model with blind data
"""
import csv 
from grid_search import final_model
from utility import normalize_data, read_cup_data

blind_data, blind_label, _, _ = read_cup_data("dataset/ML-CUP20-TR.csv", 1)
 
def create_output_file(file_path, blind_predicted):
    print("Predicted first ")
    print(blind_predicted[0])
    with open(file_path, 'w') as result_file:
            writer = csv.writer(result_file)
            writer.writerow('# Marco Natali, William Simoni')
            writer.writerow('# Scarsenal')
            writer.writerow('# ML-CUP20')
            writer.writerow('# 31/12/2020')
            
            for index, elem in enumerate(blind_predicted):
                print(elem)
                writer.writerow([
                    str(index + 1),
                    elem[0],
                    elem[1],
                ])

create_output_file("Scarsenal_ML-CUP20-TS.csv", final_model().predict(blind_data))