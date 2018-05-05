#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""
Usage: $ python2.7 create_outfit_combos.py ./polyvore/test_no_dup.json ./data/in_items_test.txt ./data/out_items_test.txt
"""

import json
from itertools import combinations
import sys
import io

def itemsToStr(outfit_id, id_array):
  string = ""
  for id in id_array:
    string += str(outfit_id) + "_" + str(id) + " "

  return string + "\n" 

# print item names
def arrayPrint(array):
  for item in array:
    print " - ", item


json_data_path_in = str(sys.argv[1])
input_items_path_out = str(sys.argv[2])
output_items_path_out = str(sys.argv[3])

# Open the json file
test_json = json.load(open(json_data_path_in))
max_outfits = len(test_json)
print "\nOutfits in this dataset: ", max_outfits

# Create the output files
input_items_file = open(input_items_path_out, 'w')
output_items_file = open(output_items_path_out, 'w') 


for i in range(0, max_outfits):
  outfit = test_json[i]
  outfit_id = outfit["set_id"]
  items = outfit["items"]
  item_indices = []

  # Get the actual indicies of the items
  for item in items:
    item_indices.append(item["index"])
  
  # Iterate through different combinations (combinations of 1, 2, ... len-1)
  for nr_of_input_items in range(1, len(item_indices)):

    # Get the combination indicies
    combo_idxs_array = list(combinations(item_indices, nr_of_input_items))

    # Get the combos
    for input_idxs in combo_idxs_array:
      output_idxs = list(set(item_indices) - set(input_idxs))

      # Write the input and output combinations as strings (outftID_itemID outftID_itemID outftID_itemID)
      input_items_file.write(itemsToStr(outfit_id, list(input_idxs)))
      output_items_file.write(itemsToStr(outfit_id, output_idxs))

# Close the files
input_items_file.close()
output_items_file.close()  
        

