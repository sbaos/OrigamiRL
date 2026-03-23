1. setup
uv sync
2. run

To generate a new pattern, run the following command:

python main.py

the old_map_value is used to store the unique map values from existing patterns. If you want to generate a new pattern and not same with existing patterns, please use get_all_map_value_in_folder("folder_name") to get all map values in the folder.

- edge_extend_as_posible: is used to extend the edges as possible when generating a new node.

3. edut/view map

python utils/mesheditor.py

4. change_name

python utils/change_name.py

to change the name of the pattern in folder [A,B,C] and store to folder D
