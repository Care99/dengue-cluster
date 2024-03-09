import data
import folder
import graph
import os
import shutil
script_directory = os.path.dirname(os.path.abspath(__file__))

check_folder_name='raw_clusters'
check_folder=os.path.join(script_directory,check_folder_name)
if(os.path.isdir(check_folder)):
    shutil.rmtree(check_folder)

check_folder_name='clusters'
check_folder=os.path.join(script_directory,check_folder_name)
if(os.path.isdir(check_folder)):
    shutil.rmtree(check_folder)

check_matrix_name='cluster_matrix.xlsx'
check_matrix=os.path.join(script_directory,check_matrix_name)
if(os.path.isfile(check_matrix)):
    os.remove(check_matrix)

check_cluster_name='cluster_cluster.svg'
check_cluster=os.path.join(script_directory,check_cluster_name)
if(os.path.isfile(check_cluster)):
    os.remove(check_cluster)

folder.folder()
graph.graph()
data.data()