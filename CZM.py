import pandas as pd
import numpy as np
from itertools import groupby,islice
from numpy import genfromtxt

##############################################################################
##############################################################################

# Step 1: Parsing the original inp file to the 'Node sets' and 'Element Sets'

## 1 - Reading the inp file

file = open ('CZM.inp', 'r')
lines = file.readlines()



## 2- Generation of the inp file containing the nodes and their coordinates

flagfoundnodes = False 
nodes = open ('node_list.inp', 'w')
for line in lines:
    if(flagfoundnodes):
        if (line[0]=='*'):
            flagfoundnodes = False 
            break
        else:
            line=line.replace(',', '')
            nodes.write (line)
        
    if (line[0:5] == '*Node'):
        flagfoundnodes = True
  
nodes.close()           

## 2- Generation of the inp file containing the elements and their connectivity

flagfoundElem = False 
elements = open ('element_list.inp', 'w')
for line in lines:
    if(flagfoundElem):
        if (line[0]=='*'):
            flagfoundElem = False 
            break
        else:
            line=line.replace(',', '')
            elements.write (line)
    if (line[0:8] == '*Element'):
        flagfoundElem = True
        
elements.close()

# 3- Generation of the element sets of matrix and iclusions 
# Searching for the equations in the imported file

line_number_equ = 1
line_number_NSET= 1
search_to_elsets = "*Elset,"  
search_to_end = "*End Part" 
output = open ('element_sets.inp', 'w')

searched_elsets = []
searched_end = []

for line in lines:
    if search_to_end in line:        
        searched_end.append(line_number_NSET)
    line_number_NSET += 1


for line in lines:
    if search_to_elsets in line:        
        searched_elsets.append(line_number_equ)
    line_number_equ += 1

for i in range (searched_elsets[0]-1, searched_end[0]-1):
    output.write(lines[i])

output.close()
        
##############################################################################
##############################################################################

# Step 2: Generation of dataframes 

# 1- dataframe of nodes and their coordinates 

with open('node_list.inp') as f:
        for k, v in groupby(islice(f, 0, None),key=lambda x:  x.strip()[0:1].isdigit()):
            val = list(v)
            if k:
                dataframe_node=pd.DataFrame(map(str.split,val),
                            columns =['Node', 'x_coord', 'y_coord', 'z_coord'])
                dataframe_node = dataframe_node.astype(float)
                dataframe_node = dataframe_node.to_numpy() 
                
nodal_List = []
for i in range (0, len(dataframe_node)):
    node_coords = [dataframe_node[i,0], dataframe_node[i,1:4]]
    nodal_List.append(node_coords)     
           
# 2- dataframe of elements and their respective connectivity list                   
# element connectivity represents the index number of the nodes that generate.. 
# the surfaces of the elements

with open('element_list.inp') as f:
        for k, v in groupby(islice(f, 0, None),key=lambda x:  x.strip()[0:1].isdigit()):
            val = list(v)
            if k:
                dataframe_element=pd.DataFrame(map(str.split,val))
                dataframe_element = dataframe_element.astype(int)
                dataframe_element = dataframe_element.to_numpy()
                

# elemConnList is a matrix which assign to each element its connectivity list 

elemConnList = []
for j in range (0, len(dataframe_element)):
    conn_list = [dataframe_element[j,0], dataframe_element[j,1:4]]
    elemConnList.append(conn_list)
                    
elemConnList = pd.DataFrame(elemConnList, columns=['Element', 'Element_Connectivity'])

###############################################################################
###############################################################################
# Step 3: Assign to each element the element set on which the element belongs to
# Important: The 'element_sets.inp' file must be started with Matrix element set
# ... otherwise this code would lead to false results
## First part: Inclusions
#
# This taske is carried out seperately for inclusions and matrix
# This part handles both generated and non generated element sets in abaqus 

element_sets = open ('element_sets.inp', 'r')
lines_elemSets = element_sets.readlines()                

line_number_Inclusion = 1
search_Inclusion = "*Elset, elset=Inclusion"
searched_Inclusions = []
inclusion_Index = []
inclusion_Elements = []

# Identifying the corresponding line number of inclusions element sets in the 
# ... 'element_sets.inp' inp file

for items in lines_elemSets:
    if search_Inclusion in items:        
        searched_Inclusions.append(line_number_Inclusion)
    line_number_Inclusion += 1
    
# To prevent error of 'error of list index out of range', the loop is carried
# ... out on all of the inclusions except for the last one             
for i in range (0, len(searched_Inclusions)-1):
    line = lines_elemSets [searched_Inclusions[i]-1]
    keywords = line.split(',')
    Inclusion_keyword = keywords[1]
    Inclusion = Inclusion_keyword.find('Inclusion')
    Inclusion_id = int (Inclusion_keyword[Inclusion+9:])
    inclusion_Index.append(Inclusion_id)
    
    
    # If the element set is generated by the 'generate' technique in ABAQUS 
    if (len(lines_elemSets[searched_Inclusions[i]].split(','))==3):
        
        line_generate = lines_elemSets [searched_Inclusions[i]]
        line_generate = line_generate.split(',')
        start = int(line_generate[0])
        end = int (line_generate[1])+1
        step_size = int (line_generate[2]) 
        elems_generate = list(range(start, end, step_size))
        inclusion_Elements.append(elems_generate)
    
    else:
        # if the element set contains rows of elements with 16 numbers
        elem_Nogen = []
        for j in range (searched_Inclusions[i], searched_Inclusions[i+1]-1):
            elem_Nogen.append(lines_elemSets[j])
        elem_Nogen_float = []
        inclusion_Elements.append(elem_Nogen_float)
        for items in elem_Nogen:
            line_Nogen = items.split(',')
            for items in line_Nogen:
                items = int (items)
                elem_Nogen_float.append(items)

# Appending the last inclusion index and element set to the corresponding lists

line_last = lines_elemSets[searched_Inclusions[len(searched_Inclusions)-1]-1]
keyword = line_last.split(',')
Inclusion_keyword_last = keyword[1]
Inclusion_last = Inclusion_keyword_last.find('Inclusion')
Inclusion_id_last = int (Inclusion_keyword_last[Inclusion+9:])
inclusion_Index.append(Inclusion_id_last)
       

elems_last_Inc = []

if len(lines_elemSets[searched_Inclusions[len(searched_Inclusions)-1]].split(','))==3:
    line_generate_end = lines_elemSets[searched_Inclusions[len(searched_Inclusions)-1]].split(',')
    start_end = int(line_generate_end[0])
    end_end = int (line_generate_end[1])+1
    step_size_end = int (line_generate_end[2]) 
    elems_generate_end = list(range(start_end, end_end, step_size_end))
    inclusion_Elements.append(elems_generate_end)

else:
    last_elementSet = []    
    for j in range (searched_Inclusions[len(searched_Inclusions)-1], len(lines_elemSets)):
        last_elementSet.append(lines_elemSets[j])                               
    last_elementSet_float = []
    inclusion_Elements.append(last_elementSet_float)
    for items_last in last_elementSet:
        line_last = items_last.split(',')
        for items in line_last:
            items = int (items)
            last_elementSet_float.append(items)



# Combing the Inclusion Index and Inclusion element sets

inclusions_elemIndex = list(zip(inclusion_Index, inclusion_Elements))
 
inclusions_elemTotal =[]
for i in range (0, len(inclusions_elemIndex)):
    elem_test = inclusions_elemIndex[i][1]
    incl_test = inclusions_elemIndex[i][0]
    for j in range (0, len(elem_test)):
        elem = elem_test[j]
        test= [elem, incl_test]
        inclusions_elemTotal.append(test)
inclusions_elemTotal = pd.DataFrame(inclusions_elemTotal)

inclusions_nodeTotal = []
for i in range(0, len(inclusions_elemTotal)):
    index = inclusions_elemTotal[0][i]
    nodeList = elemConnList['Element_Connectivity'][index-1].tolist()
    for j in range(0, 3):     
        if nodeList[j] not in inclusions_nodeTotal:
            inclusions_nodeTotal.append(nodeList[j])
inclusions_nodeTotal = pd.DataFrame(inclusions_nodeTotal)       
##############################################################################
# Assign to each element the element set on which the element belongs to
## Second part: Matrix
#
# This taske is carried out seperately for inclusions and matrix
# This part handles both generated and non generated element sets in abaqus

line_number_Matrix = 1
search_Inclusion = "*Elset, elset=Matrix"
searched_Matrix = []
matrix_Index = [0]
# '0' is chosen as the Id number of the elements that constitute the matrix
matrix_Elements = []

# Identifying the corresponding line number of inclusions element sets in the 
# ... 'element_sets.inp' inp file

for items in lines_elemSets:
    if search_Inclusion in items:        
        searched_Matrix.append(line_number_Matrix)
    line_number_Matrix += 1 

if len(lines_elemSets[searched_Matrix[len(searched_Matrix)-1]].split(','))==3:
    line_Matrix = lines_elemSets[searched_Matrix[len(searched_Matrix)-1]].split(',')
    start_Matrix = int(line_Matrix[0])
    end_Matrix = int (line_Matrix[1])+1
    step_size_Matrix = int (line_Matrix[2]) 
    elems_generate_Matrix = list(range(start_Matrix, end_Matrix, step_size_Matrix))
    matrix_Elements.append(elems_generate_Matrix)

else:
    Matrix_elementSet = []    
    for j in range (searched_Matrix[0] , searched_Inclusions[0]-1):
        Matrix_elementSet.append(lines_elemSets[j])                               
    Matrix_elementSet_float = []
    matrix_Elements.append(Matrix_elementSet_float)
    for items_Matrix in Matrix_elementSet:
        items_Matrix = items_Matrix.split(',')
        for items in items_Matrix:
            items = int (items)
            Matrix_elementSet_float.append(items) 

matrix_elemIndex = [matrix_Index, matrix_Elements]

matrix_elemTotal =[]
for i in range (0, len(matrix_Elements[0])):
    elem = matrix_Elements[0][i]
    test= [elem, 0]
    matrix_elemTotal.append(test)
    
matrix_elemTotal = pd.DataFrame(matrix_elemTotal)

matrix_nodeTotal = []
for i in range(0, len(matrix_elemTotal)):
    index = matrix_elemTotal[0][i]
    nodeList = elemConnList['Element_Connectivity'][index-1].tolist()
    for j in range(0, 3):     
        if nodeList[j] not in matrix_nodeTotal:
            matrix_nodeTotal.append(nodeList[j])
matrix_nodeTotal = pd.DataFrame(matrix_nodeTotal)           
##############################################################################
##############################################################################

# Generating the 'Node Connectivity' list 

# Step 4-1: Assign to each node the element index on which the node belongs to

elements_total = len(elemConnList) 
node_elemIndex = []
for i in range (0, elements_total):
    element_index = elemConnList['Element'][i]
    node = elemConnList['Element_Connectivity'][i]
    for nodes in node:
        node_elem = [nodes, element_index]
        node_elemIndex.append(node_elem)
        
node_elemIndex = pd.DataFrame(node_elemIndex, columns=['Node', 'Element'])

# Step 4-2: Generating the node connectivity list - Node connectivity list 
# ... represents the elements' id that surround a specific node
node_connectivity = []
for i in range(0, len(nodal_List)):
    node = nodal_List[i][0]
    b=node_elemIndex.loc[node_elemIndex.Node==node].values
    b=pd.DataFrame(b)
    b_list = b[1].tolist()
    b_list = [int(x) for x in b_list]
    node_connectivity.append(b_list)

##############################################################################
##############################################################################
# Step 5: Search through all of the nodes and find the ones on the boundary of 
# the inclusions --> ITZ modelling between inclusions and matrix

# Description: At the first step the interface nodes must be recognized, to 
# ... detect the interface node, from the 'node_connectivity' list the elements 
# ... surrounding the node are recognized, if these elements belongs to the same 
# ... inclusion or matrix that shows the node is not the interface node, otherwise 
# ... the node is located at the interface zone  

    
interface_nodes = []
elem_frame = [inclusions_elemTotal, matrix_elemTotal]
elemTotal = pd.concat(elem_frame)

for i in range (0, len(nodal_List)):

    node_index = nodal_List[i][0]
    node_coords = nodal_List[i][1].tolist()
    node_connect = node_connectivity[i]
    # node_inclusions represents the list of corresponding inclusions of elements
    # ... in the 'node_connectivity' list
    
       # The member '0' in the node_inclusions represents the element that belongs
       # ... to the matrix and other integer number represets the index of the inclusions
    node_inclusions = np.zeros(shape=(len(node_connect))) 
    count = 0
    
        
    for elements in node_connect:
        inclusion_id = pd.DataFrame(elemTotal.loc[elemTotal[0]==elements].values)
        inclusion_id = inclusion_id[1]
        node_inclusions[count] = inclusion_id
        count = count + 1
    # for i in range(0, len(inclusions_elemIndex)):
    #     inclusion_elem = inclusions_elemIndex[i][1]
    #     inclusion_id = inclusions_elemIndex[i][0]
        
    #     for elements in node_connect:
    #         if elements in (inclusion_elem):                
    #             node_inclusions[count] = inclusion_id
    #             count = count +1

 
     	# Implementation of boolean operation to check if all of the members of 
         # ... 'node_inclusions' are equal to each other or not. Different members
         # ... shows the node is located in the interface zone
    
    IsBulkNode = np.all(node_inclusions == node_inclusions[0])
    if (not IsBulkNode):
        input_interface = [] 
        interface_nodes.append(input_interface)
        input_interface.append(node_index)
        input_interface.append(node_coords)
        input_interface.append(node_connect)
        input_interface.append(node_inclusions)             
        # Calculate multiplicity
        multiplicity = len(node_connect)
        input_interface.append(multiplicity)
        # The node at the interface zone would always belongs to the element 
        # ... in the matrix with smallest id number 
        interface_nodes_index = int(min(node_inclusions))
        input_interface.append(interface_nodes_index)

        # With consideration of elements in 'node_connect' list of interfaces nodes, 
        # ... the neighbor element of interface node with the smallest index   
        # ... would capture the index number of interface node and it is stored in 'new_connect'
        node_connect_min = int(min(node_connect))		 
        
        for i in range (0, len(np.unique(node_inclusions))):
            if (np.unique(node_inclusions)[i] == interface_nodes_index):
                new_connect = node_connect_min
                input_interface.append(new_connect)
                
interface_nodes = pd.DataFrame(interface_nodes, columns=['Node', 'Coords', 
   'node_connectivity','node_inclusions' ,'multiplicity', 'interface_nodes_index','new_connect']) 

##############################################################################
##############################################################################
# Step 6-1 : Search through elements and find the shared faces between them in ITZ zone 

ITZ_interfelems = []                # list of interface elements
index_interfelems = []              # keep a list of indices of elements pairs identified to avoid repetition
ITZ_inInclusion = []                # List of ITZ elements inside inclusions
ITZ_inMatrix = []                   # List of ITZ elements inside Matrix 

for i in range(0, len(interface_nodes)):       
      interf_elem_nConnect = interface_nodes['node_connectivity'][i]
      interf_elem_nInclusions = interface_nodes['node_inclusions'][i]
     
                         
      for j in range (0, len(interf_elem_nConnect)):
        interf_elem_x = interf_elem_nConnect [j]
        x_node_inclusion = int(interf_elem_nInclusions[j])
        
        for k in range(0, len(interf_elem_nConnect)):
            interf_elem_y = interf_elem_nConnect [k]
            y_node_inclusion = int(interf_elem_nInclusions[k])
            
            if (x_node_inclusion != y_node_inclusion):
                
                xnodes = elemConnList.loc[elemConnList.Element==interf_elem_nConnect [j]].values
                xnodes = pd.DataFrame(xnodes)
                xnodes = xnodes[1][0].tolist()
                
                ynodes = elemConnList.loc[elemConnList.Element==interf_elem_nConnect [k]].values
                ynodes = pd.DataFrame(ynodes)
                ynodes = ynodes[1][0].tolist()
                           
                count_commonNodes = 0
                for c in range (0, 3):    # My elements have 3 nodes 
                    if xnodes[c] in ynodes:
                        count_commonNodes = count_commonNodes + 1 
                        
                if (count_commonNodes == 2): # this is a common face
                    if ([interf_elem_x , interf_elem_y] not in index_interfelems):
                        if ([interf_elem_y , interf_elem_x] not in index_interfelems):
                            input_interfelems = [] 
                            ITZ_interfelems.append(input_interfelems)
                            
                            input_interfelems.append(interf_elem_x)
                            input_interfelems.append(xnodes)
                            input_interfelems.append(interf_elem_y)
                            input_interfelems.append(x_node_inclusion)
                            
                            if (x_node_inclusion != 0):
                                ITZ_inInclusion.append(interf_elem_x)
                            else:
                                ITZ_inMatrix.append(interf_elem_x)    
                                
                            input_interfelems.append(interf_elem_y)
                            input_interfelems.append(ynodes)
                            input_interfelems.append(interf_elem_x)
                            input_interfelems.append(y_node_inclusion)
                            
                            if (y_node_inclusion != 0):
                                ITZ_inInclusion.append(interf_elem_y)
                            else:
                                ITZ_inMatrix.append(interf_elem_y)    
                            
                            index_interfelems.append([interf_elem_x , interf_elem_y])

ITZ_interfelems = pd.DataFrame(ITZ_interfelems, columns=['elem_ITZ_x', 'Connect_x', 
   'elem_ITZ_xy','inclusion_id_x' ,'elem_ITZ_y', 'Connect_y','elem_ITZ_yx', 'inclusion_id_y']) 

####################################################################################################
# Step 6-2 : Search through elements and find the shared faces between them inside the inclusions 
# Identifying the list of elements inside the inclusions minus the inclusions' 
# elements in the ITZ zone 
 
ITZ_inInclusion = pd.DataFrame(ITZ_inInclusion)
b_incl= inclusions_elemTotal[0]

# List of the elements inside the inclusions 
# elem_inInclusion  = pd.concat([ITZ_inInclusion, b_incl]).drop_duplicates(keep=False)
# elem_inInclusion = elem_inInclusion[0].to_list()

 
interfelem_inInclusion = []
index_interfelem_inInclusion = []
#for i in range(0, len(elem_inInclusion)):
for i in range(0, len(b_incl)):
    elementx = b_incl[i]
    xnodes = elemConnList.loc[elemConnList.Element==elementx].values
    xnodes = pd.DataFrame(xnodes)
    xnodes = xnodes[1][0].tolist()
    x_inclusion_id = pd.DataFrame(elemTotal.loc[elemTotal[0]==elementx].values)
    x_inclusion_id = x_inclusion_id[1][0]
    
#    for j in range(0, len(elem_inInclusion)):    
    for j in range(1, len(b_incl)):
        elementy = b_incl[j]
        ynodes = elemConnList.loc[elemConnList.Element==elementy].values
        ynodes = pd.DataFrame(ynodes)
        ynodes = ynodes[1][0].tolist()
        y_inclusion_id = pd.DataFrame(elemTotal.loc[elemTotal[0]==elementy].values)
        y_inclusion_id = y_inclusion_id[1][0]
        
        count_commonNodes = 0
        for c in range (0, 3):    # My elements have 3 nodes 
            if xnodes[c] in ynodes:
                count_commonNodes = count_commonNodes + 1
                
        if (count_commonNodes == 2): # this is a common face
            if ([elementx , elementy] not in index_interfelem_inInclusion):
                if ([elementy , elementx] not in index_interfelem_inInclusion):
                            input_interfelems = [] 
                            interfelem_inInclusion.append(input_interfelems)
                            index_interfelem_inInclusion.append([elementx , elementy])
                            
                            input_interfelems.append(elementx)
                            input_interfelems.append(xnodes)
                            input_interfelems.append(elementy)
                            input_interfelems.append(x_inclusion_id)
                            
                            input_interfelems.append(elementy)
                            input_interfelems.append(ynodes)
                            input_interfelems.append(elementx)
                            input_interfelems.append(y_inclusion_id)

interfelem_inInclusion = pd.DataFrame(interfelem_inInclusion, columns=['elem_x', 'Connect_x', 
   'elem_xy','inclusion_id_x' ,'elem_y', 'Connect_y','elem_yx', 'inclusion_id_y'])

#############################################################################################
# Step 6-3 : Search through elements and find the shared faces between them inside the matrix 
# Identifying the list of elements in the matrix minus the Matrix's elements in the ITZ zone

ITZ_inMatrix = pd.DataFrame(ITZ_inMatrix)
b_Mat = matrix_elemTotal[0]

# List of the elements inside the inclusions 
# elem_inMatrix  = pd.concat([ITZ_inMatrix, b_Mat]).drop_duplicates(keep=False)
# elem_inMatrix = elem_inMatrix[0].to_list()

interfelem_inMatrix = []
index_interfelem_inMatrix = []
for i in range(0, len(b_Mat)):
    elementx = b_Mat[i]
    xnodes = elemConnList.loc[elemConnList.Element==elementx].values
    xnodes = pd.DataFrame(xnodes)
    xnodes = xnodes[1][0].tolist()
    x_inclusion_id = 0
        
    for j in range(1, len(b_Mat)):
        elementy = b_Mat[j]
        ynodes = elemConnList.loc[elemConnList.Element==elementy].values
        ynodes = pd.DataFrame(ynodes)
        ynodes = ynodes[1][0].tolist()
        y_inclusion_id = 0
        
        count_commonNodes = 0
        for c in range (0, 3):    # My elements have 3 nodes 
            if xnodes[c] in ynodes:
                count_commonNodes = count_commonNodes + 1
                
        if (count_commonNodes == 2): # this is a common face
            if ([elementx , elementy] not in index_interfelem_inMatrix):
                if ([elementy , elementx] not in index_interfelem_inMatrix):
                            input_interfelems = [] 
                            interfelem_inMatrix.append(input_interfelems)
                            index_interfelem_inMatrix.append([elementx , elementy])
                            
                            input_interfelems.append(elementx)
                            input_interfelems.append(xnodes)
                            input_interfelems.append(elementy)
                            input_interfelems.append(x_inclusion_id)
                            
                            input_interfelems.append(elementy)
                            input_interfelems.append(ynodes)
                            input_interfelems.append(elementx)
                            input_interfelems.append(y_inclusion_id)

interfelem_inMatrix = pd.DataFrame(interfelem_inMatrix, columns=['elem_x', 'Connect_x', 
   'elem_xy','inclusion_id_x' ,'elem_y', 'Connect_y','elem_yx', 'inclusion_id_y'])

##############################################################################
##############################################################################                            

# Step 7: Assigning faces to the interface elements
# Tetrahedral 2D elements --> Total three faces
# Each paired elements have 2 nodes in common 

# Step 7-1: Assigning faces to the elements in ITZ zone

elem1_ITZ_corrnodes = []
elem2_ITZ_corrnodes = []  
elem1_ITZ_facetype = []
elem2_ITZ_facetype = []
#for i in range(0, len(ITZ_interfelems)):
for i in range(0, len(ITZ_interfelems)):
    elem1_connect = np.array(ITZ_interfelems['Connect_x'][i])
    elem2_connect = np.array(ITZ_interfelems['Connect_y'][i])
	# check positions of the common interface nodes in the elements
    posintnodes1 = np.zeros(shape=(2))
    posintnodes2 = np.zeros(shape=(2))

    count = 0
    input1 = []
    elem1_ITZ_corrnodes.append(input1)    
    for n in range(0,3):
        if elem1_connect[n] in elem2_connect:
            posintnodes1[count] = n+1                       # index from 1 to 3
            dove = np.where(elem2_connect == elem1_connect[n])
            input1.append([n+1,int(dove[0])+1])             # index from 1 to 3
            count = count + 1                            
                            
    count = 0
    input2 = []
    elem2_ITZ_corrnodes.append(input2)    
    for n in range(0,3):
        if elem2_connect[n] in elem1_connect:
            posintnodes2[count] = n+1                       # index from 1 to 3
            dove = np.where(elem1_connect == elem2_connect[n])
            input2.append([n+1,int(dove[0])+1])             # index from 1 to 3
            count = count + 1
            
            # assign face type to interface element
    if 1 in posintnodes1:
        if 2 in posintnodes1:
            facetype = 1 
            elem1_ITZ_facetype.append(facetype)
        else:
            facetype = 3
            elem1_ITZ_facetype.append(facetype)
    else:
        facetype = 2
        elem1_ITZ_facetype.append(facetype)
             

    if 1 in posintnodes2:
        if 2 in posintnodes2:
            facetype = 1 
            elem2_ITZ_facetype.append(facetype)
        else:
            facetype = 3
            elem2_ITZ_facetype.append(facetype)
    else:
        facetype = 2
        elem2_ITZ_facetype.append(facetype)    

ITZ_interfelems['elem_x_facetype'] = elem1_ITZ_facetype 
ITZ_interfelems['elem_y_facetype'] = elem2_ITZ_facetype                      
                        

# Step 7-2: Assigning faces to the elements inside inclusions                 
    
elem1_inclusion_corrnodes = []
elem2_inclusion_corrnodes = []  
elem1_inclusion_facetype = []
elem2_inclusion_facetype = []
 
for i in range(0, len(interfelem_inInclusion)):
    elem1_connect = np.array(interfelem_inInclusion['Connect_x'][i])
    elem2_connect = np.array(interfelem_inInclusion['Connect_y'][i])
	# check positions of the common interface nodes in the elements
    posintnodes1 = np.zeros(shape=(2))
    posintnodes2 = np.zeros(shape=(2))

    count = 0
    input1 = []
    elem1_inclusion_corrnodes.append(input1)    
    for n in range(0,3):
        if elem1_connect[n] in elem2_connect:
            posintnodes1[count] = n+1                       # index from 1 to 3
            dove = np.where(elem2_connect == elem1_connect[n])
            input1.append([n+1,int(dove[0])+1])             # index from 1 to 3
            count = count + 1                            
                            
    count = 0
    input2 = []
    elem2_inclusion_corrnodes.append(input2)    
    for n in range(0,3):
        if elem2_connect[n] in elem1_connect:
            posintnodes2[count] = n+1                       # index from 1 to 3
            dove = np.where(elem1_connect == elem2_connect[n])
            input2.append([n+1,int(dove[0])+1])             # index from 1 to 3
            count = count + 1
            
            # assign face type to interface element
    if 1 in posintnodes1:
        if 2 in posintnodes1:
            facetype = 1 
            elem1_inclusion_facetype.append(facetype)
        else:
            facetype = 3
            elem1_inclusion_facetype.append(facetype)
    else:
        facetype = 2
        elem1_inclusion_facetype.append(facetype)
             

    if 1 in posintnodes2:
        if 2 in posintnodes2:
            facetype = 1 
            elem2_inclusion_facetype.append(facetype)
        else:
            facetype = 3
            elem2_inclusion_facetype.append(facetype)
    else:
        facetype = 2
        elem2_inclusion_facetype.append(facetype)    

interfelem_inInclusion['elem_x_facetype'] = elem1_inclusion_facetype 
interfelem_inInclusion['elem_y_facetype'] = elem2_inclusion_facetype 

# Step 7-3: Assigning faces to the elements inside matrix 

elem1_matrix_corrnodes = []
elem2_matrix_corrnodes = []  
elem1_matrix_facetype = []
elem2_matrix_facetype = []
 
for i in range(0, len(interfelem_inMatrix)):
    elem1_connect = np.array(interfelem_inMatrix['Connect_x'][i])
    elem2_connect = np.array(interfelem_inMatrix['Connect_y'][i])
	# check positions of the common interface nodes in the elements
    posintnodes1 = np.zeros(shape=(2))
    posintnodes2 = np.zeros(shape=(2))

    count = 0
    input1 = []
    elem1_matrix_corrnodes.append(input1)    
    for n in range(0,3):
        if elem1_connect[n] in elem2_connect:
            posintnodes1[count] = n+1                       # index from 1 to 3
            dove = np.where(elem2_connect == elem1_connect[n])
            input1.append([n+1,int(dove[0])+1])             # index from 1 to 3
            count = count + 1                            
                            
    count = 0
    input2 = []
    elem2_matrix_corrnodes.append(input2)    
    for n in range(0,3):
        if elem2_connect[n] in elem1_connect:
            posintnodes2[count] = n+1                       # index from 1 to 3
            dove = np.where(elem1_connect == elem2_connect[n])
            input2.append([n+1,int(dove[0])+1])             # index from 1 to 3
            count = count + 1
            
            # assign face type to interface element
    if 1 in posintnodes1:
        if 2 in posintnodes1:
            facetype = 1 
            elem1_matrix_facetype.append(facetype)
        else:
            facetype = 3
            elem1_matrix_facetype.append(facetype)
    else:
        facetype = 2
        elem1_matrix_facetype.append(facetype)
             

    if 1 in posintnodes2:
        if 2 in posintnodes2:
            facetype = 1 
            elem2_matrix_facetype.append(facetype)
        else:
            facetype = 3
            elem2_matrix_facetype.append(facetype)
    else:
        facetype = 2
        elem2_matrix_facetype.append(facetype)    

interfelem_inMatrix['elem_x_facetype'] = elem1_matrix_facetype 
interfelem_inMatrix['elem_y_facetype'] = elem2_matrix_facetype 

OrderCohNodes = []

# if (facetype == 1):
OrderCohNodes.append([1, np.array([1, 2])])
    
# if (facetype == 2):
OrderCohNodes.append([2, np.array([2, 3])])
    
# if (facetype == 3):
OrderCohNodes.append([3, np.array([3, 1])])

OrderCohNodes = pd.DataFrame(OrderCohNodes, columns = ['facetype','NodeOrder'])
 
##############################################################################
##############################################################################

# Step 8: Generation of cohesive nodes 
# Step 8-1: Generation of Cohesive nodes in the ITZ zone 

czm_ITZ = []
total_Nodes = len(dataframe_node)
count = total_Nodes
for i in range(0, len(interface_nodes)):    
    originalNodeIndex = interface_nodes['Node'][i]
    NodeIndex = originalNodeIndex
    coordinate = interface_nodes['Coords'][i]
    originalElement = interface_nodes['new_connect'][i]
    multip = interface_nodes['multiplicity'][i]
    node_connect = interface_nodes['node_connectivity'][i]
    index1 = np.where(np.array(node_connect) == originalElement)
    material_id = int(interface_nodes['node_inclusions'][i][index1])
    
    element_Connt = elemConnList['Element_Connectivity'][originalElement-1]
    index2 = np.where(element_Connt == originalNodeIndex)    
    element_newConnt = np.zeros(3)
    element_newConnt[index2]= originalNodeIndex
    element_newConnt = [int(x) for x in element_newConnt]
  
    a = [[NodeIndex, originalNodeIndex , originalElement ,element_Connt, element_newConnt, 
                              node_connect , material_id ,  coordinate, multip]]
   
   
    for j in range(0, multip):
        element = node_connect[j]
        if (element == originalElement):
            continue 
        else:
            node_ITZ_original = originalNodeIndex
            node_ITZ_index = count + 1 
            node_ITZ_coords = interface_nodes['Coords'][i]
            node_ITZ_elemConnect = elemConnList['Element_Connectivity'][element-1]
            node_ITZ_connect = interface_nodes['node_connectivity'][i]
            node_ITZ_multiplicity = multip
            node_ITZ_material = interface_nodes['node_inclusions'][i][j]
        
            index2 = np.where(node_ITZ_elemConnect == originalNodeIndex)    
            node_ITZ_element_newConnt = np.zeros(3)
            node_ITZ_element_newConnt[index2]= node_ITZ_index
            node_ITZ_element_newConnt = [int(x) for x in node_ITZ_element_newConnt]
            
            test = [node_ITZ_index ,node_ITZ_original ,element ,node_ITZ_elemConnect 
            ,node_ITZ_element_newConnt, node_ITZ_connect, node_ITZ_material, 
            node_ITZ_coords, node_ITZ_multiplicity]
            
            a.append(test)
        
        count = count + 1 

    df1 = pd.DataFrame(a, columns = ['genNode', 'OriginalNode', 'elem',
         'elemConn','elemNewConn','nodeConn','material_id','coords','multip'])
    czm_ITZ.append(df1)

czm_ITZ = pd.concat(czm_ITZ, ignore_index=True)
    
# Step 8-2: Generation of Cohesive nodes inside the inclusions

# List of the nodes inside the inclusions 
node_inInclusion  = pd.concat([interface_nodes['Node'], inclusions_nodeTotal]).drop_duplicates(keep=False)
node_inInclusion = node_inInclusion[0].to_list()

czm_inInclusion = []
for i in range(0, len(node_inInclusion)):    
    originalNodeIndex = int(node_inInclusion[i])
    NodeIndex = originalNodeIndex
    node_connect = node_connectivity[originalNodeIndex-1]
    originalElement = int(min(node_connect))
    multip = len(node_connect)
    coordinate = nodal_List[originalNodeIndex-1][1].tolist()
    material_id = pd.DataFrame(elemTotal.loc[elemTotal[0]==originalElement].values)
    material_id = int(material_id[1])
    
    
    element_Connt = elemConnList['Element_Connectivity'][originalElement-1]
    index2 = np.where(element_Connt == originalNodeIndex)    
    element_newConnt = np.zeros(3)
    element_newConnt[index2]= originalNodeIndex    
    element_newConnt = [int(x) for x in element_newConnt]
    a = [[NodeIndex, originalNodeIndex , originalElement ,element_Connt, element_newConnt, 
                              node_connect , material_id ,  coordinate, multip]]
   
    for j in range(0, multip):
        element = node_connect [j]
        if (element == originalElement):
            continue 
        else:
            
            czm_inInclusionOriginalNode = originalNodeIndex
            czm_inInclusionIndex = count + 1 
            czm_inInclusionCoords = nodal_List[NodeIndex-1][1].tolist()
            czm_inInclusionNodeConnect = node_connectivity[NodeIndex-1]
            czm_inInclusionMultiplicity = multip
            czm_inInclusionMaterial = pd.DataFrame(elemTotal.loc[elemTotal[0]==element].values)
            czm_inInclusionMaterial = int(czm_inInclusionMaterial[1])
            czm_inInclusion_elemConnect = elemConnList['Element_Connectivity'][element-1]
            
            index2 = np.where(czm_inInclusion_elemConnect == originalNodeIndex)    
            czm_inInclusion_newConnt = np.zeros(3)
            czm_inInclusion_newConnt[index2]= czm_inInclusionIndex 
            czm_inInclusion_newConnt = [int(x) for x in czm_inInclusion_newConnt]
            
            
            test = [czm_inInclusionIndex ,czm_inInclusionOriginalNode ,element ,czm_inInclusion_elemConnect 
            ,czm_inInclusion_newConnt, czm_inInclusionNodeConnect, czm_inInclusionMaterial, 
            czm_inInclusionCoords, czm_inInclusionMultiplicity]
            
            a.append(test)
        count = count + 1 

    df2 = pd.DataFrame(a,  columns = ['genNode', 'OriginalNode', 'elem',
         'elemConn','elemNewConn','nodeConn','material_id','coords','multip'])
    czm_inInclusion.append(df2)
    
czm_inInclusion = pd.concat(czm_inInclusion, ignore_index=True)

# Step 8-3: Generation of Cohesive nodes inside the Matrix

# List of the nodes inside the matrix 
node_inMatrix  = pd.concat([interface_nodes['Node'], matrix_nodeTotal]).drop_duplicates(keep=False)
node_inMatrix = node_inMatrix[0].to_list()

czm_inMatrix = []
for i in range(0, len(node_inMatrix)):    
    originalNodeIndex = int(node_inMatrix[i])
    NodeIndex = originalNodeIndex
    node_connect = node_connectivity[originalNodeIndex-1]
    originalElement = int(min(node_connect))
    multip = len(node_connect)
    coordinate = nodal_List[originalNodeIndex-1][1].tolist()
    material_id = pd.DataFrame(elemTotal.loc[elemTotal[0]==originalElement].values)
    material_id = int(material_id[1])
       
    element_Connt = elemConnList['Element_Connectivity'][originalElement-1]
    index2 = np.where(element_Connt == originalNodeIndex)    
    element_newConnt = np.zeros(3)
    element_newConnt[index2]= originalNodeIndex    
    element_newConnt = [int(x) for x in element_newConnt]
    a = [[NodeIndex, originalNodeIndex , originalElement ,element_Connt, element_newConnt, 
                              node_connect , material_id ,  coordinate, multip]]
   
    for j in range(0, multip):
        element = node_connect [j]
        if (element == originalElement):
            continue 
        else:
            
            czm_inMatrixOriginalNode = originalNodeIndex
            czm_inMatrixIndex = count + 1 
            czm_inMatrixCoords = nodal_List[NodeIndex-1][1].tolist()
            czm_inMatrixNodeConnect = node_connectivity[NodeIndex-1]
            czm_inMatrixMultiplicity = multip
            czm_inMatrixMaterial = pd.DataFrame(elemTotal.loc[elemTotal[0]==element].values)
            czm_inMatrixMaterial = int(czm_inMatrixMaterial[1])
            czm_inMatrix_elemConnect = elemConnList['Element_Connectivity'][element-1]
            
            index2 = np.where(czm_inMatrix_elemConnect == originalNodeIndex)    
            czm_inMatrix_newConnt = np.zeros(3)
            czm_inMatrix_newConnt[index2]= czm_inMatrixIndex 
            czm_inMatrix_newConnt = [int(x) for x in czm_inMatrix_newConnt]
            
            
            test = [czm_inMatrixIndex ,czm_inMatrixOriginalNode ,element ,czm_inMatrix_elemConnect 
            ,czm_inMatrix_newConnt, czm_inMatrixNodeConnect, czm_inMatrixMaterial, 
            czm_inMatrixCoords, czm_inMatrixMultiplicity]
            
            a.append(test)
        count = count + 1 

    df3 = pd.DataFrame(a,  columns = ['genNode', 'OriginalNode', 'elem',
         'elemConn','elemNewConn','nodeConn','material_id','coords','multip'])
    czm_inMatrix.append(df3)

czm_inMatrix = pd.concat(czm_inMatrix, ignore_index=True)

elem_newConnList = []
for i in range(0, len(dataframe_element)):
    element = dataframe_element[i][0]
    list1=[]
    
    index1 = czm_ITZ.loc[czm_ITZ.elem==element].index
    for j in range(0, len(index1)):
        elem_newConn = czm_ITZ.loc[czm_ITZ.elem==element].values[j][4]
        list1.append(elem_newConn)

    index2 = czm_inInclusion.loc[czm_inInclusion.elem==element].index
    for j in range(0, len(index2)):
        elem_newConn = czm_inInclusion.loc[czm_inInclusion.elem == element].values[j][4]
        list1.append(elem_newConn)

    index3 = czm_inMatrix.loc[czm_inMatrix.elem==element].index
    for j in range(0, len(index3)):
        elem_newConn = czm_inMatrix.loc[czm_inMatrix.elem==element].values[j][4]
        list1.append(elem_newConn)

    list1 = [sum(x) for x in zip(*list1)]
    list2 = [[element, list1]]
    elem_newConnList.append(pd.DataFrame(list2))
    
elem_newConnList = pd.concat(elem_newConnList, ignore_index=True)

##############################################################################
##############################################################################

# Step 9: Updating the element connectivity list in 'ITZ_interfelems',
# 'interfelem_inMatrix' and 'interfelem_inInclusion'

# Step 9-1: Element connectivity updating in ITZ_interfelems

list1 = []
list2 = []
for i in range(0, len(ITZ_interfelems)):
    element = ITZ_interfelems['elem_ITZ_x'][i]
    facetype_x = ITZ_interfelems['elem_x_facetype'][i]
    nodeOrder = OrderCohNodes['NodeOrder'][facetype_x-1]
    index1 = np.where(elem_newConnList[0] == element)
    connList =  elem_newConnList[1][index1[0][0]]
    list1.append([connList, nodeOrder])

for i in range(0, len(ITZ_interfelems)):
    element = ITZ_interfelems['elem_ITZ_y'][i]
    facetype_y = ITZ_interfelems['elem_y_facetype'][i]
    nodeOrder = OrderCohNodes['NodeOrder'][facetype_y-1]
    index2 = np.where(elem_newConnList[0] == element)
    connList =  elem_newConnList[1][index2[0][0]]
    list2.append([connList, nodeOrder])   

list1 = pd.DataFrame(list1)  
list2 = pd.DataFrame(list2)   
ITZ_interfelems.insert(loc=2, column='newConnect_x' , value = list1[0].tolist())
ITZ_interfelems.insert(loc=8, column='newConnect_y' , value = list2[0].tolist())
ITZ_interfelems.insert(loc=11, column='NodeOrder_x' , value = list1[1].tolist())
ITZ_interfelems.insert(loc=13, column='NodeOrder_y' , value = list2[1].tolist())

# Step 9-2: Element connectivity updating in inclusions
list3 = []
list4 = []
for i in range(0, len(interfelem_inInclusion)):
    element = interfelem_inInclusion['elem_x'][i]
    facetype_x = interfelem_inInclusion['elem_x_facetype'][i]
    nodeOrder = OrderCohNodes['NodeOrder'][facetype_x-1]
    index1 = np.where(elem_newConnList[0] == element)
    connList =  elem_newConnList[1][index1[0][0]]
    list3.append([connList, nodeOrder])
for i in range(0, len(interfelem_inInclusion)):
    element = interfelem_inInclusion['elem_y'][i]
    facetype_y = interfelem_inInclusion['elem_y_facetype'][i]
    nodeOrder = OrderCohNodes['NodeOrder'][facetype_y-1]
    index2 = np.where(elem_newConnList[0] == element)
    connList =  elem_newConnList[1][index2[0][0]]
    list4.append([connList, nodeOrder])   
    
list3 = pd.DataFrame(list3)  
list4 = pd.DataFrame(list4)   
interfelem_inInclusion.insert(loc=2, column='newConnect_x' , value = list3[0].tolist())
interfelem_inInclusion.insert(loc=8, column='newConnect_y' , value = list4[0].tolist())
interfelem_inInclusion.insert(loc=11, column='NodeOrder_x' , value = list3[1].tolist())
interfelem_inInclusion.insert(loc=13, column='NodeOrder_y' , value = list4[1].tolist())   

# Step 9-3: Element connectivity updating in matrix
list5 = []
list6 = []
for i in range(0, len(interfelem_inMatrix)):
    element = interfelem_inMatrix['elem_x'][i]
    facetype_x = interfelem_inMatrix['elem_x_facetype'][i]
    nodeOrder = OrderCohNodes['NodeOrder'][facetype_x-1]
    index1 = np.where(elem_newConnList[0] == element)
    connList =  elem_newConnList[1][index1[0][0]]
    list5.append([connList, nodeOrder])
for i in range(0, len(interfelem_inMatrix)):
    element = interfelem_inMatrix['elem_y'][i]
    facetype_y = interfelem_inMatrix['elem_y_facetype'][i]
    nodeOrder = OrderCohNodes['NodeOrder'][facetype_y-1]
    index2 = np.where(elem_newConnList[0] == element)
    connList =  elem_newConnList[1][index2[0][0]]
    list6.append([connList, nodeOrder])   
    
list5 = pd.DataFrame(list5)  
list6 = pd.DataFrame(list6)   
interfelem_inMatrix.insert(loc=2, column='newConnect_x' , value = list5[0].tolist())
interfelem_inMatrix.insert(loc=8, column='newConnect_y' , value = list6[0].tolist())
interfelem_inMatrix.insert(loc=11, column='NodeOrder_x' , value = list5[1].tolist())
interfelem_inMatrix.insert(loc=13, column='NodeOrder_y' , value = list6[1].tolist())    

##############################################################################
##############################################################################
# Step 10: Generation of 2D cohesive elements
# Step 10-1: cohesive elements in ITZ zone 
cohelems_ITZ = []
count = len(elemTotal)
for i in range(0, len(ITZ_interfelems)):
    elem1 = ITZ_interfelems['elem_ITZ_x'][i]
    elem2 = ITZ_interfelems['elem_ITZ_y'][i]
    elem1_nOrdre = ITZ_interfelems['NodeOrder_x'][i]  
    elem1_corr = elem1_ITZ_corrnodes[i]
    cohnodes = np.zeros(shape=(4))
    count = count + 1 
    for j in range(0,2):
        n1 = ITZ_interfelems['newConnect_x'][i][elem1_nOrdre[1]-1]
        n2 = ITZ_interfelems['newConnect_x'][i][elem1_nOrdre[0]-1]
        cohnodes[0] = n1
        cohnodes[1] = n2
        for n in range(0,2):
            if elem1_corr[n][0] == elem1_nOrdre[j]:
                index2 = elem1_corr[n][1]
                cohnodes[j+2] = int(ITZ_interfelems['newConnect_y'][i][index2-1])
    cohelems_ITZ.append([count, int(cohnodes[0]), int(cohnodes[1]), int(cohnodes[2]), int(cohnodes[3])])
cohelems_ITZ = pd.DataFrame(cohelems_ITZ)
   
    
# Step 10-2: cohesive elements inside inclusions   
cohelems_inclusions= []
for i in range(0, len(interfelem_inInclusion)):
    elem1 = interfelem_inInclusion['elem_x'][i]
    elem2 = interfelem_inInclusion['elem_y'][i]
    elem1_nOrdre = interfelem_inInclusion['NodeOrder_x'][i]  
    elem1_corr = elem1_inclusion_corrnodes[i]
    cohnodes = np.zeros(shape=(4))
    count = count + 1 
    for j in range(0,2):
        n1 = interfelem_inInclusion['newConnect_x'][i][elem1_nOrdre[1]-1]
        n2 = interfelem_inInclusion['newConnect_x'][i][elem1_nOrdre[0]-1]
        cohnodes[0] = n1
        cohnodes[1] = n2
        for n in range(0,2):
            if elem1_corr[n][0] == elem1_nOrdre[j]:
                index2 = elem1_corr[n][1]
                cohnodes[j+2] = int(interfelem_inInclusion['newConnect_y'][i][index2-1])
    cohelems_inclusions.append([count, int(cohnodes[0]), int(cohnodes[1]), int(cohnodes[2]), int(cohnodes[3])])
cohelems_inclusions = pd.DataFrame(cohelems_inclusions)

# Step 10-3: cohesive elements inside matrix   
cohelems_matrix= []
for i in range(0, len(interfelem_inMatrix)):
    elem1 = interfelem_inMatrix['elem_x'][i]
    elem2 = interfelem_inMatrix['elem_y'][i]
    elem1_nOrdre = interfelem_inMatrix['NodeOrder_x'][i]  
    elem1_corr = elem1_matrix_corrnodes[i]
    cohnodes = np.zeros(shape=(4))
    count = count + 1 
    for j in range(0,2):
        n1 = interfelem_inMatrix['newConnect_x'][i][elem1_nOrdre[1]-1]
        n2 = interfelem_inMatrix['newConnect_x'][i][elem1_nOrdre[0]-1]
        cohnodes[0] = n1
        cohnodes[1] = n2
        for n in range(0,2):
            if elem1_corr[n][0] == elem1_nOrdre[j]:
                index2 = elem1_corr[n][1]
                cohnodes[j+2] = int(interfelem_inMatrix['newConnect_y'][i][index2-1])
    cohelems_matrix.append([count, int(cohnodes[0]), int(cohnodes[1]), int(cohnodes[2]), int(cohnodes[3])])
cohelems_matrix = pd.DataFrame(cohelems_matrix)

##############################################################################
##############################################################################
# Step 11: Exporting the new data 

# Step 11-1: Writting the elements with updated connectivity list
newElem = np.zeros(shape=(len(elem_newConnList),4))
for i in range(0, len(elem_newConnList)):
    elem = elem_newConnList[0][i]
    node = elem_newConnList[1][i]     
    newElem[i,0] = elem
    newElem[i,1:4] = np.int_(node)  
np.savetxt('new_bulkElems.inp', newElem, fmt='%d,%d,%d,%d')

newElem_inInclusions = np.zeros(shape=(2*len(interfelem_inInclusion),4))
Nelems = len(interfelem_inInclusion)
for i in range(0, Nelems):
    elem1 = interfelem_inInclusion['elem_x'][i]
    elem2 = interfelem_inInclusion['elem_y'][i]
    node1 = interfelem_inInclusion['newConnect_x'][i]
    node2 = interfelem_inInclusion['newConnect_y'][i]
    newElem_inInclusions[2*i,0] = elem1
    newElem_inInclusions[2*i,1:4] = np.int_(node1)  
    newElem_inInclusions[2*i+1,0] = elem2
    newElem_inInclusions[2*i+1,1:4] = np.int_(node2)
newElem_inInclusions = pd.DataFrame(newElem_inInclusions).drop_duplicates()
np.savetxt('inInclusions_bulkElems.inp', newElem_inInclusions, fmt='%d,%d,%d,%d')

newElem_inMatrix = np.zeros(shape=(2*len(interfelem_inMatrix),4))
Nelems = len(interfelem_inMatrix)
for i in range(0, Nelems):
    elem1 = interfelem_inMatrix['elem_x'][i]  
    elem2 = interfelem_inMatrix['elem_y'][i]
    test.append(elem2)
    node1 = interfelem_inMatrix['newConnect_x'][i]
    node2 = interfelem_inMatrix['newConnect_y'][i]
    newElem_inMatrix[2*i,0] = elem1
    newElem_inMatrix[2*i,1:4] = np.int_(node1)  
    newElem_inMatrix[2*i+1,0] = elem2
    newElem_inMatrix[2*i+1,1:4] = np.int_(node2)

newElem_inMatrix = pd.DataFrame(newElem_inMatrix).drop_duplicates()
np.savetxt('inMatrix_bulkElems.inp', newElem_inMatrix, fmt='%d,%d,%d,%d')

# Step 11-2: Writting the nodes  
newNode_ITZ = np.zeros(shape=(len(czm_ITZ),4))
for i in range(0, len(czm_ITZ)):
    node = czm_ITZ['genNode'][i]
    coord = czm_ITZ['coords'][i]
    newNode_ITZ[i,0] = node
    newNode_ITZ[i,1:4] = coord

newNode_inclusion = np.zeros(shape=(len(czm_inInclusion),4))
for i in range(0, len(czm_inInclusion)):
    node = czm_inInclusion['genNode'][i]
    coord = czm_inInclusion['coords'][i]
    newNode_inclusion[i,0] = node
    newNode_inclusion[i,1:4] = coord

newNode_matrix = np.zeros(shape=(len(czm_inMatrix),4))
for i in range(0, len(czm_inMatrix)):
    node = czm_inMatrix['genNode'][i]
    coord = czm_inMatrix['coords'][i]
    newNode_matrix[i,0] = node
    newNode_matrix[i,1:4] = coord

newNode = np.concatenate((newNode_ITZ, newNode_inclusion, newNode_matrix), axis=0)
np.savetxt('new_node.inp', newNode, fmt='%d,%10.5f,%10.5f,%10.5f')


# Step 11-3: Writting the cohesive elements
cohesive_elems = pd.concat([cohelems_ITZ, cohelems_inclusions, cohelems_matrix], ignore_index=True)
np.savetxt('cohesive_elements.inp', cohesive_elems, fmt='%d,%d,%d,%d,%d')

 

    

 



















































































































































































