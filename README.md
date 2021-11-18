# Cohesive-interface-element
- Python code to generate zero thickness cohesive interface elements (COH2D4) among S3 shell elements in Abaqus

- This code generates cohesive elements in three different zones separately, which  are cohesive elements inside the inclusions, inside the matrix and in the ITZ (interfacial transition zone) zone

-  It is possible to simulate the fracture and crack propagation in these zones w.r.t the defined cohesive properties

- This code is most suited for modelling the fracture and crack propagation of brittle materials like concretes

- In the code, I wrote many comments to make it more understandable

- It should be point out that from a computational point of view, this code is not appropriate. The main purpose of this code is to give you a learning platform and   ideas to develop your own codes. I guess you would understand it easily. Later on I would modify this code with OOP programming to make it faster and more efficient

- In the near future I would add another python code which generates cohesive elements in 3D case in Abaqus

- In the Abaqus folder you can find two inp files, the raw one is without cohesive elements and the other is with cohesive elements
