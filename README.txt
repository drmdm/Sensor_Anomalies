Files and Folders:

File/Folder                            	Details            
---------------------------------- 	-------------------
./data				    	Folder to store journey .csv files 
crash_detector.py		    	Module to use for prediction
DataExploration.html			HTML example of Jupyter notebook		
DataExploration.ipynb			Jupyter Notebook containing data exploration
exploration.py				Module for data exploration notebook and other ideas
Sensor_Challenge.pdf		        Report
predict.py				Main prediction script
README.txt				Details of .zip contents
results.csv				CSV file containing prediction results

To Run the Program:
To run this program on your own data edit the predict.py file
 -store your journey .csv file(s) in the ./data folder in the same location as predict.py
 -add the filepath to your data in 'datadir' in predict.py

Run the script in your IDE or on command line.

on command line (navigate to the folder contianing the data and predict.py):
 python predict.py

DataExploration.ipynb can be opened as a Jupyter notebook or I have provided a HTML file which
can be opened on any machine. 


Packages and version I used:

Package                            Version            
---------------------------------- -------------------           
jupyter                            1.0.0                          
Keras                              2.2.4              
Keras-Applications                 1.0.8              
Keras-Preprocessing                1.1.0                         
matplotlib                         3.1.3                           
mplleaflet                         0.0.5                     
numpy                              1.18.1                      
pandas                             0.24.1                              
scikit-learn                       0.22.1             
scipy                              1.4.1              
seaborn                            0.10.0                       
sklearn                            0.0                     
tensorboard                        1.15.0             
tensorboardX                       2.0                
tensorflow                         1.15.0             
tensorflow-estimator               1.15.1
