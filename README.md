# Build_a_THON
OUR SOLUTION FOR BUILD A THON 

### Dataset
For dataset visit https://bhartischool.iitd.ac.in/build_a_thon/index.html and download the trainset for problem statement 1 and 2.
### Use of the Files

- task1.py, this script contains the code that reads the test set  and predicts the specified output for Task Problem Statement I: Slip Detection and Force Estimation in CSV format (one per line). The format of test set will be same as pre-evaluation dataset (i.e., the path will be the root folder containing all CSV files in the testset)

- task2.py, this script contains the code that reads the test set and predicts the specified output for Task Problem Statement II: Object Detection in CSV format (one per line). A sample output is here. The format of test set will be same as pre-evaluation dataset (i.e., the path will be the complete path to ONE AND ONLY CSV file in the testset)

-requirements.txt, this contains the requirements to load the dependent modules for your scripts.

 - task1_training.py, this script contains the code to read a train set and train a model with the data that must be saved/dumped as ./model_checkpoint_task1
 
 - task2_training.py, this script contains the code to read a train set and train a model with the data that must be saved/dumped as ./model_checkpoint_task2
 
 - model_checkpoint1
 - model_checkpoint2

### How to run the files 

python task1.py --model_path ./model_checkpoint1 --input_data /path/to/testset/directory --output ./path/to/predictions/directory
--input_data argument contains the path to the directory of the hidden test set. 
--input_data expects the path of a root directory where multiple CSV files exist belonging to the test-set. Your script must parse this argument and be able to read all CSV files from this directory path. The format of the input is CSV (exactly the same as the pre-evaluation dataset).
--output argument defines the path to the directory where the predictions are to be dumped. Your scripts must parse this argument and be able to dump/save the output predictions (in CSV format) in the directory (specified by --output argument), corresponding to each input CSV file (with the same filename as that of the input CSV file). Individual output CSV files will contain TWO LABELS (separated with a comma) per line corresponding to each row in the input CSV. The first line of all the output CSV files must exactly be as "Slip, Crumple" referring to the two labels.
--model_path argument contains the path to the model checkpoint/dump that you have previously trained for this problem.



python task1_training.py --model_path ./model_checkpoint1 --input_data ./path/to/trainset/directory
--input_data argument contains the path to the directory of the training set. 
--input_data expects the path of a root directory containing multiple CSV files belonging to the training set. Your script must parse this argument and be able to read CSV files from this path. The format of the input is (exactly the same as training dataset)
--model_path argument contains the path where the model checkpoint will be dumped.



python task2.py --model_path ./model_checkpoint2 --input_data ./path/to/testset.csv --output ./predictions_task2.csv
--input_data argument contains the end path to the hidden test set. 
--input_data expects the argument to be the complete path to the single CSV file representing the testset. Your script must parse this argument and be able to read input from this path. The format of the input is CSV (exactly the same as pre-evaluation dataset for this problem)
--output argument defines the path where the predictions are to be dumped in CSV format. Your scripts must parse this argument and be able to save the output prediction in CSV format - one prediction per line. The output CSV file will contain only ONE LABEL per line corresponding to each row in the input CSV. The first line of the output CSV file must exactly be as "Object_Held", referring to the label of the object.
--model_path argument contains the path to the model checkpoint/dump that you have previously trained for this problem.


python task2_training.py --model_path ./model_checkpoint2 --input_data ./path/to/trainset.csv
--input_data argument contains the path to the training set. Your script must parse this argument and be able to read input from this path. The format of the input is csv (exactly the same as training dataset)
--model_path argument contains the path where the model checkpoint will be dumped.

