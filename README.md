# AXE: A Task Decomposition Approach to Learned LSM Tuning

AXE is a novel learned LSM tuning paradigm that decomposes the tuning task into two steps:
1. AXE trains a learned cost model using existing performance modeling or execution logs, acting as a surrogate cost
function in the tuning process
2. AXE efficiently generates arbitrarily many training samples for a learned tuner that is optimized to identify high performance tunings using the learned cost model as its loss function.


## Instructions to Run AXE
These are the general steps that can be used to generate data, train each model and run the model baseline using various LSM compaction policies.

1. **Install Python (3.7 or later)**  
   Download Python from the official repository:  \
   [https://www.python.org/downloads/](https://www.python.org/downloads/)  \
   Follow the installation instructions for your operating system (Windows, Mac, Linux).

2. **Verify Python installation**  
   Run the following command to ensure Python is installed correctly:
   ```bash
   python --version
   ```
   This command should output something of the form 3.x.x

3. **Add Python to the PATH** \
   a) Check the option for "Add Python to PATH" if installing freshly. (This is the easiest option) \
   b) If Python is already installed but not added to "PATH", these are some helpful links to do this: \
       i)  https://phoenixnap.com/kb/add-python-to-path  \
       ii) https://realpython.com/add-python-to-path/  
   
4. **Install pip** \
   https://pip.pypa.io/en/stable/installation/
   
5. **Verify pip installation** 
   ```bash
   pip --version
   ```
   
6. **(Optional) Create a virtual environment and activate it** \
   To prevent any clash in requirements, we recommend using a virtual environment (Feel free to use whatever is convenient. We provide instructions for the default python virtual environment):
   [https://docs.python.org/3/library/venv.html](https://docs.python.org/3/library/venv.html) 
      
7. **Install requirements using the requirements.txt file** \
   Please install all the requirements provided in the requirements.txt file in the root of the project using
   ```bash
   pip install -r requirements.txt
   ```
8. **Configure the endure.toml file (More details can be found below)** \
   The [endure.toml](#Configuration-File) file contains all required options for the experiments and jobs. Please configure all the jobs and their respective parameters  required to successfully run the project.

9. **Run the project** \
    The project is structured such that all experiments can be run by just running the endure.py file. Use the following command to run it:
   ```bash
   python endure.py
   ```
## Configuration File
All of our configuration is handled by the endure.toml file placed in the root directory of our project. 
The config file is divided into sections denoted by a heading within the square brackets (for example *[app]*)  
1. To run a specific job, change the variable **run** under the *[app]* header. All the available jobs are already present in the variable and provided as commented out code. Uncomment all jobs that have to be run to run multiple jobs or a single job.   

2. To change log levels, use the *[log]* section of the toml file that controls the filenames generated, the name, the log level (warning, debug, error), date formats and whether [tqdm](https://tqdm.github.io/) is enabled.  

3. The *[I/O]* section determines the directory where data will be stored.  

4. To control the design structure of the Log Structured Merge (LSM) Tree, please use the *[lsm]* section. The **design** parameter controls which compaction policy the LSM Tree will use. We currently support Classic (either Leveling or Tiering), YZHybrid (Dostoevsky model), KHybrid (the most flexible model with each level supporting variable number of runs), QFixed (all the *K* values for the KLSM model is same and equal to *Q*), Tiering, Leveling. The *[lsm.bounds]* determine the maximum resources available to the LSM Tree. The *[lsm.system]* section contains a default set of System parameters (description is provided in file for each parameter) in case this is not randomly selected from the available resources. Similarly, the *[lsm.workload]* contains the default values for the workload composition (percentage of empty reads given by **z0**, non-empty reads given by **z1**, the percentage of range queries **q** and the percentage of writes given by **w**).  

5. The *[job]* section controls each job details determined by the parameter exactly after the period (.). The **use_gpu_if_avail** allows the user to train the model faster if GPU is available by setting this value to **True**.  

   &emsp;a. The *[job.DataGen]* is used to control all parameters associated to data generation. This can be configured to generate data either for the cost model or for the tuner respectively. The **generator** parameter controls which module the data is generated for (*LTuner* for the tuner model and *LCM* for the cost model). The **dir** determines the exact directory where the files will get generated. The description of all parameters are provided in the toml file.  

   &emsp;b. The *[job.LCMTrain]* can be used to control the model structure for the *Learned Cost Model* including the epochs, checkpoints, the loss function, optimizer, the Learning Rate scheduler as well as the batch size and the storage directories for each of the training and testing data.  

   &emsp;c. The *[job.LTuneTrain]* can be used to control the model structure for the *Learned Tuner* model including the epochs, checkpoints, the loss function, optimizer, the Learning Rate scheduler as well as the batch size and the storage directories for each of the training and testing data.  

## Project Structure
Our project is separated into repositories to maintain a consistent structure. Below is a short description for each of the repositories to help any user of this project:

1. **jobs** - This repo contains the main entry point defined in endure.py for each job. There are separate files dedicated to each job that can be identified using the filenames.

2. **endure** - This repo contains multiple other repositories divided by use case as defined below:  

   &emsp;a. **endure/lcm** - This repo contains the folders and files responsible for the *Learned Cost Model (LCM)* that helps learn the cost surface for the implemented solution. Within this directory, there is a *model* folder that contains the files for each LCM model structure (Classic, KLSM, QHybrid, Doestoevsky (YZLSM)). The *util* folder contains all utilities used by the models within *lcm*. The *data* folder contains code required for generating and using the input (data) files by the models.  

   &emsp;b. **endure/lsm** - This repo contains the analytical solver repository called *solver* that is used as the basis of comparison as well as files associated with the structure of the lsm. It has a *types.py* file that is used throughout our project to define each type used. The project also has cost files that use the equations from the cost model (as stated in the paper) to calculate the cost of each operation for all models. There is a also a data_generator file that is used for generation of data from a given sample space uniformly at random.  

   &emsp;c. **endure/ltune** - This repo contains the folders and files responsible for the *Learned Tuner (LTuner)* that helps predict the best configuration for the LSM Tree as per the solution proposed in the paper. Within this directory, there is a *model* folder that contains the files for each Ltuner model structure (Classic, KLSM, QHybrid, Doestoevsky (YZLSM)). The *util* folder contains all utilities used by the models within *ltune*. The *data* folder contains code required for generating and using the input (data) files by the models.  

   &emsp;d. **endure/util** - Contains utility files that are used generally by all modules.  