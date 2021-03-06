## A minimal implementation of Proximal Policy Optimization algorithm.
> PPO Vs DDPG or TD3.
+ If the environment is expensive to sample from, use DDPG, 
  since they're more sample efficient. 
+ If it's cheap to sample from, using PPO or a REINFORCE-based algorithm, 
  since they're straightforward to implement, robust to hyperparameters, 
  and easy to get working. You'll spend less wall-clock time training a 
  PPO-like algorithm in a cheap environment.
+ If you need to decide between DDPG and SAC, choose TD3. The performance 
  of SAC and DDPG is nearly identical when you compare on the basis of 
  whether or not a twin delayed update is used.
  
> Results:
+ Tested in gym and MuJoCo, both TD3 and PPO works.
+ TD3 is very sample efficient but use far more time for training.
+ PPO is faster than off policy algorithm, because using parallel
  training methods via mpi4py and computing efficiency.
+ TD3 may usually have better performance.  
  
> Usage:
+ Windows: change the code 
  
  ```os.path.abspath('.')``` -> ```os.path.abspath('..')```
  
  to store the experiments results in current folder.
  
+ Linux:
  Has been tested, just run the following code in terminal
  
  ```cd PPO```
  
  ```chmod +x ./test.sh```

  ```./run_experiments.sh```
