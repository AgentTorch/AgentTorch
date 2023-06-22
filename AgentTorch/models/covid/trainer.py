### Debugging pseudocode only

from runner import Runner
import torch.nn as nn

args = parser.parse_args()
config_file = args.config

runner1 = Runner(config_file)
runner2 = Runner(config_file)

# calibration logic
calib_nn = CalibNN()

opt = my_optimizer([runner1.parameters() + runner2.parameters() + calibNN.parameters()], lr=0.01)

for ix in range(num_train_steps):
        
    opt.zero_grad()
    
    learning_params = CalibNN(metadata)
    runner1.execute_from_parameters(learning_params)
    runner2.execute_from_parameters(learning_params)
    
    output1 = runner1.output_variable
    output2 = runner2.output_variable
    
    loss = nn.MSELoss()((output1 + output2)/2, ground_truth)
    
    loss.backward()
    opt.step()
    
    runner2.set_parameters()
    
    for name, param in runner.named_parameters(): 
        print(name, param.data)


