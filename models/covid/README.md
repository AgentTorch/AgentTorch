- Steps to run a simulation

1. Start calibration process: python trainer.py --c config_opt_llm.yaml

2. Change population and mobility network
    - Go to simulator_data (follow ayush notes)
    - Update num_agents variable in config
    - Regenerate disease_stages.csv (see: disease_stage_file.ipynb) to update initial infections

3. Change initial infections
    - Run disease_stages_file.ipynb (gives you a new disease_stages.csv)

Claims to make:
    - R_t = R_0 x behavior

1. Given R_t and behavior -> estimate R_O {Calibration} 
    R0 <----- {behavior} R_t

2. Given behavior, change R_0 ---> get new R_t {Variant changes: alpha to omicron: peak change}
    del(R0) {behavior} ----> R_t

3. Given R_0, change behavior ---> get new R_t {Behavior changes: fatigue?}
    del(behavior) {R0} ----> R_t

Goal is to improve model stability.
Ideas currently trying:
1. Learning rate scheduler
2. Differential learning rate for R0 and alpha parameters: this seems to be helping
3. Gradient clipping: not sure
4. Smaller learning rate: 1e-5 order seems to be helping [5e-3]
5. Change loss function: i) signed loss did not help much. ii) MSE-Loss does seem to help.
6. Make CalibNN architecture much smaller [did not converge]: Did not help much. DON'T USE THIS. Mobility conditioning data actually helps.


Good result experiments:
a) Experiment 1
1. Differential learning rate for R0 and alpha parameters
2. MSELoss
3. Base LR: 5e-3; R0_LR: 0.1*Base_LR; alpha_LR: 0.01*Base_LR
==================================================================================================
    running episode 0...
r0 values: tensor([3.5294, 3.5467], device='cuda:0', grad_fn=<SqueezeBackward0>)
align values: tensor([0.4737, 0.4980, 0.4560, 0.4927, 0.5208, 0.5749], device='cuda:0',
       grad_fn=<MeanBackward1>)

starting week 0... #cases, #cases_4wk for prompt 45, 43... sampling isolation probabilities
day 0, number of isolating agents 11958.0
day 1, number of isolating agents 11861.0
day 2, number of isolating agents 12005.0
day 3, number of isolating agents 11927.0
day 4, number of isolating agents 12003.0
day 5, number of isolating agents 11970.0
day 6, number of isolating agents 12046.0

starting week 1... incoming #cases 42.0... #cases, #cases_4wk for prompt 42, 47... sampling isolation probabilities
day 7, number of isolating agents 11337.0
day 8, number of isolating agents 11428.0
day 9, number of isolating agents 11370.0
day 10, number of isolating agents 11168.0
day 11, number of isolating agents 11226.0
day 12, number of isolating agents 11141.0
day 13, number of isolating agents 11285.0
predicted number of cases: tensor([42., 98.], device='cuda:0', grad_fn=<ToCopyBackward0>), actual number of cases: tensor([ 66., 101.], device='cuda:0'), train loss: 292.5, val loss: nan

running episode 1...
Skipping:  objects
r0 values: tensor([4.3400, 4.3400], device='cuda:0', grad_fn=<SqueezeBackward0>)
align values: tensor([0.5708, 0.2507, 0.2102, 0.7869, 0.7452, 0.5638], device='cuda:0',
       grad_fn=<MeanBackward1>)

starting week 0... #cases, #cases_4wk for prompt 45, 43... sampling isolation probabilities
day 0, number of isolating agents 11083.0
day 1, number of isolating agents 11097.0
day 2, number of isolating agents 11018.0
day 3, number of isolating agents 11026.0
day 4, number of isolating agents 11102.0
day 5, number of isolating agents 11109.0
day 6, number of isolating agents 11042.0

starting week 1... incoming #cases 55.0... #cases, #cases_4wk for prompt 55, 50... sampling isolation probabilities
day 7, number of isolating agents 13553.0
day 8, number of isolating agents 13510.0
day 9, number of isolating agents 13674.0
day 10, number of isolating agents 13570.0
day 11, number of isolating agents 13520.0
day 12, number of isolating agents 13586.0
day 13, number of isolating agents 13548.0
predicted number of cases: tensor([ 55., 148.], device='cuda:0', grad_fn=<ToCopyBackward0>), actual number of cases: tensor([ 66., 101.], device='cuda:0'), train loss: 1165.0, val loss: nan

running episode 2...
Skipping:  objects
r0 values: tensor([4.1444, 4.1396], device='cuda:0', grad_fn=<SqueezeBackward0>)
align values: tensor([0.4885, 0.5635, 0.4661, 0.5739, 0.5231, 0.5325], device='cuda:0',
       grad_fn=<MeanBackward1>)

starting week 0... #cases, #cases_4wk for prompt 45, 43... sampling isolation probabilities
day 0, number of isolating agents 12433.0
day 1, number of isolating agents 12412.0
day 2, number of isolating agents 12309.0
day 3, number of isolating agents 12559.0
day 4, number of isolating agents 12411.0
day 5, number of isolating agents 12388.0
day 6, number of isolating agents 12364.0

starting week 1... incoming #cases 50.0... #cases, #cases_4wk for prompt 50, 49... sampling isolation probabilities
day 7, number of isolating agents 14702.0
day 8, number of isolating agents 14730.0
day 9, number of isolating agents 14615.0
day 10, number of isolating agents 14679.0
day 11, number of isolating agents 14406.0
day 12, number of isolating agents 14643.0
day 13, number of isolating agents 14641.0
predicted number of cases: tensor([ 50., 100.], device='cuda:0', grad_fn=<ToCopyBackward0>), actual number of cases: tensor([ 66., 101.], device='cuda:0'), train loss: 128.5, val loss: nan

running episode 3...
Skipping:  objects
r0 values: tensor([4.1023, 4.0997], device='cuda:0', grad_fn=<SqueezeBackward0>)
align values: tensor([0.3551, 0.6539, 0.4602, 0.4954, 0.4179, 0.4713], device='cuda:0',
       grad_fn=<MeanBackward1>)

starting week 0... #cases, #cases_4wk for prompt 45, 43... sampling isolation probabilities
day 0, number of isolating agents 13483.0
day 1, number of isolating agents 13401.0
day 2, number of isolating agents 13437.0
day 3, number of isolating agents 13388.0
day 4, number of isolating agents 13515.0
day 5, number of isolating agents 13429.0
day 6, number of isolating agents 13479.0

starting week 1... incoming #cases 56.0... #cases, #cases_4wk for prompt 56, 50... sampling isolation probabilities
day 7, number of isolating agents 16499.0
day 8, number of isolating agents 16417.0
day 9, number of isolating agents 16580.0
day 10, number of isolating agents 16676.0
day 11, number of isolating agents 16540.0
day 12, number of isolating agents 16538.0
day 13, number of isolating agents 16531.0
predicted number of cases: tensor([ 56., 100.], device='cuda:0', grad_fn=<ToCopyBackward0>), actual number of cases: tensor([ 66., 101.], device='cuda:0'), train loss: 50.5, val loss: nan

running episode 4...
Skipping:  objects
r0 values: tensor([4.1388, 4.1337], device='cuda:0', grad_fn=<SqueezeBackward0>)
align values: tensor([0.2620, 0.6537, 0.3665, 0.4840, 0.3832, 0.3862], device='cuda:0',
       grad_fn=<MeanBackward1>)

starting week 0... #cases, #cases_4wk for prompt 45, 43... sampling isolation probabilities
day 0, number of isolating agents 10961.0
day 1, number of isolating agents 10868.0
day 2, number of isolating agents 11039.0
day 3, number of isolating agents 10894.0
day 4, number of isolating agents 10942.0
day 5, number of isolating agents 10690.0
day 6, number of isolating agents 10801.0

starting week 1... incoming #cases 57.0... #cases, #cases_4wk for prompt 57, 51... sampling isolation probabilities
day 7, number of isolating agents 12316.0
day 8, number of isolating agents 12517.0
day 9, number of isolating agents 12361.0
day 10, number of isolating agents 12446.0
day 11, number of isolating agents 12631.0
day 12, number of isolating agents 12396.0
day 13, number of isolating agents 12263.0
predicted number of cases: tensor([ 57., 128.], device='cuda:0', grad_fn=<ToCopyBackward0>), actual number of cases: tensor([ 66., 101.], device='cuda:0'), train loss: 405.0, val loss: nan

running episode 5...
Skipping:  objects
r0 values: tensor([4.0391, 4.0230], device='cuda:0', grad_fn=<SqueezeBackward0>)
align values: tensor([0.3055, 0.7276, 0.3806, 0.4303, 0.3905, 0.4055], device='cuda:0',
       grad_fn=<MeanBackward1>)

starting week 0... #cases, #cases_4wk for prompt 45, 43... sampling isolation probabilities
day 0, number of isolating agents 12478.0
day 1, number of isolating agents 12434.0
day 2, number of isolating agents 12530.0
day 3, number of isolating agents 12341.0
day 4, number of isolating agents 12479.0
day 5, number of isolating agents 12403.0
day 6, number of isolating agents 12476.0

starting week 1... incoming #cases 55.0... #cases, #cases_4wk for prompt 55, 50... sampling isolation probabilities
day 7, number of isolating agents 15071.0
day 8, number of isolating agents 14822.0
day 9, number of isolating agents 14947.0
day 10, number of isolating agents 14881.0
day 11, number of isolating agents 14946.0
day 12, number of isolating agents 14863.0
day 13, number of isolating agents 14875.0
predicted number of cases: tensor([ 55., 119.], device='cuda:0', grad_fn=<ToCopyBackward0>), actual number of cases: tensor([ 66., 101.], device='cuda:0'), train loss: 222.5, val loss: nan

==================================================================================================
Experiment 2
1. Differential learning rate for R0 and alpha parameters
2. L1Loss
3. Base LR: 5e-3; R0_LR: 0.1*Base_LR; alpha_LR: 0.01*Base_LR

Experiment 3
1. Differential learning rate for R0 and alpha parameters
2. L1Loss
3. Base LR: 5e-3; R0_LR: 0.1*Base_LR; alpha_LR: 0.01*Base_LR
4. we calibrate R0 and 1-alpha. Decrease both decreases case spread [previously alpha and R0 were in opposite directions]

r0 values: tensor([3.0286, 3.0565], device='cuda:0', grad_fn=<SqueezeBackward0>)
how non compliant is LLM?: tensor([0.4310, 0.6010, 0.5681, 0.6720, 0.4859, 0.4406], device='cuda:0',
       grad_fn=<MeanBackward1>)

starting week 0... #cases, #cases_4wk for prompt 45, 43... sampling isolation probabilities
day 0, number of isolating agents 11656.0
day 1, number of isolating agents 11655.0
day 2, number of isolating agents 11594.0
day 3, number of isolating agents 11613.0
day 4, number of isolating agents 11659.0
day 5, number of isolating agents 11536.0
day 6, number of isolating agents 11640.0

starting week 1... incoming #cases 45.0... #cases, #cases_4wk for prompt 45, 48... sampling isolation probabilities
day 7, number of isolating agents 11481.0
day 8, number of isolating agents 11227.0
day 9, number of isolating agents 11317.0
day 10, number of isolating agents 11369.0
day 11, number of isolating agents 11498.0
day 12, number of isolating agents 11379.0
day 13, number of isolating agents 11320.0
predicted number of cases: tensor([ 45., 111.], device='cuda:0', grad_fn=<ToCopyBackward0>), actual number of cases: tensor([ 66., 101.], device='cuda:0'), train loss: 15.5, val loss: nan

running episode 1...
Skipping:  objects
r0 values: tensor([2.8785, 2.8779], device='cuda:0', grad_fn=<SqueezeBackward0>)
how non compliant is LLM?: tensor([0.1902, 0.3782, 0.4697, 0.5849, 0.3276, 0.3898], device='cuda:0',
       grad_fn=<MeanBackward1>)

starting week 0... #cases, #cases_4wk for prompt 45, 43... sampling isolation probabilities
day 0, number of isolating agents 13750.0
day 1, number of isolating agents 13697.0
day 2, number of isolating agents 13721.0
day 3, number of isolating agents 13837.0
day 4, number of isolating agents 13816.0
day 5, number of isolating agents 13900.0
day 6, number of isolating agents 13910.0

starting week 1... incoming #cases 40.0... #cases, #cases_4wk for prompt 40, 46... sampling isolation probabilities
day 7, number of isolating agents 12542.0
day 8, number of isolating agents 12618.0
day 9, number of isolating agents 12602.0
day 10, number of isolating agents 12557.0
day 11, number of isolating agents 12613.0
day 12, number of isolating agents 12540.0
day 13, number of isolating agents 12634.0
predicted number of cases: tensor([40., 75.], device='cuda:0', grad_fn=<ToCopyBackward0>), actual number of cases: tensor([ 66., 101.], device='cuda:0'), train loss: 26.0, val loss: nan

running episode 2...
Skipping:  objects
r0 values: tensor([3.5340, 3.6400], device='cuda:0', grad_fn=<SqueezeBackward0>)
how non compliant is LLM?: tensor([0.3883, 0.6409, 0.4919, 0.3340, 0.3738, 0.3966], device='cuda:0',
       grad_fn=<MeanBackward1>)

starting week 0... #cases, #cases_4wk for prompt 45, 43... sampling isolation probabilities
day 0, number of isolating agents 12840.0
day 1, number of isolating agents 12618.0
day 2, number of isolating agents 12831.0
day 3, number of isolating agents 12855.0
day 4, number of isolating agents 12687.0
day 5, number of isolating agents 12737.0
day 6, number of isolating agents 12717.0

starting week 1... incoming #cases 49.0... #cases, #cases_4wk for prompt 49, 49... sampling isolation probabilities
day 7, number of isolating agents 13936.0
day 8, number of isolating agents 14119.0
day 9, number of isolating agents 14092.0
day 10, number of isolating agents 14073.0
day 11, number of isolating agents 14280.0
day 12, number of isolating agents 13990.0
day 13, number of isolating agents 14058.0
predicted number of cases: tensor([ 49., 102.], device='cuda:0', grad_fn=<ToCopyBackward0>), actual number of cases: tensor([ 66., 101.], device='cuda:0'), train loss: 9.0, val loss: nan

running episode 3...
Skipping:  objects
r0 values: tensor([3.7570, 3.7627], device='cuda:0', grad_fn=<SqueezeBackward0>)
how non compliant is LLM?: tensor([0.3408, 0.6663, 0.4725, 0.2228, 0.3270, 0.3162], device='cuda:0',
       grad_fn=<MeanBackward1>)

starting week 0... #cases, #cases_4wk for prompt 45, 43... sampling isolation probabilities
day 0, number of isolating agents 12145.0
day 1, number of isolating agents 12234.0
day 2, number of isolating agents 12242.0
day 3, number of isolating agents 12228.0
day 4, number of isolating agents 12231.0
day 5, number of isolating agents 12234.0
day 6, number of isolating agents 12220.0

starting week 1... incoming #cases 48.0... #cases, #cases_4wk for prompt 48, 48... sampling isolation probabilities
day 7, number of isolating agents 14065.0
day 8, number of isolating agents 14051.0
day 9, number of isolating agents 13873.0
day 10, number of isolating agents 14037.0
day 11, number of isolating agents 14141.0
day 12, number of isolating agents 14078.0
day 13, number of isolating agents 13846.0
predicted number of cases: tensor([ 48., 107.], device='cuda:0', grad_fn=<ToCopyBackward0>), actual number of cases: tensor([ 66., 101.], device='cuda:0'), train loss: 12.0, val loss: nan

running episode 4...
Skipping:  objects
r0 values: tensor([3.4906, 3.4555], device='cuda:0', grad_fn=<SqueezeBackward0>)
how non compliant is LLM?: tensor([0.2927, 0.6028, 0.3634, 0.2865, 0.3412, 0.2625], device='cuda:0',
       grad_fn=<MeanBackward1>)

starting week 0... #cases, #cases_4wk for prompt 45, 43... sampling isolation probabilities
day 0, number of isolating agents 14814.0
day 1, number of isolating agents 14724.0
day 2, number of isolating agents 14876.0
day 3, number of isolating agents 14869.0
day 4, number of isolating agents 14775.0
day 5, number of isolating agents 14682.0
day 6, number of isolating agents 14931.0

starting week 1... incoming #cases 43.0... #cases, #cases_4wk for prompt 43, 47... sampling isolation probabilities
day 7, number of isolating agents 13086.0
day 8, number of isolating agents 13279.0
day 9, number of isolating agents 13221.0
day 10, number of isolating agents 13122.0
day 11, number of isolating agents 13265.0
day 12, number of isolating agents 13170.0
day 13, number of isolating agents 12943.0
predicted number of cases: tensor([43., 76.], device='cuda:0', grad_fn=<ToCopyBackward0>), actual number of cases: tensor([ 66., 101.], device='cuda:0'), train loss: 24.0, val loss: nan

running episode 5...
Skipping:  objects
r0 values: tensor([3.8021, 3.7635], device='cuda:0', grad_fn=<SqueezeBackward0>)
how non compliant is LLM?: tensor([0.2874, 0.6274, 0.3876, 0.1956, 0.3191, 0.2641], device='cuda:0',
       grad_fn=<MeanBackward1>)

starting week 0... #cases, #cases_4wk for prompt 45, 43... sampling isolation probabilities
day 0, number of isolating agents 14048.0
day 1, number of isolating agents 13913.0
day 2, number of isolating agents 13940.0
day 3, number of isolating agents 14159.0
day 4, number of isolating agents 14125.0
day 5, number of isolating agents 14243.0
day 6, number of isolating agents 14120.0

starting week 1... incoming #cases 50.0... #cases, #cases_4wk for prompt 50, 49... sampling isolation probabilities
day 7, number of isolating agents 16339.0
day 8, number of isolating agents 16173.0
day 9, number of isolating agents 16328.0
day 10, number of isolating agents 16325.0
day 11, number of isolating agents 16252.0
day 12, number of isolating agents 16264.0
day 13, number of isolating agents 16215.0
predicted number of cases: tensor([50., 85.], device='cuda:0', grad_fn=<ToCopyBackward0>), actual number of cases: tensor([ 66., 101.], device='cuda:0'), train loss: 16.0, val loss: nan

==================================================================================================