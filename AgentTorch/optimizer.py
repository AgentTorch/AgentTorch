from runner import Runner

# Supervised learning given step-wise values
ground_truth_data = None #Read from file
#To define model here
#Setup optimizer and scheduler
optimizer = None
scheduler = None
loss_fn = None #MSE or NLL based on type of data
while True: #Some learning dependent condition, #epochs
    model_pred = runner.step(num_steps = 1) #To insert model params here
    loss = loss_fn(model_pred, ground_truth_data)
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    if cond:
        break
    
# Supervised learning given emergent values
ground_truth_data = None #Read from file
#To define model here
#Setup optimizer and scheduler
optimizer = None
scheduler = None
loss_fn = None #MSE or NLL based on type of data
while True: #Some learning dependent condition, #epochs
    model_pred = runner() #To insert model params here, this tuns for one entire episode
    loss = loss_fn(model_pred, ground_truth_data)
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    if cond:
        break

# Reinforcement Learning (analytic PG) #This can be repeated for every class of agent
agent_class = None
objective = None #Define objective as a string, should be a part of state variables list
agent_policy = None #(fetch from controller?)
#Setup optimizer and scheduler
optimizer = None
scheduler = None
loss_fn = None #MSE or NLL based on type of data
while True: #Some learning dependent condition, #epochs
    model_pred = runner() #To insert model params here, this tuns for one entire episode
    agent_traj =  f(model_pred)
    agent_return = r(agent_traj)
    loss = -agent_return
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    if cond:
        break