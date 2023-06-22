import torch

iput = torch.randn(6)
gt = 8*iput

loss = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(run_model.parameters(), lr = 0.01)
print(run_model.state_dict().items())

for i in range(5):
    print(f"Iteration {i}")
    optimizer.zero_grad()
    y = run_model(iput)
    l = loss(y, gt)
    l.backward()
    optimizer.step()
    print(run_model.state_dict().items())