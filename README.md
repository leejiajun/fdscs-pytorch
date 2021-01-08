# fdscs-pytorch

The original model is from [FDSCS](https://github.com/ayanc/fdscs).

We has reproduced fdscs in PyTorch. The main work is to implement Cencus, Hamming and Abosulte Difference by the operators of PyTorch. 

You can directly put fdscs into the project [PSM](https://github.com/JiaRenChang/PSMNet), then strat training.

# Loss function:
```
output = model(imgL, imgR)
output = torch.squeeze(output, 1)

loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)
L1 = torch.abs(output - disp_true)
pc = (L1 < 1.0).float()
pc = torch.mean(pc[mask])
pc3 = (L1 < 3.0).float()
pc3 = torch.mean(pc3[mask])

print("loss={:.4f}, pc={:.4f}, pc3={:.4f}".format(loss, pc, pc3), end=" ")
```
