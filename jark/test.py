import torch
input1 = torch.randn(2, 3)
input2 = torch.randn(2, 3)
input3 = torch.randn(5, 3)
input_list = [input1, input2, input3]


output1 = torch.cat(input_list, dim=0)
print(output1)
print(output1.size()) # torch.Size([9, 3, 4])

output2 = torch.cat(input_list, dim=1) # error
print(output2.size()) 

output3 = torch.cat(input_list, dim=2) # error
print(output3.size()) 