import torch
test_set = torch.load('../../data/processed/test.pt')
testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*test_set), batch_size=64, shuffle=True)

def predict_class(data = testloader):
    model = torch.load('../../models/model.pth')
    test_picture, test_label = next(iter(data))
    output = model(test_picture.float())
    print(output)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(1, dim=1)
    print(top_class)
    

predict_class()