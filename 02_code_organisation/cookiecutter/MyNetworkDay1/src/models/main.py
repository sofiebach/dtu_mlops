import sys
import argparse
import numpy as np
import torch
from model import MyAwesomeModel
from torch import nn, optim
import matplotlib.pyplot as plt
import os

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.003)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        model = MyAwesomeModel()
        train_set = torch.load('../../data/processed/training.pt')
        trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*train_set), batch_size=64, shuffle=True)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        epochs = 5
        steps = 0

        train_losses, test_losses = [], []
        loss_epoch = []
        epoch_no = []
        for e in range(epochs):
            print("Starting epoch ", e+1)
            running_loss = 0
            for images, labels in trainloader:
                model.train()
                optimizer.zero_grad()
        
                log_ps = model(images.float())
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            print('Loss: ', np.mean(train_losses))

            # for epoch vs loss plot 
            epoch_no.append(e+1)
            loss_epoch.append(np.mean(train_losses))

        torch.save(model, '../../models/model.pth') # save model 
        plt.plot(epoch_no, loss_epoch, label='Training loss')
        plt.xlabel('Epoch number')
        plt.legend()
        plt.show()    
        plt.savefig('../../reports/figures/loss_curve')    
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="model.pth")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        if args.load_model_from:
            model = MyAwesomeModel()
            model.load_state_dict(torch.load('../../models/model.pth')) 
            #model = torch.load(args.load_model_from)
        test_set = torch.load('../../data/processed/test.pt')
        testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*test_set), batch_size=64, shuffle=True)
        
        running_accuracy = []
        # turn off gradients
        with torch.no_grad():
  
             # validation pass here
            for images, labels in testloader:
                model.eval()
                # validation pass here
                output = model(images.float())
                #print(output.shape)
                ps = torch.exp(output)

                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                running_accuracy.append(accuracy)

            # Output of the network are log-probabilities, need to take exponential for probabilities    
            #helper.view_classify(images.view(1, 28, 28), ps)    
            print('Accuracy:' , np.mean(running_accuracy)*100, '%')

    def test(self):
        #Load model and data
        model = MyAwesomeModel()
        model.load_state_dict(torch.load('./models/model.pth'))
        test_set = torch.load('./data/processed/test.pt')
        testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
        # Test out your network!
        model.eval()

        dataiter = iter(testloader)
        images, labels = next(dataiter)
        print(images.shape)

        #img = images[0]
        # Turn off gradients to speed up this part
        with torch.no_grad():
            logps = model(images)

        # Output of the network are log-probabilities, need to take exponential for probabilities
        ps = torch.exp(logps)

        # Plot the image and probabilities
        helper.view_classify(images.view(1, 28, 28), ps)

    



if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    