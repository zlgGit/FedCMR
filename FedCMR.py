import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import Utils
from models import GeneratorCifar10,Cifar10CNN,Cifar100CNN,GeneratorCifar100,SimpleCNN,SimpleGenerator,TinyImageNetCNN,TinyImageNetGenerator
import copy
import argparse
import time
import Saveflie

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

learning_rate = 0.01
weight_decay = 0.00001  
momentum = 0.9         

def kl_divergence(p, q):
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        return torch.sum(p * (torch.log(p + 1e-8) - torch.log(q + 1e-8)), dim=-1).mean()

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
    
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            features,output = model(data)
            predicted = torch.argmax(output,dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total

def evaluate_model_fedcmr(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            features = model.feature_extractor(data)  
            output = model.classifier(features)

            predicted = torch.argmax(output, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy

class Client:
    def __init__(self, model, train_loader, generator=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9) 
        self.criterion = nn.CrossEntropyLoss()
        self.generator=generator
        
    def train_model(self, num_epochs=2):
        self.model.train()
        for epoch in range(num_epochs):
            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                _,output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

        return self.model.state_dict()

    def train_model_with_kl(self, num_epochs=2):
        self.generator_optimizer = optim.SGD(
            self.generator.parameters(), 
            lr=0.01, momentum=0.9)
        self.generator.to(device)
        self.model.train()
        self.generator.train()
        for epoch in range(num_epochs):
            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)
                
                self.optimizer.zero_grad()

                features_real = self.model.feature_extractor(data)

                z = torch.randn(data.size(0), 100).to(device)  
                target_onehot = F.one_hot(target, num_classes=num_class).float().to(device)  
                noise = torch.cat((z, target_onehot), dim=1) 
                features_fake = self.generator(noise)  
 
                output=self.model.classifier(features_real)

                kl_loss = kl_divergence(output, target_onehot)

                classification_loss = self.criterion(output, target)
                total_loss=classification_loss+kl_loss
                total_loss.backward()  
                self.optimizer.step()
                
                generator_loss = kl_divergence(features_real.detach(), features_fake)
                generator_loss.backward()
                self.generator_optimizer.step()


        return self.model.feature_extractor.state_dict(), \
               self.model.classifier.state_dict(), \
               self.generator.state_dict()
    
    def train_model_with_dis(self, globel_model,num_epochs=2):
        self.model.train()
        old_model=copy.deepcopy(self.model)

        globel_model=global_model
        for epoch in range(num_epochs):
            index=0
            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)
 

                self.optimizer.zero_grad()
                currunt_header,output_current = self.model(data)
                loss = self.criterion(output_current, target)
                global_model.eval()
                global_header,output_global = global_model(data)
                older_header,output_old=old_model(data)
  

                cosine_g = torch.nn.functional.cosine_similarity(currunt_header, global_header,dim=-1).mean()
                cosine_c = torch.nn.functional.cosine_similarity(currunt_header, older_header,dim=-1).mean()
                cons_los=-torch.log(torch.exp(cosine_g)/(torch.exp(cosine_g)+torch.exp(cosine_c)))
                    
                loss=loss+cons_los
                loss.backward()
                self.optimizer.step()                  

        return self.model.state_dict()
    
    def get_parameters_generator(self):
        return self.generator.state_dict()

    def set_parameters_generator(self, parameters):
        self.generator.load_state_dict(parameters)
    
    def get_parameters_classfier(self):
        return self.model.classifier.state_dict()

    def set_parameters_classfier(self, parameters):
        self.model.classifier.load_state_dict(parameters) 

    def get_parameters_extractor(self):
        return self.feature_extractor.state_dict()

    def set_parameters_extractor(self, parameters):
        self.model.feature_extractor.load_state_dict(parameters) 

    def get_parameters(self):
        return self.model.state_dict()

    def set_parameters(self, parameters):
        self.model.load_state_dict(parameters)

    def predict_fedcfr(self, input_data):
        
        self.model.classifier.eval()
        with torch.no_grad():
            input_data = input_data.to(device)
            heaader,output = self.model.forward_classifier(input_data)

        return heaader,output 
    def predict(self, input_data):
        self.model.eval()
        with torch.no_grad():
            input_data = input_data.to(device)
            output = self.model(input_data)

        return output  
def contrastiveLoss(anchor, positive, negatives, temperature=0.5, margin=0.99):

    pos_sim = F.cosine_similarity(anchor, positive, dim=-1)
    pos_sim_loss=torch.mean(pos_sim - 1)** 2
    
    neg_sim = F.cosine_similarity(anchor.unsqueeze(0), negatives, dim=-1)
    neg_sim_loss = torch.mean(neg_sim - margin)** 2

    pos_sim_exp = torch.exp((pos_sim)/ temperature)
    neg_sim_exp = torch.exp((-neg_sim) / temperature)
    total_sim_exp = neg_sim_exp.sum(dim=0) 
    loss = -torch.log((pos_sim_exp) /(total_sim_exp+pos_sim_exp))
    total_loss = loss.mean()

    # print("pos_sim:", pos_sim)
    # print("neg_sim:", neg_sim)

    # print("pos_sim_exp:", pos_sim_exp)
    # print("neg_sim_exp:", neg_sim_exp)
    # print("contrastiveLoss:", total_loss)

    return total_loss

def trainCFR(global_model, generator, clients, global_optimizer,serverEpoch=100):
    
    generator=generator.to(device)
    global_model=global_model.to(device)
    serverEpoch=100

    for i in range(serverEpoch):

        batch_size = 16 
        z = torch.randn(batch_size, 100).to(device) 
        random_labels = torch.randint(0, 10, (batch_size,)).to(device) 
        random_labels_one_hot = F.one_hot(random_labels, num_classes=num_class).float().to(device)
        noise = torch.cat((z, random_labels_one_hot), dim=1).to(device)  
        generated_features = generator(noise)
        generated_features=generated_features.to(device)

        all_client_outputs = [] 
        all_client_headers = [] 

        global_model.train()
        global_model.zero_grad()

        output_global_header,output_global = global_model.forward_classifier(generated_features)

        contrastive_loss=0

        for client in clients:

            local_header,output_client = client.predict_fedcfr(generated_features)
            all_client_outputs.append(output_client.detach())  
            all_client_headers.append(local_header.detach()) 


        weighted_average_output = torch.mean(torch.stack(all_client_outputs), dim=0)
        weighted_average_header = torch.mean(torch.stack(all_client_headers), dim=0)  

 
        contrastive_loss=contrastiveLoss(output_global_header,weighted_average_header,torch.stack(all_client_headers))


        softmax_output = F.softmax(weighted_average_output, dim=1)
        predicted_labels = torch.argmax(softmax_output, dim=1)
        target=F.one_hot(predicted_labels, num_classes=num_class).float().to(device) 

        loss = kl_divergence(output_global,target)

        total_loss=10*loss+0.01*contrastive_loss

        total_loss.backward()    
        global_optimizer.step()

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--rounds', type=int, default=100,  help='number of rounds')
    parser.add_argument('--epochs', type=int, default=1,  help='local epochs')
    parser.add_argument('--clients', type=int, default=10,  help='number of clients')
    parser.add_argument('--dir', type=float, default=0.1,  help='dirichlet distribution')
    parser.add_argument('--model', type=str, default="fedcmr",  help='model')
    return parser.parse_args() 



if __name__ == "__main__":

    args=get_args()
    filename = f"{args.model}_{args.dataset}_rounds{args.rounds}_clients{args.clients}_dir{args.dir}.txt"

    if args.dataset.lower()== "cifar10":

        global_model = Cifar10CNN()
        dataset=Utils.load_train_data_CIFAR10()
        test_loader=Utils.load_test_loader_CIFAR10()
        global_generator=GeneratorCifar10()
        num_class=10

    elif args.dataset.lower() == "cifar100":
    
        global_model = Cifar100CNN()
        dataset=Utils.load_train_data_CIFAR100()
        test_loader=Utils.load_test_loader_CIFAR100()
        global_generator=GeneratorCifar100()
        num_class=100
        global_model = Cifar100CNN(num_classes=num_class)

    elif args.dataset.lower() == "mnist":

        global_model = SimpleCNN()
        dataset=Utils.load_train_data_MNIST()
        test_loader=Utils.load_test_loader_MNIST()
        global_generator=SimpleGenerator()
        num_class=10

    elif args.dataset.lower() == "fmnist":

        global_model = SimpleCNN()
        dataset=Utils.load_train_data_FashionMNIST()
        test_loader=Utils.load_test_loader_FashionMNIST()
        global_generator=SimpleGenerator()
        num_class=10

    elif args.dataset.lower() == "tinyimagenet":

        global_model = TinyImageNetCNN()
        dataset,class_to_idx=Utils.load_train_data_TinyImageNet()
        test_loader=Utils.load_test_loader_TinyImageNet(class_to_idx)
        global_generator=TinyImageNetGenerator()
        num_class=200

    else:
        raise ValueError("Invalid dataset") 


    if args.model.lower() == "fedavg":

        global_model.to(device)
        global_weights = global_model.state_dict()

        client_loaders = Utils.create_client_loaders_dirichlet(dataset=dataset, num_clients=args.clients, alpha=args.dir,num_class=num_class)

        clients = [Client(copy.deepcopy(global_model), client_loader) for client_loader in client_loaders]

        for round in range(args.rounds):

                if round==0:
                    start_time=time.perf_counter()
                client_weights = []
                for client in clients:
                    client.set_parameters(global_weights)

                for client in clients:
                    weights= client.train_model(num_epochs=args.epochs)
                    client_weights.append(weights)
                
                global_weights = Utils.aggregate_weights(client_weights)
                global_model.load_state_dict(global_weights)
                accuracy = evaluate_model(global_model,test_loader=test_loader)
                print(f"Accuracy {round + 1}: {accuracy:.4f}")
                Saveflie.save_training_info(round=round,accuracy=accuracy,filename=filename)


    elif args.model.lower() == "moon":
            
            global_model.to(device)
            global_weights = global_model.state_dict()

            client_loaders = Utils.create_client_loaders_dirichlet(dataset=dataset, num_clients=args.clients, alpha=args.dir,num_class=num_class)
            clients = [Client(copy.deepcopy(global_model), client_loader) for client_loader in client_loaders]

            for round in range(args.rounds):

                if round==0:
                    start_time=time.perf_counter()
                client_weights = []
                for client in clients:
                    client.set_parameters(global_weights)

                for client in clients:
                    weights= client.train_model_with_dis(globel_model=global_model,num_epochs=args.epochs)
                    client_weights.append(weights)
                
                global_weights = Utils.aggregate_weights(client_weights)

                global_model.load_state_dict(global_weights)

                accuracy = evaluate_model(global_model,test_loader=test_loader)
                
                print(f"Accuracy {round + 1}: {accuracy:.4f}")

                Saveflie.save_training_info(round=round,accuracy=accuracy,filename=filename)


    elif args.model.lower() == "fedcmr":

        global_model.to(device)
        global_generator.to(device)
        global_weights_classfier = global_model.classifier.state_dict()
        global_weights_extractor = global_model.feature_extractor.state_dict()
        global_weights_generator = global_generator.state_dict()

        client_loaders = Utils.create_client_loaders_dirichlet(dataset=dataset, num_clients=args.clients, alpha=args.dir,num_class=num_class)
        clients = [Client(copy.deepcopy(global_model), client_loader,copy.deepcopy(global_generator)) for client_loader in client_loaders]

        for round in range(args.rounds):
                client_weights_classfier = []
                client_weights_extractor = []
                client_weights_generator = []

                for client in clients:
                    client.set_parameters_extractor(global_weights_extractor)
                    client.set_parameters_classfier(global_weights_classfier)
                    client.set_parameters_generator(global_weights_generator)

                for client in clients:
            
                        weight_extractor,weight_classfier,weight_generator = client.train_model_with_kl(num_epochs=args.epochs)

                        client_weights_extractor.append(weight_extractor)
                        client_weights_classfier.append(weight_classfier)
                        client_weights_generator.append(weight_generator)

                global_weights_classfier = Utils.aggregate_weights(client_weights_classfier)
                global_weights_extractor = Utils.aggregate_weights(client_weights_extractor)
                global_weights_generator = Utils.aggregate_weights(client_weights_generator)

                global_model.classifier.load_state_dict(global_weights_classfier)
                global_model.feature_extractor.load_state_dict(global_weights_extractor)
                global_generator.load_state_dict(global_weights_generator)

                accuracy = evaluate_model(global_model,test_loader=test_loader)
                print(f"Accuracy {round + 1}: {accuracy:.4f}")

                Saveflie.save_training_info(round=round,accuracy=accuracy,filename=filename)

                learning_rate = 0.01
                weight_decay = 0.00001 
                momentum = 0.9          
                global_optimizer = optim.SGD(global_model.classifier.parameters(),lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
                trainCFR(global_model=global_model, generator=global_generator, clients=clients, global_optimizer=global_optimizer, serverEpoch=100)
                global_weights_classfier=global_model.classifier.state_dict()