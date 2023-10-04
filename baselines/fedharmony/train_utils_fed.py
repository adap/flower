# Nicola Dinsdale 2022
# Functions for training and validating the model
########################################################################################################################
# Import dependencies
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score


########################################################################################################################
def val_fedprox_gaussian_unlearning_4_sites(args, models, val_loader, criterions, dist):
    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [criteron, conf_criterion, domain_criterion] = criterions

    encoder.eval()
    regressor.eval()
    domain_predictor.eval()

    val_loss = 0
    val_acc = 0

    true_domains = []
    pred_domains = []

    batches = 0
    with torch.no_grad():
        for batch_idx, (data, target, domain) in enumerate(val_loader):
            target = target.float()
            max_batch = len(data)
            # Now update just the domain classifier
            n1 = np.random.randint(1, max_batch - 3)  # Must be at least one from each
            n2 = np.random.randint(1, max_batch - n1 - 2)
            n3 = np.random.randint(1, max_batch - n1 - n2 - 1)
            n4 = max_batch - n1 - n2 - n3
            if n4 < 1:
                assert ValueError('N4 must be greater that zero')

            if cuda:
                data, target, domain = data.cuda(), target.cuda(), domain.cuda()
            data, target, domain = Variable(data), Variable(target), Variable(domain)

            if list(data.size())[0] == args.batch_size:
                batches += 1
                features = encoder(data)
                output_pred = regressor(features)
                loss = criteron(output_pred, target, [encoder.state_dict(), regressor.state_dict()])
                val_loss += loss

                features = encoder(data)
                features = features[:n1]
                domain = domain[:n1]
                features2 = np.zeros((n2, 64))
                for i in range(0, n2):
                    features2[i] = np.random.normal(dist[1], dist[2])
                domain2 = np.ones((n2,)) * dist[0]
                features3 = np.zeros((n3, 64))
                for i in range(0, n3):
                    features3[i] = np.random.normal(dist[4], dist[5])
                domain3 = np.ones((n3,)) * dist[3]
                features4 = np.zeros((n4, 64))
                for i in range(0, n4):
                    features4[i] = np.random.normal(dist[7], dist[8])
                domain4 = np.ones((n4,)) * dist[6]

                features2 = torch.from_numpy(features2)
                features3 = torch.from_numpy(features3)
                features4 = torch.from_numpy(features4)
                domain2 = torch.from_numpy(domain2)
                domain3 = torch.from_numpy(domain3)
                domain4 = torch.from_numpy(domain4)

                features2 = features2.type(torch.FloatTensor)
                features3 = features3.type(torch.FloatTensor)
                features4 = features4.type(torch.FloatTensor)
                domain = domain.type(torch.LongTensor)
                domain2 = domain2.type(torch.LongTensor)
                domain3 = domain3.type(torch.LongTensor)
                domain4 = domain4.type(torch.LongTensor)
                features2, features3, features4 = Variable(features2), Variable(features3), Variable(
                    features4)
                domain, domain2, domain3, domain4 = Variable(domain), Variable(domain2), Variable(
                    domain3), Variable(domain4)

                if cuda:
                    features = features.cuda()
                    features2 = features2.cuda()
                    features3 = features3.cuda()
                    features4 = features4.cuda()
                    domain = domain.cuda()
                    domain2 = domain2.cuda()
                    domain3 = domain3.cuda()
                    domain4 = domain4.cuda()

                features = torch.cat((features, features2, features3, features4), 0)
                domain = domain.view(-1)
                domain = torch.cat((domain, domain2, domain3, domain4), 0)

                output_dm = domain_predictor.forward(features)
                output_dm = np.argmax(output_dm.detach().cpu().numpy(), axis=1)
                domain_target = domain.detach().cpu().numpy()
                true_domains.append(domain_target)
                pred_domains.append(output_dm)

    val_loss = val_loss / batches
    val_loss_copy = np.copy(val_loss.detach().cpu().numpy())

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)
    av_acc_copy = np.copy(acc)
    del val_loss
    del val_acc

    print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss_copy, flush=True))
    print('Validation set: Average Acc: {:.4f}\n'.format(av_acc_copy, flush=True))

    return val_loss_copy, av_acc_copy

def train_fedprox_gaussian_unlearning_4_sites(args, models, train_loader, optimizers, criterions, epoch, dist):
    cuda = torch.cuda.is_available()
    [encoder, regressor, domain_predictor] = models
    [optimizer, optimizer_conf, optimizer_dm] = optimizers
    [criteron, conf_criterion, domain_criterion] = criterions

    regressor_loss = 0
    domain_loss = 0
    conf_loss = 0

    encoder.train()
    regressor.train()
    domain_predictor.train()

    true_domains = []
    pred_domains = []

    batches = 0

    for batch_idx, (data, target, domain) in enumerate(train_loader):
        print("batch number: "+str(batches)+" batch idx : "+str(batch_idx))
        max_batch = len(data)
        print("batch size:  "+str(max_batch))
        target = target.float()
        if cuda:
            data, target, domain = data.cuda(), target.cuda(), domain.cuda()
        data, target, domain = Variable(data), Variable(target), Variable(domain)

        if list(data.size())[0] == args.batch_size:
            batches += 1

            # First update the encoder and regressor
            optimizer.zero_grad()
            features = encoder(data)
            output_pred = regressor(features)
            loss = criteron(output_pred, target, [encoder.state_dict(), regressor.state_dict()])
            loss.backward()
            optimizer.step()

            # Now update just the domain classifier
            n1 = np.random.randint(1, max_batch - 3)  # Must be at least one from each
            n2 = np.random.randint(1, max_batch - n1 - 2)
            n3 = np.random.randint(1, max_batch - n1 - n2 - 1)
            n4 = max_batch - n1 - n2 - n3
            if n4 < 1:
                assert ValueError('N4 must be greater that zero')

            features = encoder(data)
            features = features[:n1]
            domain = domain[:n1]
            features2 = np.zeros((n2, 64))
            for i in range(0, n2):
                features2[i] = np.random.normal(dist[1], dist[2])
            domain2 = np.ones((n2,)) * dist[0]
            features3 = np.zeros((n3, 64))
            for i in range(0, n3):
                features3[i] = np.random.normal(dist[4], dist[5])
            domain3 = np.ones((n3,)) * dist[3]
            features4 = np.zeros((n4, 64))
            for i in range(0, n4):
                features4[i] = np.random.normal(dist[7], dist[8])
            domain4 = np.ones((n4,)) * dist[6]

            features2 = torch.from_numpy(features2)
            features3 = torch.from_numpy(features3)
            features4 = torch.from_numpy(features4)
            domain2 = torch.from_numpy(domain2)
            domain3 = torch.from_numpy(domain3)
            domain4 = torch.from_numpy(domain4)

            features2 = features2.type(torch.FloatTensor)
            features3 = features3.type(torch.FloatTensor)
            features4 = features4.type(torch.FloatTensor)
            domain = domain.type(torch.LongTensor)
            domain2 = domain2.type(torch.LongTensor)
            domain3 = domain3.type(torch.LongTensor)
            domain4 = domain4.type(torch.LongTensor)
            features2, features3, features4 = Variable(features2), Variable(features3), Variable(
                features4)
            domain, domain2, domain3, domain4 = Variable(domain), Variable(domain2), Variable(
                domain3), Variable(domain4)

            if cuda:
                features = features.cuda()
                features2 = features2.cuda()
                features3 = features3.cuda()
                features4 = features4.cuda()
                domain = domain.cuda()
                domain2 = domain2.cuda()
                domain3 = domain3.cuda()
                domain4 = domain4.cuda()

            features = torch.cat((features, features2, features3, features4), 0)

            domain = domain.view(-1)
            domain = torch.cat((domain, domain2, domain3, domain4), 0)

            optimizer_dm.zero_grad()
            output_dm = domain_predictor(features.detach())
            # print(domain.shape)
            # print(output_dm.shape)
            loss_dm = 10 * domain_criterion(output_dm, domain)  # was 10
            loss_dm.backward()
            optimizer_dm.step()

            # Now update just the encoder using the domain loss
            optimizer_conf.zero_grad()
            output_dm_conf = domain_predictor(features)
            loss_conf = 100 * conf_criterion(output_dm_conf, domain)  # Get rid of the weight for not unsupervised
            loss_conf.backward(retain_graph=False)
            optimizer_conf.step()

            domain_loss += loss_dm
            conf_loss += loss_conf
            regressor_loss += loss

            output_dm_conf = np.argmax(output_dm.detach().cpu().numpy(), axis=1)
            domain_target = domain.detach().cpu().numpy()
            true_domains.append(domain_target)
            pred_domains.append(output_dm_conf)

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss.item()), flush=True)
                print('\t \t Confusion loss = ', loss_conf.item())
                print('\t \t Domain Loss = ', loss_dm.item(), flush=True)
            del target
            del loss
            del features

    av_loss = regressor_loss / batches
    av_loss_copy = np.copy(av_loss.detach().cpu().numpy())

    av_conf = conf_loss / batches
    av_conf_copy = np.copy(av_conf.detach().cpu().numpy())

    av_dom = domain_loss / batches
    av_dm_copy = np.copy(av_dom.detach().cpu().numpy())

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)
    av_acc_copy = np.copy(acc)

    del av_loss
    del acc
    del av_dom

    print('\nTraining set: Average loss: {:.4f}'.format(av_loss_copy, flush=True))
    print('\nTraining set: Average Conf loss: {:.4f}'.format(av_conf_copy, flush=True))
    print('\nTraining set: Average Dom loss: {:.4f}'.format(av_dm_copy, flush=True))

    print('\nTraining set: Average Acc: {:.4f}\n'.format(av_acc_copy, flush=True))

    return av_loss_copy, av_acc_copy, av_dm_copy, av_conf_copy








