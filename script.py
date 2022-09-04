
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import random_split
from Topic_model import *
from utils import *


if __name__ == '__main__':
    class_name = 'senwave_preprocess_version2'
    if class_name == 'senwave_preprocess_version3':
        output_channels = 11
        index = range(4,15)
    elif class_name == 'senwave_preprocess_version2':
        output_channels = 3
        index = range(15,18) # 训练集列
    else:
        print('There is no file called : {}'.format(class_name))

    # version3 for 11 labels, version2 for 3 classes
    df = pd.read_csv(class_name + '.csv')
    texts = df['Tweet']
    texts = texts.dropna()
    texts.isna().sum()
    common_texts = []
    for text in texts:
        tmp = text.split(' ')
        common_texts.append(list(filter(None, tmp)))

    texts = texts.values.tolist()
    ldabert_Model=Topic(texts, common_texts, class_name, method='LDA_BERT', k=3) #三类
    vec = ldabert_Model.vectorize(method='BERT')

    print('shape of representation is {}'.format(vec.shape))
    dataset = Data_training(df,vec,index)
    print(df.shape)
    train_dataset, valid_dataset = random_split(dataset, [9000, len(dataset)-9000], generator=torch.Generator().manual_seed(42))
    print('length of trainset is {}'.format(len(train_dataset)))
    print('length of validset is {}'.format(len(valid_dataset)))

    #set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Hyperparameters
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 100

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)
    # data,target = next(iter(train_loader))


    # Initialize network
    input_channels = vec.shape[1]
    model = Dense(input_channels,output_channels).to(device)

    # Loss and optimizer
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Network
    for epoch in range(num_epochs):
        print('--------The {}th epoch begins--------'.format(epoch + 1))
        model.train()
        loss_sum = 0
        step = 0
        for batch_idx, (data, targets) in enumerate(train_loader):

            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)


            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
            loss_sum += loss.item()
            step += 1
        print('The loss is:{}'.format(loss_sum/step))

        model.eval()
        with torch.no_grad():
            acc = 0
            for source, label in valid_loader:
                source = source.to(device=device)
                pred = nn.Softmax(dim=1)(model(source))
                count = sum(label[0])
                _,indices = pred.topk(int(count.item()), dim=1, largest = True)
                pred_binary = torch.FloatTensor(output_channels).fill_(0)
                pred_binary[indices] = 1
                if sum(pred_binary == label[0]).item() == output_channels:
                    acc += 1
                else:
                    acc += 0

            print('The accuracy of validation set is: {}'.format(acc/len(valid_dataset)))









