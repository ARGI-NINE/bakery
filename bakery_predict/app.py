import io
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
import time

from bottle import Bottle, request, run, response, static_file


import warnings
warnings.filterwarnings("ignore")

class LSTM_Regression(nn.Module):
    """
        使用LSTM进行回归
        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x


PATH = "./model0.pth"
model = torch.load(PATH)


def create_dataset(dataset, look_back = 20, num = 5):
    
    predict = []
    data = dataset.tolist()
    start_idx = len(data) - look_back
    for i in range(start_idx, start_idx + num):
        a = data[i : (i + look_back)]

        a = np.array(a, dtype='float32')
        a = np.reshape(a, (a.shape[0], a.shape[1], 1))
        a = a.reshape(-1, 1, look_back)

        pred_y = model(torch.from_numpy(a))
        pred_y = pred_y.view(-1).data.numpy()

        pred = 20 * (pred_y - np.average(pred_y)) + pred_y
        predict.append(pred)

        data.append(pred)
    return predict
    
def handleData(fileData):
    df = pd.read_csv(fileData)
    df = df[:2419] #2419total 2405No MAY
    df.replace(np.nan, 0, inplace = True)
    df_sales = df[['total']]
    data = df_sales['total'].values.astype('float32')
    data = data[:, np.newaxis]

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data)
    train_size = int(len(data) * 0.8)
    testlist = dataset[train_size:]

    predict = create_dataset(testlist)

    for i in range(len(predict)):
        if(predict[i] < 0):
            predict[i] = 0
    predict = pd.DataFrame(predict)
    out = f'{int(time.time())}.csv'
    predict.to_csv('files/'+out)
    return out

def result(code, data):
    return json.dumps({
        'code': code,
        'data': data
    })


app = Bottle()


@app.route('/upload', method='POST')
def upload():
    response.content_type = 'application/json'
    upload_file = request.files.get('file')

    if upload_file:
        data = upload_file.file.read()
        data = io.BytesIO(data)
        try:
            res = handleData(data)
            return result(200, res)
        except Exception as e:
            return result(500, str(e))
    else:
        return result(400, 'file not found')

@app.route('/download/<filename:path>')
def download_file(filename):
    return static_file(filename, root='files', download=filename)

if __name__ == '__main__':
    run(app, host='localhost', port=8000)
