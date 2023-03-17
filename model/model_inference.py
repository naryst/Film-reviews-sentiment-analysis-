import re
import string

import torch
import torch.nn
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from transformers import DistilBertModel
from transformers import DistilBertTokenizerFast

stop = stopwords.words('english')
punctuation = list(string.punctuation)
stop += punctuation


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


# Removing the stopwords from text
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop and i.strip().lower().isalpha():
            final_text.append(i.strip().lower())
    return " ".join(final_text)


# Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    return text


MAX_LENGTH = 256


def batch_encode(tokenizer, texts, batch_size=16, max_length=MAX_LENGTH):
    input_ids = []
    attention_mask = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer.batch_encode_plus(batch,
                                             max_length=max_length,
                                             padding='max_length',
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_token_type_ids=False
                                             )
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])

    return torch.tensor(input_ids), torch.tensor(attention_mask)


class TorchModel(torch.nn.Module):
    def __init__(self, transformer):
        super(TorchModel, self).__init__()
        self.fc1 = torch.nn.Linear(768, 1)
        self.transformer = transformer
        self.sigmoid = torch.sigmoid

    def forward(self, x):
        ids = x[0]
        attn = x[1]
        last_hidden_state = self.transformer(ids, attention_mask=attn)[0]
        cls_token = last_hidden_state[:, 0, :]
        x = self.fc1(cls_token)
        x = self.sigmoid(x)
        return x


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
score_quantile = [0.32044193,
                  0.3692736,
                  0.40665343,
                  0.44061545,
                  0.4739672,
                  0.5081665,
                  0.5448841,
                  0.5870045,
                  0.6409643]


def get_score(pred):
    for i, score in enumerate(score_quantile):
        if pred < score:
            return i + 1
    return 10


def load_model():
    model = TorchModel(DistilBertModel.from_pretrained('distilbert-base-uncased'))
    model.load_state_dict(torch.load('torch_model.pt', map_location=torch.device('cpu')))
    model.eval()
    return model


def predict(model, text):
    text = denoise_text(text)
    ids, attn = batch_encode(tokenizer, [text])
    with torch.no_grad():
        pred = model((ids, attn))
    return get_score(pred.item())


if __name__ == '__main__':
    model = load_model()
    text = 'I like this product'
    print(predict(model, text))
