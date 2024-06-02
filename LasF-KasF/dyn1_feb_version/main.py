
import os
import json
import torch
from torch import nn, Tensor

from transformers import GPT2Tokenizer, GPT2Model
from torchtext.datasets import WikiText2, WikiText103, PennTreebank

import time
from datetime import datetime

import random

import dyn1_feb.cortical_column as cortical_column
import dyn1_feb.dyn_LM as dyn_LM
import dyn1_feb.load_corpus as load_corpus
import dyn1_feb.neuronal_broadcast as NB
import dyn1_feb.train_utils as train_utils


SEED_ID = 42
torch.manual_seed(SEED_ID)
#random.seed(SEED_ID)

rand_id = str(round(random.random()*100000))
print('rand_id:', rand_id)

print('SEED ID:', SEED_ID)

#bptt, batch_size, eval_batch_size = 1000, 2, 2
bptt, batch_size, eval_batch_size = 100, 16, 32

emsize, dyn_Hdim = 1024, 768
NL_signal, NLS_use_act, LR_ratio = 0, 0, 5
p_norm, q_dim, num_D, num_G = 2, 15, 2, 15

dyn1_dropout, PE_dropout = 0.1, 0.1
lr, schedule_step, schedule_gamma = 0.28, 2, 0.9

cuda_id = 5

device = torch.device('cuda:'+str(cuda_id) if torch.cuda.is_available() else 'cpu')
print('------ Device:', device)

w103_repeat, w2_repeat, ptb_repeat = 5, 10, 10

saved_dir = 'dyn_params/model_cudaID'+str(cuda_id)+'_randID'+rand_id+'/'
os.mkdir(saved_dir)



GPT2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
ntokens = len(GPT2_tokenizer)

print('--- actual vocab_size:', ntokens)

model = dyn_LM.LM_Model(bptt, emsize, dyn_Hdim, NL_signal, NLS_use_act, LR_ratio, p_norm, q_dim, \
                        num_D, num_G, ntokens, dyn1_dropout, PE_dropout, device).to(device)
all_params = sum([param.numel() for param in model.parameters()])
print('No.Params of Model:',  all_params)


config_dict = {
    'rand_id': rand_id, 'SEED_ID': SEED_ID, 
    'bptt': bptt, 'batch_size': batch_size, 'eval_batch_size': eval_batch_size,
    'emsize': emsize, 'dyn_Hdim': dyn_Hdim, 
    'NL_signal': NL_signal, 'NLS_use_act': NLS_use_act, 'LR_ratio': LR_ratio,
    'p_norm': p_norm, 'q_dim': q_dim, 'num_D': num_D, 'num_G': num_G,
    'dyn1_dropout': dyn1_dropout, 'PE_dropout': PE_dropout,
    'lr': lr, 'schedule_step': schedule_step, 'schedule_gamma': schedule_gamma,
    'w103_repeat': w103_repeat, 'w2_repeat': w2_repeat, 'ptb_repeat': ptb_repeat,
    'No.Params': all_params,
}

json_object = json.dumps(config_dict, indent=4)
with open(saved_dir+"config.json", "w") as outfile: outfile.write(json_object)


optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_step, gamma=schedule_gamma)

print(datetime.now().time())
w2_train_iter, w2_val_iter, w2_test_iter = WikiText2()

w2_train_data = load_corpus.data_process(w2_train_iter, GPT2_tokenizer)
print('w2_train:', datetime.now().time(), w2_train_data.shape)
w2_val_data = load_corpus.data_process(w2_val_iter, GPT2_tokenizer)
print('w2_val:', datetime.now().time(), w2_val_data.shape)
w2_test_data = load_corpus.data_process(w2_test_iter, GPT2_tokenizer)
print('w2_test', datetime.now().time(), w2_test_data.shape)

w2_val_data = load_corpus.batchify(w2_val_data, eval_batch_size, device)
print('w2_val_batch:', datetime.now().time(), w2_val_data.shape)
w2_test_data = load_corpus.batchify(w2_test_data, eval_batch_size, device)
print('w2_test_batch:', datetime.now().time(), w2_test_data.shape)


w103_train_iter, w103_val_iter, w103_test_iter = WikiText103()

print(datetime.now().time())
w103_train_data = load_corpus.data_process(w103_train_iter, GPT2_tokenizer)
print('w103_train:', datetime.now().time(), w103_train_data.shape)
w103_val_data = load_corpus.data_process(w103_val_iter, GPT2_tokenizer)
print('w103_val:', datetime.now().time(), w103_val_data.shape)
w103_test_data = load_corpus.data_process(w103_test_iter, GPT2_tokenizer)
print('w103_test:', datetime.now().time(), w103_test_data.shape)

w103_val_data = load_corpus.batchify(w103_val_data, eval_batch_size, device)
print('w103_val_batch:', datetime.now().time(), w103_val_data.shape)
w103_test_data = load_corpus.batchify(w103_test_data, eval_batch_size, device)
print('w103_test_batch:', datetime.now().time(), w103_test_data.shape)



PTB_train_iter, PTB_val_iter, PTB_test_iter = PennTreebank()

PTB_train_data = load_corpus.data_process(PTB_train_iter, GPT2_tokenizer)
print('ptb_train:', datetime.now().time(), PTB_train_data.shape)
PTB_val_data = load_corpus.data_process(PTB_val_iter, GPT2_tokenizer)
print('ptb_val:', datetime.now().time(), PTB_val_data.shape)
PTB_test_data = load_corpus.data_process(PTB_test_iter, GPT2_tokenizer)
print('ptb_test:', datetime.now().time(), PTB_test_data.shape)

PTB_val_data = load_corpus.batchify(PTB_val_data, eval_batch_size, device)
print('ptb_val_batch:', datetime.now().time(), PTB_val_data.shape)
PTB_test_data = load_corpus.batchify(PTB_test_data, eval_batch_size, device)
print('ptb_test_batch:', datetime.now().time(), PTB_test_data.shape)


#train_data = load_corpus.batchify(w2_train_data, batch_size, device)

train_data = torch.cat([w2_train_data]*max(w2_repeat-1, 0) + [PTB_train_data]*max(ptb_repeat-1,0) + [w103_train_data]*max(w103_repeat-1,0) + 
                                            [PTB_train_data]*min(1, ptb_repeat) + [w103_train_data]*min(1,w103_repeat) + [w2_train_data]*min(1,w2_repeat))

train_data = load_corpus.batchify(train_data, batch_size, device)

val_datas = [w2_val_data, w103_val_data, PTB_val_data]



criterion = nn.CrossEntropyLoss()

best_val_PPL, best_val_lPPL = float('inf'), float('inf')
epochs = 50
log_interval = 80000

print(datetime.now().time())


for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_utils.train(model, train_data, val_datas, log_interval, bptt, device, ntokens, optimizer, epoch, scheduler)

    w2_val_res = train_utils.evaluate(model, w2_val_data, bptt, device, ntokens)
    w103_val_res = train_utils.evaluate(model, w103_val_data, bptt, device, ntokens)
    ptb_val_res = train_utils.evaluate(model, PTB_val_data, bptt, device, ntokens)

    print('--- WikiText103_val:', w103_val_res)
    print('--- WikiText2_val:', w2_val_res)
    print('--- PennTreebank_val:', ptb_val_res)

    print('-' * 89)
    

    if w103_val_res['distmap']['PPL'] < best_val_PPL or w103_val_res['distmap']['lPPL'] < best_val_lPPL:

        if w103_val_res['distmap']['PPL'] < best_val_PPL: best_val_PPL = w103_val_res['distmap']['PPL']
        if w103_val_res['distmap']['lPPL'] < best_val_lPPL: best_val_lPPL = w103_val_res['distmap']['lPPL']

        w103_test_res = train_utils.evaluate(model, w103_test_data, bptt, device, ntokens)
        w2_test_res = train_utils.evaluate(model, w2_test_data, bptt, device, ntokens)
        ptb_test_res = train_utils.evaluate(model, PTB_test_data, bptt, device, ntokens)

        print(f'| End of Epoch {epoch:3d}', '--- time:', datetime.now().time())
        print('--- WikiText103_test:', w103_test_res)
        print('--- WikiText2_test:', w2_test_res)
        print('--- PennTreebank_test:', ptb_test_res)

        print('-' * 89)

        torch.save(model.state_dict(), \
        saved_dir+'cID'+str(cuda_id)+'_rID'+rand_id+'_Ep'+str(epoch)+\
        '_W103PPL'+str(round(w103_test_res['fcn']['PPL'],1))+'_W103lPPL'+str(round(w103_test_res['distmap']['lPPL'],1))+\
        '_PTBPPL'+str(round(ptb_test_res['fcn']['PPL'],1))+'_PTBlPPL'+str(round(ptb_test_res['distmap']['lPPL'],1))+\
        '.dynLM')

    scheduler.step()





