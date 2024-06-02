
import math
import torch
from torch import nn, Tensor
import dyn1_feb.load_corpus as load_corpus
from datetime import datetime

import time

criterion = nn.CrossEntropyLoss()

def train(model, train_data, val_datas, log_interval, bptt, device, ntokens, optimizer, epoch, scheduler):
    model.train()  # turn on train mode
    total_loss = 0.
    start_time = time.time()

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):

        if batch != num_batches:
            data, targets = load_corpus.get_batch(train_data, i, bptt)
            src_mask = model.generate_square_subsequent_mask(data.shape[0], device)

            output_dict = model(data, src_mask)

            output_distmap = output_dict['distmap']['output']
            GLBInd_distmap = output_dict['distmap']['global_indicator']

            output_fcn = output_dict['fcn']['output']
            GLBInd_fcn = output_dict['fcn']['global_indicator']

            output_distmap_flat = output_distmap.view(-1, ntokens)
            loss_distmap = criterion(output_distmap_flat, targets) + GLBInd_distmap

            output_fcn_flat = output_fcn.view(-1, ntokens)
            loss_fcn = criterion(output_fcn_flat, targets) + GLBInd_fcn

            loss = loss_distmap + loss_fcn

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()

            total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:

            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval

            w2_val_res = evaluate(model, val_datas[0], bptt, device, ntokens)
            w103_val_res = evaluate(model, val_datas[1], bptt, device, ntokens)
            ptb_val_res = evaluate(model, val_datas[2], bptt, device, ntokens)

            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)

            print(f'| Epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} |', 'Time:', datetime.now().time(),
                  '--- WikiText103_val:', w103_val_res,
                  '--- WikiText2_val:', w2_val_res,
                  '--- PennTreebank_val:', ptb_val_res)

            total_loss = 0
            start_time = time.time()


def evaluate(model, eval_data, bptt, device, ntokens):
    model.eval()  # turn on evaluation mode
    datapiece_total = 0
    distmap_total_loss, fcn_total_loss = 0, 0
    _distmap_total_loss, _fcn_total_loss = 0, 0
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            try:
                data, targets = load_corpus.get_batch(eval_data, i, bptt)
                seq_len = data.size(0)
                src_mask = model.generate_square_subsequent_mask(seq_len, device)

                output_dict = model(data, src_mask)

                output_distmap = output_dict['distmap']['output']
                GLBInd_distmap = output_dict['distmap']['global_indicator']

                output_fcn = output_dict['fcn']['output']
                GLBInd_fcn = output_dict['fcn']['global_indicator']

                output_distmap_flat = output_distmap.view(-1, ntokens)
                output_fcn_flat = output_fcn.view(-1, ntokens)

                distmap_total_loss += seq_len * criterion(output_distmap_flat, targets).item()
                fcn_total_loss += seq_len * criterion(output_fcn_flat, targets).item()

                _distmap_total_loss += criterion(output_distmap_flat[-1], targets[-1]).item()
                _fcn_total_loss += criterion(output_fcn_flat[-1], targets[-1]).item()

                datapiece_total += 1

            except:
                continue


    res_dict = {
        'distmap': {
            'PPL': round(math.exp(min(distmap_total_loss/(len(eval_data) - 1), 10)), 3),
            'lPPL': round(math.exp(min(_distmap_total_loss/datapiece_total, 10)), 3)
        },
        'fcn': {
            'PPL': round(math.exp(min(fcn_total_loss/(len(eval_data) - 1), 10)), 3),
            'lPPL': round(math.exp(min(_fcn_total_loss/datapiece_total, 10)), 3)
        },
    }

    return res_dict