import argparse,json,os,time,random,shutil,torch,pytorch_lightning as pl,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_lightning import seed_everything
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel,BertTokenizerFast
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
os.environ["TOKENIZERS_PARALLELISM"] = "true" 

from util.data import SADataset,collate_fn,label2id,id2label
from util.inference import getQuadruple,from_token_span_idx_to_word_char_idx
from util.model import *
from util.evaluate_single_dataset import evaluate_single_dataset

def init_args():
    parser = argparse.ArgumentParser()
    # bert-base-multilingual-uncased
    parser.add_argument('--model',default='bert-base-multilingual-uncased',type=str)
    # CLN+BiAtt / CLN+CNN / CLN+CNN2 / CLN / Init: SELU + meanpooling / NOP: Init without meanpooling
    parser.add_argument('--model_type',default='CLN+BiAtt',type=str)
    # mpqa ds ca eu norec
    parser.add_argument('--dataset',default='eu',type=str)
    parser.add_argument('--dataset_path',default='../Dataset/v1',type=str)
    parser.add_argument("--train_epochs", default=40, type=int)
    parser.add_argument('--batch_size',default=4,type=int) 
    parser.add_argument('--gpu',default='0,',type=str) 
    parser.add_argument('--use_fp16',action='store_true', default=True) 

    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument("--learning_rate", default=2e-3, type=float) 
    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--do_eval", action='store_true', default=False)
    parser.add_argument("--do_statistics", action='store_true', default=False)

    parser.add_argument("--is_biaffine", default=1, type=int)

    # eval
    parser.add_argument('--eval_model_dir', default='',type=str) 
    parser.add_argument('--trainer_logger_version',default=5,type=int) 

    parser.add_argument('--eval_batch_size',default=20,type=int) 
    parser.add_argument('--test_batch_size',default=8,type=int)
    parser.add_argument("--max_len", default=150, type=int) 
    parser.add_argument('--cache',default='/home/hyc/res/cache',type=str)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--warmup", default=0.1, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)

    # model
    parser.add_argument("--lstm_layer_num", default=4, type=int)
    parser.add_argument("--r1_hid_size", default=768, type=int)
    parser.add_argument("--r2_hid_size", default=0, type=int)
    parser.add_argument("--label_num", default=11, type=int)
    parser.add_argument("--dist_emb_size", default=70, type=int)
    parser.add_argument("--reg_emb_size", default=30, type=int)
    parser.add_argument("--bert_hid_size", default=768, type=int)
    parser.add_argument("--conv_hid_size", default=128, type=int)
    parser.add_argument("--biaffine_size", default=512, type=int)
    parser.add_argument("--ffnn_hid_size", default=384, type=int)

    parser.add_argument("--emb_dropout", default=0.5, type=float)
    parser.add_argument("--conv_dropout", default=0.5, type=float)
    parser.add_argument("--out_dropout", default=0.3, type=float)

     # for BiAxialAttention
    parser.add_argument("--num_head", default=2, type=int)

    # for boundary 
    parser.add_argument("--D", default=1, type=int)
    parser.add_argument("--eps", default=0.2, type=float)

    # for cnn2
    parser.add_argument("--cnn_kernel_size", default=3, type=int)
    parser.add_argument("--cnn_depth", default=3, type=int)
    parser.add_argument("--cnn_groups", default=1, type=int) # 1:1 or 2:2 or 0:input_dim

    # for prodictor
    parser.add_argument("--td", default=0.7, type=float)

    args = parser.parse_args()

    args_model = args.model.replace('\\','-') if '\\' in args.model else args.model
    args.output_dir = f'./outputs/{args.dataset}/{args.model_type}-{args.is_biaffine}'
    os.makedirs(args.output_dir,exist_ok=True)

    args.is_multi_gpu = True if args.gpu.count(',')>1 else False
    args.is_multi_language = True if 'm' in args_model else False

    args.dilation = [1,2,3]
    args.weight = [1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4]
    return args

args = init_args()


def get_dataset(tokenizer,type,args):
    data_dir = f'{args.dataset_path}/{args.dataset}/{type}.json'
    return SADataset(tokenizer,data_dir,type,args)

class TableModel(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.save_hyperparameters() 

        self.bert = BertModel.from_pretrained(args.model, cache_dir=args.cache)
        self.tokenizer = BertTokenizerFast.from_pretrained(args.model, cache_dir=args.cache)

        self.dis_embs = nn.Embedding(20, args.dist_emb_size)
        self.reg_embs = nn.Embedding(3, args.reg_emb_size)

        self.lstm = nn.LSTM(args.bert_hid_size, args.bert_hid_size // 2, num_layers=args.lstm_layer_num, batch_first=True, bidirectional=True)

        conv_input_size = args.bert_hid_size + args.dist_emb_size + args.reg_emb_size

        if 'CNN' in args.model_type:
            self.convLayer = ConvolutionLayer(conv_input_size, args.conv_hid_size, args.dilation, args.conv_dropout)
            if args.cnn_groups == 0:
                self.convLayer2 = ConvolutionLayer2(conv_input_size, conv_input_size, args.cnn_kernel_size, args.cnn_depth, conv_input_size)
            else:
                self.convLayer2 = ConvolutionLayer2(conv_input_size, conv_input_size, args.cnn_kernel_size, args.cnn_depth, args.cnn_groups)

        self.dropout = nn.Dropout(args.emb_dropout)
        self.predictor = CoPredictor(args.label_num, args.bert_hid_size, args.biaffine_size,
                                     args.conv_hid_size * len(args.dilation), args.ffnn_hid_size,
                                     args.out_dropout)
        
        self.biaxialAtt = BiAxialAttention(args.num_head, args.bert_hid_size)

        self.cln = LayerNorm(args.bert_hid_size, args.bert_hid_size, conditional=True)

        self.rep1 = Rep_SELU(args.bert_hid_size,args.r1_hid_size)

        if args.model_type == 'Init':
            predict_hid_size = 5 * args.r1_hid_size + args.r2_hid_size + args.dist_emb_size + args.reg_emb_size
        elif args.model_type == 'CLN+CNN':
            predict_hid_size = args.conv_hid_size * len(args.dilation)
        elif args.model_type == 'CLN+BiAtt':
            predict_hid_size = 3 * args.bert_hid_size + args.dist_emb_size + args.reg_emb_size
        else :
            predict_hid_size =  args.r1_hid_size + args.r2_hid_size + args.dist_emb_size + args.reg_emb_size

        self.predictor2 = Predictor(args.label_num,
                                    args.bert_hid_size, args.ffnn_hid_size,
                                    predict_hid_size, args.biaffine_size)

        self.layernorm1 = nn.LayerNorm(args.r1_hid_size)

    def average_reps(self, d, bert_reps):
        final_reps = []
        for idx,(sent, bert_rep) in enumerate(zip(d['sentence'], bert_reps)):
            len_subtokens = max(d['sentence_len']) - d['sentence_len'][idx] + d['token_len'][idx]
            sent_reps = []
            sub_reps = []
            i = 0
            k = 0
            tokens = sent.split(' ')
            for j, rep in enumerate(bert_rep[:len_subtokens]):
                sub_reps.append(rep)

                len_subsubtokens = 0
                if i<len(d['token_range'][idx]):
                    len_subsubtokens = d['token_range'][idx][i][1]-d['token_range'][idx][i][0] + 1
                else:
                    len_subsubtokens = 1

                if k == len_subsubtokens - 1:
                    ave_rep = torch.stack(sub_reps, dim=0).mean(0)
                    sent_reps.append(ave_rep)
                    sub_reps = []
                    i += 1
                    k = 0
                else:
                    k += 1
            sent_reps = torch.stack(sent_reps, dim=0)
            final_reps.append(sent_reps)

        final_reps = torch.stack(final_reps, dim=0)

        return final_reps

    def forward(self, d):

        word_reps = self.bert(input_ids=d['source_ids'], attention_mask=d['source_mask']).last_hidden_state
        word_reps = word_reps[:,1:-1,:] # remove CLS/SEQ rep
        word_reps = self.average_reps(d,word_reps)
        packed_embs = pack_padded_sequence(word_reps, d['sentence_len'], batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.lstm(packed_embs)
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=max(d['sentence_len']))

        if args.model_type == 'Init' or  args.model_type == 'NOP':
            rep1 = self.rep1(word_reps)
            rep1 = self.layernorm1(rep1)
        elif 'CLN' in args.model_type:
            rep1 = self.cln(word_reps.unsqueeze(2), word_reps)

        if args.model_type == 'Init':
            rep_row_1 = torch.mean(rep1,dim=2,keepdim=True).expand_as(rep1)
            rep_col_1 = torch.mean(rep1,dim=1,keepdim=True).expand_as(rep1)

            rep2 = rep1.permute((0,2,1,3))
            rep_row_2 = torch.mean(rep2,dim=2,keepdim=True).expand_as(rep1)
            rep_col_2 = torch.mean(rep2,dim=1,keepdim=True).expand_as(rep1)

        dis_emb = self.dis_embs(d['dis_inputs'])
        reg_emb = self.reg_embs(d['reg_inputs'])

        if args.model_type == 'Init':
            inputs = torch.cat([rep1, dis_emb, reg_emb, rep_row_1, rep_row_2, rep_col_1, rep_col_2], dim=-1)
        elif args.model_type == 'CLN+BiAtt':
            inputs = rep1
        else:
            inputs = torch.cat([rep1, dis_emb, reg_emb], dim=-1)
        
        if args.model_type == 'CLN+CNN':
            conv_inputs = torch.masked_fill(inputs, d['reg_inputs'].eq(0).unsqueeze(-1), 0.0)
            conv_outputs = self.convLayer(conv_inputs)
            inputs = torch.masked_fill(conv_outputs, d['reg_inputs'].eq(0).unsqueeze(-1), 0.0)
        elif args.model_type == 'CLN+CNN2':
            conv_inputs = torch.masked_fill(inputs, d['reg_inputs'].eq(0).unsqueeze(-1), 0.0)
            conv_outputs = self.convLayer2(conv_inputs)
            inputs = torch.masked_fill(conv_outputs, d['reg_inputs'].eq(0).unsqueeze(-1), 0.0)
        elif args.model_type == 'CLN+BiAtt':
            inputs = torch.masked_fill(inputs, d['reg_inputs'].eq(0).unsqueeze(-1), 0.0)
            inputs = self.biaxialAtt(inputs)
            inputs = torch.masked_fill(inputs, d['reg_inputs'].eq(0).unsqueeze(-1), 0.0)
            inputs = torch.cat([inputs, dis_emb, reg_emb], dim=-1)

        outputs = self.predictor2(word_reps,inputs,is_biaffine = (args.is_biaffine==1), td=args.td)

        return outputs

    def _step(self, batch):

        outputs = self(batch)

        labels = batch["label"]

        weight = torch.tensor(args.weight).float().to(outputs.device)

        loss = F.cross_entropy(outputs.reshape([-1,outputs.shape[-1]]),labels.reshape([-1]),weight=weight)

        return loss

    def on_fit_start(self):
        seed_everything(args.seed,workers=True)


    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 sync_dist = self.args.is_multi_gpu, batch_size=args.batch_size)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_train_loss,
                 sync_dist = self.args.is_multi_gpu, batch_size=args.batch_size)

    def validation_step(self, batch, batch_idx):
        return 0

    def validation_epoch_end(self, outputs):

        dev_res = test(self,args.logger_version,'dev')
        dev_f1 = dev_res['sentiment_tuple/f1']
        self.log("dev_f1", dev_f1,
                  sync_dist = args.is_multi_gpu, batch_size=args.batch_size)

        with open(f"{args.output_dir}/lightning_logs/version_{args.logger_version}/result.txt",'a+') as f:
            f.write(f'{self.trainer.current_epoch}-dev_res:{dev_res} \n')

    def configure_optimizers(self):

        bert_params = set(self.bert.parameters())
        other_params = list(set(self.parameters()) - bert_params)

        for param in self.bert.parameters(): # frozen BERT params
            param.requires_grad = False

        params = [
            {'params': other_params,
             'lr': args.learning_rate,
             'weight_decay': args.weight_decay},
        ]

        optimizer = AdamW(params, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        t_total = ((len(self.train_dataset) // (self.args.batch_size))
                   // self.args.gradient_accumulation_steps
                   * float(self.args.train_epochs))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = self.args.warmup * t_total,
            num_training_steps=t_total,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer],[scheduler]

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def setup(self, stage=None):
        self.train_dataset = get_dataset(self.tokenizer, "train",self.args)
        self.val_dataset = get_dataset(self.tokenizer,"dev",self.args)
        self.test_dataset = get_dataset(self.tokenizer,"test",self.args)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, collate_fn=collate_fn,
                          drop_last=False, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.eval_batch_size, collate_fn=collate_fn,
                          drop_last=False, shuffle=False, num_workers=4, pin_memory=True)


def check_available(d):
    if len(d['Polar_expression'][0])>0 and len(d['Source'][0])==0 and len(d['Target'][0])==0 and args.dataset not in ['eu','ca','norec']:
        return False
    return True


def test(module,trainer_logger_version,type,load_model_dir=None):
    device = f"cuda:{args.gpu.split(',')[0]}"

    if load_model_dir!=None:
        module = module.load_from_checkpoint(load_model_dir,args=args)
        module.to(device)

    module.eval()

    tokenizer = module.tokenizer
    test_dataset = get_dataset(tokenizer,type,args)
    data_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=4,collate_fn=collate_fn, pin_memory=True)

    text_list,result_json=[],[]

    for i,batch in enumerate(data_loader):
        for k in batch.keys():
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device)

        with torch.no_grad():
            outs = module(batch)

        preds = torch.max(outs,dim=-1)[1] # b*n*n
        for j in range(preds.shape[0]):

            token_len = batch['sentence_len'][j]
            preds_j = preds[j,:token_len,:token_len]
            e_res = getQuadruple(preds_j,args)

            result_ = {}
            result_['sent_id'] = batch['text'][j][0]
            result_['text'] = batch['text'][j][1]
            result_['opinions'] = []

            d = {}
            for ei in e_res:
                d['Source'],d['Target'],d['Polar_expression'],d['Polarity']=[[],[]],[[],[]],[[],[]],'Neutral'
                for h_span in ei['h']:
                    idx_l,idx_r = h_span
                    l,r = batch['token_offset'][j][idx_l][0], batch['token_offset'][j][idx_r][1]
                    # l,r = from_token_span_idx_to_word_char_idx(batch['token_range'][j],batch['token_offset'][j],h_span)
                    d['Source'][0].append(batch['text'][j][1][l:r])
                    d['Source'][1].append(f'{l}:{r}')
                for t_span in ei['t']:
                    idx_l,idx_r = t_span
                    l,r = batch['token_offset'][j][idx_l][0], batch['token_offset'][j][idx_r][1]
                    # l,r = from_token_span_idx_to_word_char_idx(batch['token_range'][j],batch['token_offset'][j],t_span)
                    d['Target'][0].append(batch['text'][j][1][l:r])
                    d['Target'][1].append(f'{l}:{r}')
                for e_span in ei['e']:
                    idx_l,idx_r = e_span
                    l,r = batch['token_offset'][j][idx_l][0], batch['token_offset'][j][idx_r][1]
                    # l,r = from_token_span_idx_to_word_char_idx(batch['token_range'][j],batch['token_offset'][j],e_span)
                    d['Polar_expression'][0].append(batch['text'][j][1][l:r])
                    d['Polar_expression'][1].append(f'{l}:{r}')

                if 'POS' in ei['p']:
                    d['Polarity'] = 'Positive'
                elif 'NEG' in ei['p']:
                    d['Polarity'] = 'Negative'

                if check_available(d):
                    result_['opinions'].append(d)

                d = {}

            result_json.append(result_)

        text_list.extend(batch['text'])

    time_now_str = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    with open(f"{args.output_dir}/lightning_logs/version_{trainer_logger_version}/predict-{time_now_str}-{type}.json","w") as f:
        json.dump(result_json,f)

    result = evaluate_single_dataset(
        gold_file = f"{args.dataset_path}/{args.dataset}/{type}.json",
        pred_file = f"{args.output_dir}/lightning_logs/version_{trainer_logger_version}/predict-{time_now_str}-{type}.json"
    )
    f1 = result['sentiment_tuple/f1']

    try:
        os.rename(f"{args.output_dir}/lightning_logs/version_{trainer_logger_version}/predict-{time_now_str}-{type}.json",f"{args.output_dir}/lightning_logs/version_{trainer_logger_version}/{module.trainer.current_epoch}-predict-{type}-{f1:.5f}.json")
    except:
        os.rename(f"{args.output_dir}/lightning_logs/version_{trainer_logger_version}/predict-{time_now_str}-{type}.json",f"{args.output_dir}/lightning_logs/version_{trainer_logger_version}/predict-{time_now_str}-{type}-{f1:.5f}.json")

    return result


if __name__ == '__main__':

    if args.do_train:
        print("\n****** Conduct Training ******")

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='dev_f1', mode='max', save_top_k=5,
            filename='{epoch}-{test_f1:.5f}',
        )

        train_params = dict(
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            accelerator='gpu',
            devices=args.gpu,
            gradient_clip_val=1.0,
            max_epochs=args.train_epochs,
            callbacks=[checkpoint_callback],
            strategy = "ddp" if args.is_multi_gpu else None,
            precision=16 if args.use_fp16 else 32,
            num_sanity_val_steps=0,
            deterministic=True
            # logger=[], # mulit gpu need
        )
        trainer = pl.Trainer(**train_params)

        args.logger_version = trainer.logger.version
        print(args)
        model = TableModel(args)

        seed_everything(args.seed,workers=True)

        trainer.fit(model)


    if args.do_eval:
        print("\n****** Conduct Testing ******")
        trainer_logger_version = args.trainer_logger_version
        if args.do_train:
            trainer_logger_version = trainer.logger.version
            del trainer
            del model
            g = os.walk(f'{args.output_dir}/lightning_logs/version_{trainer_logger_version}/checkpoints')
            dif,min_key = {},1e5
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if '.ckpt' in file_name:
                        key_loss = float(file_name[-10:-5]) 
                        dif[key_loss] = file_name
                        min_key = min(min_key,key_loss)
            load_model_dir = f'{args.output_dir}/lightning_logs/version_{trainer_logger_version}/checkpoints/{dif[min_key]}'
        else:
            load_model_dir = args.eval_model_dir
        print(f'load_model_dir:{load_model_dir}')

        module = TableModel(args)
        result = test(module,trainer_logger_version,'test',load_model_dir)

        if args.do_train:
            load_model_dir = f'{args.output_dir}/lightning_logs/version_{trainer_logger_version}/epoch_last.ckpt'
            test(module,trainer_logger_version,'test',load_model_dir)













