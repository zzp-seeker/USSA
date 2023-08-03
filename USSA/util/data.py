import json,nltk,copy,torch,numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

label2id = {'N':0,'e-NW':1,'h-NW':2,'t-NW':3,'H-S':4,'H-E':5, 'T-S':6,'T-E':7, 'e-NEU':8,'e-POS':9,'e-NEG':10,'H-SE':11,'T-SE':12}
id2label = {v:k for k,v in label2id.items()}
tk = nltk.tokenize.simple.SpaceTokenizer()

def convert_char_offsets_to_token_idxs(char_offsets, token_offsets,text):
    """
    >>> text = "I think the new uni ( ) is a great idea"
    >>> char_offsets = ["8:19","20:26"]
    >>> token_offsets = [(0,1), (2,7), (8,11), (12,15), (16,19), (20,21), (22,23), (24,26), (27,28), (29,34), (35,39)]
    >>> convert_char_offsets_to_token_idxs(char_offsets, token_offsets)
    >>> [(2,7)] # [(2,4),(5,7)] 
    """
    if len(char_offsets)==0: return []
    token_idxs = []
    for char_offset in char_offsets:
        sidx, eidx = char_offset.split(":")
        sidx, eidx = int(sidx), int(eidx)
        s_result,e_result,s_flag,e_flag = -1,-1,False,False
        for i, (s, e) in enumerate(token_offsets):
            if s <= sidx <e and not s_flag:
                s_flag,s_result = True,i
            if s < eidx <=e and not e_flag:
                e_flag,e_result = True,i
            if s_flag and e_flag:
                token_idxs.append((s_result,e_result))
                break
    if token_idxs == []: return []


    token_idxs.sort(key=lambda k:(k[0],k[1]))
    token_idxs_r = []
    s_to_add,e_to_add = token_idxs[0][0],token_idxs[0][1]

    for i,(s,e) in enumerate(token_idxs[1:]):
        if s <= e_to_add+1:
            e_to_add = e
        else:
            token_idxs_r.append((s_to_add,e_to_add))
            s_to_add,e_to_add = token_idxs[i][0],token_idxs[i][1]

    token_idxs_r.append((s_to_add,e_to_add))
    return token_idxs

# se_l:[(1,1),(5,10)]  return: [1,5,6,7,8,9,10]
def convert_se_to_list(se_l):
    r = []
    for i,(s,e) in enumerate(se_l):
        r.extend(list(range(s,e+1)))
    return r

def cal_null_e(d):
    num = 0
    for o in d['opinions']:
        if len(o[2])==0: # h,t,e,p
            num += 1
    return num

def cal_single_e(d):
    num = 0
    for o in d['opinions']:
        if len(o[2])!=0 and len(o[1])==0 and len(o[0])==0:
            num += 1
    return num

def check_htep_all_used(d):
    for o in d['opinions']:
        if len(o[2])!=0 and len(o[1])!=0 and len(o[0])!=0:
            return True
    return False

def cal_entity_and_overlap(d):
    r = [[],[],[]] # h,t,e
    for o in d['opinions']: # (h,t,e,p)
        l0 = convert_se_to_list(o[0]) # h
        if len(l0)>0:
            r[0].append(l0)
        l1 = convert_se_to_list(o[1]) # t
        if len(l1)>0:
            r[1].append(l1)
        l2 = convert_se_to_list(o[2]) # e
        if len(l2)>0:
            r[2].append(l2)

    r_overlap = [[0]*len(r[0]),[0]*len(r[1]),[0]*len(r[2])]
    for idx in range(3):
        for i in range(len(r[idx])):
            for j in range(len(r[idx])):
                if j>i:
                    if set(r[idx][i])!=set(r[idx][j]) and len(set(r[idx][i]) & set(r[idx][j]))!=0:
                    # if len(set(r[idx][i]) & set(r[idx][j]))!=0:
                        r_overlap[idx][i],r_overlap[idx][j] = 1,1

    # h,t,e,overlap_h,overlap_t,overlap_e
    return len(r[0]),len(r[1]),len(r[2]),sum(r_overlap[0]),sum(r_overlap[1]),sum(r_overlap[2])

def cal_overlap_tuple(d):
    r = [[],[],[]] # h,t,e
    for o in d['opinions']: # (h,t,e,p)
        r[0].append(convert_se_to_list(o[0]))
        r[1].append(convert_se_to_list(o[1]))
        r[2].append(convert_se_to_list(o[2]))

    r_overlap = [[0]*len(d['opinions']) for _ in range(3)]
    for idx in range(3):
        for i in range(len(r[idx])):
            for j in range(len(r[idx])):
                if j>i:
                    if set(r[idx][i])!=set(r[idx][j]) and len(set(r[idx][i]) & set(r[idx][j]))!=0:
                        r_overlap[idx][i],r_overlap[idx][j] = 1,1

    # r_overlap [[0,0],[1,1],[0,0]] max=> [0,1,0] sum=> 1
    return sum(torch.max(torch.tensor(r_overlap,dtype=torch.int),dim=0).values.tolist())

def cal_discontinuous(d):
    res = [0,0,0] 

    for idx in range(3):
        for o in d['opinions']: # (h,t,e,p)
            to_check = convert_se_to_list(o[idx])
            for j in range(len(to_check)):
                if j!=0 and to_check[j]!=to_check[j-1]+1:
                    res[idx] += 1
                    break

    # discontinuous_h,discontinuous_t,discontinuous_e
    return res[0],res[1],res[2]

def cal_discontinuous_tuple(d):
    res = 0
    for o in d['opinions']:
        flag = 0
        for idx in range(3):
            if flag == 0:
                to_check = convert_se_to_list(o[idx])
                for j in range(len(to_check)):
                    if j!=0 and to_check[j]!=to_check[j-1]+1:
                        flag = 1
                        break
        res += flag
    return res


def get_data_from_file(path,tokenizer,type,args):
    data_post = []
    with open(path) as file:
        data = json.load(file)

        sentence_num,has_opinions_sentence_num,exceed_max_sentence_len = 0,0,0
        tuple_num, single_e_tuple_num, null_e_tuple_num, overlap_tuple_num, discontinuous_tuple_num = 0,0,0,0,0
        e_num,h_num,t_num = 0,0,0
        overlap_e_num,overlap_h_num,overlap_t_num = 0,0,0
        discontinuous_e_num,discontinuous_h_num,discontinuous_t_num = 0,0,0

        for data_i in data:
            d = {}
            d['sent_id']=data_i['sent_id']
            d['text']=data_i['text']
            d['opinions_pre']=data_i['opinions']
            d['opinions'],f=[],{}
            d['sentence_len'] = len(d['text'].strip().split())

            token_offsets = list(tk.span_tokenize(d['text']))
            d['token_offset'] = token_offsets

            for o in data_i['opinions']:
                holder_char_idxs = o["Source"][1]
                target_char_idxs = o["Target"][1]
                exp_char_idxs = o["Polar_expression"][1]

                h = convert_char_offsets_to_token_idxs(holder_char_idxs, token_offsets,d['text'])
                t = convert_char_offsets_to_token_idxs(target_char_idxs, token_offsets,d['text'])
                e = convert_char_offsets_to_token_idxs(exp_char_idxs, token_offsets,d['text'])
                p = o["Polarity"] if o["Polarity"]!=None else 'Neutral' 
                d['opinions'].append((h, t, e, p))

            d['token_range'] = []
            token_start = 0

            is_discard = False
            for i, w, in enumerate(d['text'].strip().split()):
                token_end = token_start + len(tokenizer.encode(w, add_special_tokens=False))
                if token_end == token_start:
                    is_discard = True 

                d['token_range'].append([token_start, token_end-1])
                token_start = token_end
            assert len(d['token_range']) == d['sentence_len']

            d['tokenized_opinions'] = []
            for o in d['opinions']:
                h,t,e,p = [],[],[],o[-1]
                for h_s,h_e in o[0]:
                    h.append((d['token_range'][h_s][0],d['token_range'][h_e][1]))
                for t_s,t_e in o[1]:
                    t.append((d['token_range'][t_s][0],d['token_range'][t_e][1]))
                for e_s,e_e in o[2]:
                    e.append((d['token_range'][e_s][0],d['token_range'][e_e][1]))
                d['tokenized_opinions'].append((h,t,e,p))

            if args.do_statistics:
                sentence_num += 1
                if len(d['opinions'])>0:
                    has_opinions_sentence_num += 1
                    if token_offsets[-1][-1] > args.max_len:
                        exceed_max_sentence_len += 1

                tuple_num += len(d['opinions'])
                single_e_tuple_num += cal_single_e(d)
                null_e_tuple_num += cal_null_e(d)

                h,t,e,overlap_h,overlap_t,overlap_e = cal_entity_and_overlap(d)
                h_num,t_num,e_num,overlap_h_num,overlap_t_num,overlap_e_num = h_num+h,t_num+t,e_num+e,overlap_h_num+overlap_h,overlap_t_num+overlap_t,overlap_e_num+overlap_e
                if overlap_h + overlap_t + overlap_e > 0:
                    overlap_tuple_num += cal_overlap_tuple(d)

                discontinuous_tuple_num += cal_discontinuous_tuple(d)
                discontinuous_h,discontinuous_t,discontinuous_e = cal_discontinuous(d)
                discontinuous_h_num,discontinuous_t_num,discontinuous_e_num = discontinuous_h_num+discontinuous_h,discontinuous_t_num+discontinuous_t,discontinuous_e_num+discontinuous_e

            if not is_discard:
                data_post.append(d)

        if args.do_statistics:
            print(f'{"-"*20}')
            print(f'sentence_num:{sentence_num}  has_opinions_sentence_num:{has_opinions_sentence_num} exceed_max_sentence_len:{exceed_max_sentence_len}')
            print(f'tuple_num:{tuple_num}  overlap_tuple_num:{overlap_tuple_num}  discontinuous_tuple_num:{discontinuous_tuple_num}  single_e_tuple_num:{single_e_tuple_num}  null_e_tuple_num:{null_e_tuple_num}')
            print(f'e_num:{e_num}  h_num:{h_num}  t_num:{t_num}')
            print(f'overlap_e_num:{overlap_e_num}  overlap_h_num:{overlap_h_num}  overlap_t_num:{overlap_t_num}')
            print(f'discontinuous_e_num:{discontinuous_e_num}  discontinuous_h_num:{discontinuous_h_num}  discontinuous_t_num:{discontinuous_t_num}')
            print(f'{"-"*20}')

    return data_post


class SADataset(Dataset):
    def __init__(self,tokenizer,data_path,type,args):
        self.data_path,self.tokenizer,self.max_len,self.type,self.args = data_path,tokenizer,args.max_len,type,args
        self.text,self.inputs,self.labels = [],[],[]
        self.dis_inputs,self.reg_inputs = [],[]
        self.sentence_len,self.token_len,self.token_range,self.token_offset = [],[],[],[]

        self.build_main()

    def build_main(self):
        args = self.args
        data = get_data_from_file(self.data_path,self.tokenizer,self.type,self.args)

        dis_inputs = torch.zeros(self.max_len,self.max_len,dtype=torch.long)
        reg_inputs = torch.zeros(self.max_len,self.max_len,dtype=torch.long)

        # dis 0:0 1:1 2-3:2 4-7:3 8-15:4 16-31:5 32:63:6 64-127:7 ...
        dis2idx = np.zeros(((self.max_len + 256)), dtype='int64')
        for i in range(9):
            dis2idx[2**i:] = i+1 # 0-9

        for i in range(self.max_len):
            for j in range(self.max_len):
                if i>=j:
                    dis_inputs[i][j] = dis2idx[i-j]
                    reg_inputs[i][j] = 1
                else:
                    dis_inputs[i][j] = 19 - dis2idx[j-i]
                    # dis_inputs[i][j] = dis2idx[j-i]
                    reg_inputs[i][j] = 2
        


        for i,d_i in tqdm(enumerate(data)):
            text = (d_i['sent_id'], d_i['text'])
            token_range = d_i['token_range']
            token_offset = d_i['token_offset']
            max_len = d_i['sentence_len']

            tokenized_input = self.tokenizer.batch_encode_plus([d_i['text']],return_tensors="pt",max_length=self.max_len+2,truncation='longest_first').data
            # assert tokenized_input['input_ids'].shape[1] == max_len + 2

            label = torch.zeros(max_len,max_len).long()
            

            for o in d_i['opinions']:
                h,t,e,p = o # h,t,e:  span list  exp: [(2, 3), (4, 5)]
                # assert len(e)>0
                # if len(e)==0 and self.type=='train':
                #     continue
                if len(e)==0:  
                    continue

                e_str = 'e-NEU'
                if p.lower() == 'negative': e_str = 'e-NEG'
                elif p.lower() == 'positive': e_str = 'e-POS'
                elif p.lower()!='neutral': print(f'sentiment error {d_i}:{e}')


                label[e[-1][-1],e[0][0]] = label2id[e_str]
                for i in range(len(e)):
                    span = e[i]
                    for start in range(span[0],span[1]):
                        label[start,start + 1] = label2id['e-NW']
                    if i!= len(e)-1:
                        label[e[i][1],e[i+1][0]] = label2id['e-NW']


                for i in range(len(h)):
                    span = h[i]
                    for start in range(span[0],span[1]):
                        label[start,start+1] = label2id['h-NW']
                    if i!= len(h)-1:
                        label[h[i][1],h[i+1][0]] = label2id['h-NW']
                for i in range(len(t)):
                    span = t[i]
                    for start in range(span[0],span[1]):
                        label[start,start+1] = label2id['t-NW']
                    if i!= len(t)-1:
                        label[t[i][1],t[i+1][0]] = label2id['t-NW']

                if len(h)>0:
                    start,end = (e[0][0],h[0][0]) if e[0][0]>h[0][0] else (h[0][0],e[0][0])
                    label[start,end] = label2id['H-S']
                    start,end = (e[-1][-1],h[-1][-1]) if e[-1][-1]>h[-1][-1] else (h[-1][-1],e[-1][-1])
                    label[start,end] = label2id['H-E']
                if len(t)>0:
                    start,end = (e[0][0],t[0][0]) if e[0][0]>t[0][0] else (t[0][0],e[0][0])
                    label[start,end] = label2id['T-S']
                    start,end = (e[-1][-1],t[-1][-1]) if e[-1][-1]>t[-1][-1] else (t[-1][-1],e[-1][-1])
                    label[start,end] = label2id['T-E']


            dis_inputs_to_add,reg_inputs_to_add = dis_inputs.clone(),reg_inputs.clone()
            end_idx = d_i['sentence_len']
            if end_idx < self.max_len:
                dis_inputs_to_add[end_idx:,:] = 19
                dis_inputs_to_add[:,end_idx:] = 19

                reg_inputs_to_add[end_idx:,:] = 0
                reg_inputs_to_add[:,end_idx:] = 0


            self.text.append(text)
            self.inputs.append(tokenized_input)
            self.labels.append(label)
            self.dis_inputs.append(dis_inputs_to_add)
            self.reg_inputs.append(reg_inputs_to_add)
            self.token_len.append(d_i['token_range'][-1][-1]+1)
            self.sentence_len.append(max_len)
            self.token_range.append(token_range)
            self.token_offset.append(token_offset)


    def __getitem__(self, index):
        text = self.text[index]
        source_ids = self.inputs[index]["input_ids"].squeeze()
        source_mask = self.inputs[index]["attention_mask"].squeeze()
        label = self.labels[index]
        sentence_len = self.sentence_len[index]
        token_len = self.token_len[index]
        token_range = self.token_range[index]
        token_offset = self.token_offset[index]

        dis_inputs = self.dis_inputs[index]
        reg_inputs = self.reg_inputs[index]

        return {"text":text, "source_ids": source_ids, "source_mask": source_mask,
                "label": label,
                "reg_inputs":reg_inputs, "dis_inputs": dis_inputs, 'sentence_len':sentence_len,
                'token_len':token_len,'token_range':token_range,'token_offset':token_offset}

    def __len__(self):
        return len(self.inputs)

# modified from torch RNN
def pad_sequence_modified(sequences, max_len, batch_first=False, padding_value=0.0):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor

def collate_fn(data):

    d = {}
    for key in data[0].keys():
        d[key] = []
        for di in data:
            d[key].append(di[key])
    
    if d['sentence_len'] == [24, 6, 160, 5]:
        print()

    # max_len = max(d['token_len']) 
    max_len = max(d['sentence_len'])
    batch_size = len(data)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data
    
    def fill2(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1], :] = x
        return new_data

    max_sen_len = max(d['sentence_len'])
    max_input_id_len = max([max_sen_len-d['sentence_len'][i]+d['token_len'][i] for i in range(len(d['token_len']))]) + 2 

    d['source_ids'] = pad_sequence_modified(d['source_ids'],max_input_id_len,True)
    d['source_mask'] = pad_sequence_modified(d['source_mask'],max_input_id_len,True)
    
    label = torch.full((batch_size,max_len,max_len),-100, dtype=torch.long)
    d['label'] = fill(d['label'],label)

    d['reg_inputs'] = torch.stack(d['reg_inputs'])[:,:max_len,:max_len]
    d['dis_inputs'] = torch.stack(d['dis_inputs'])[:,:max_len,:max_len]

    d['token_range'],d['token_offset'] = [],[]
    for di in data:
        d['token_range'].append(di['token_range'])
        d['token_offset'].append(di['token_offset'])

    d['sentence'] = []
    max_sen_len = max(d['sentence_len'])
    for id_str,sen in d['text']:
        s = sen.strip().split()
        len_ = len(s)
        s += ['[PAD]']*(max_sen_len-len_)
        d['sentence'].append(' '.join(s))

    return d




