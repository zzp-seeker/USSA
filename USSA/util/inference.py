import torch,copy

# from utils.data import label2id,id2label

label2id = {'N':0,'e-NW':1,'h-NW':2,'t-NW':3,'H-S':4,'H-E':5, 'T-S':6,'T-E':7, 'e-NEU':8,'e-POS':9,'e-NEG':10,'H-SE':11,'T-SE':12}
id2label = {v:k for k,v in label2id.items()}

def from_span_to_word_list(grid,l,r,id):
    if l==-1 and r==-1: return False,[]
    if l==r:
        return True,[l]

    res,res_final,end_flag = [],[],False

    def dfs(s):
        nonlocal res,res_final,end_flag
        res.append(s)
        if s==r:
            res_final = res.copy()
            end_flag = True
            return
        canditate_l = (s + 1 + (grid[s][s+1:]==id).nonzero().squeeze(1)).tolist()
        for c in canditate_l:
            if not end_flag:
                dfs(c)
        res.remove(s)

    dfs(l)
    dfs_flag = (len(res_final)!=0)

    if len(res_final)==0:
        end = l
        while end <= r:
            res_final.append(end)
            canditate_l = end + 1 + (grid[end][end+1:]==id).nonzero().squeeze(1) # Tensor (len,)
            if len(canditate_l.shape)>0 and canditate_l.shape[0]>0:
                end = canditate_l[0].item() 
            else:
                end = end + 1 

    return dfs_flag,res_final

def from_e_to_corr_span_list(grid,e_l,e_r,flag=0):
    H_id,E_id = 4,5 # head/tail id
    if flag==1: H_id,E_id = 6,7

    h_l_list = torch.cat(((grid[e_l,:e_l]==H_id).nonzero(),e_l + (grid[e_l:,e_l]==H_id).nonzero())).squeeze(1) # Tensor(len,)
    h_r_list = torch.cat(((grid[e_r,:e_r]==E_id).nonzero(),e_r + (grid[e_r:,e_r]==E_id).nonzero())).squeeze(1)

    if e_l == e_r and h_r_list.shape[0]>0 and h_l_list.shape[0]==0:
        h_l_list = h_r_list.clone()

    h_span_list = [] # [(),(),...] 
    l_used_list,r_used_list = [False] * h_l_list.shape[0],[False] * h_r_list.shape[0]
    for hj in range(h_r_list.shape[0]):
        for hi in range(h_l_list.shape[0]-1,-1,-1): #
            if h_l_list[hi]<=h_r_list[hj] and (not l_used_list[hi] or not r_used_list[hj]):
                l_used_list[hi],r_used_list[hj] = True,True
                h_span_list.append((h_l_list[hi].item(),h_r_list[hj].item()))
    if len(h_span_list)==0:
        h_span_list = [(-1,-1)]

    return h_span_list

def cal_score(exp,e_l,e_r,alpha,beta):
    idx_list = (exp==label2id['e-NW']).nonzero().tolist()
    el_list = [x[0] for x in idx_list]
    er_list = [x[1] for x in idx_list]

    score = 0
    if e_l in el_list:
        score += alpha
    if e_r in er_list:
        score += alpha



    return score

def remove_nested(exp,index_list,label_id,order=True):
    res,connected_flag = [],[]

    for r,l in index_list:
        if order == False:
            l,r = r,l
        if l > r: continue
        append_flag = True
        res_copy = copy.deepcopy(res)
        for pre_r,pre_l in res:
            if not (r==pre_r and l==pre_l):
                if l<=pre_l<=pre_r<=r:
                    if not from_span_to_word_list(exp,l,r,label_id)[0]:
                        append_flag = False
                        pass
                    else:
                        res_copy.remove([pre_r,pre_l])
                        pass
                elif pre_l<=l<=r<=pre_r:
                    if not from_span_to_word_list(exp,pre_l,pre_r,label_id)[0]:
                        res_copy.remove([pre_r,pre_l])
                        pass
                    else:
                        append_flag = False
                        pass
        res = res_copy
        if append_flag: res.append([r,l])
    if order==False:
        res = [[t[1],t[0]] for t in res]
    return res

def getQuadruple(exp,args):
    res = []
    e_index_list = (exp>=label2id['e-NEU']).nonzero().cpu().tolist()
    # e_index_list = remove_nested(exp,e_index_list,label2id['e-NW'])
    e_index_process_overlap_list = []

    if args.dataset in ['eu','ca']:
        for e_r,e_l in e_index_list:
            if e_l > e_r: continue

            length = len(e_index_process_overlap_list)
            append_flag = True
            for j in range(length-1,-1,-1):
                check_r,check_l = e_index_process_overlap_list[j]
                if check_l <= e_l <= check_r:
                    if e_r-e_l>check_r-check_l:
                        score1,score2 = cal_score(exp,e_l,e_r,2.5,1),cal_score(exp,check_l,check_r,2,1)
                    else:
                        score1,score2 = cal_score(exp,e_l,e_r,2,1),cal_score(exp,check_l,check_r,2.5,1)
                    if score1 == score2:
                        break
                    elif score1 < score2:
                        append_flag = False
                        break
                    else:
                        e_index_process_overlap_list.pop(j)
                else:
                    break

            if append_flag:
                e_index_process_overlap_list.append([e_r,e_l])
    else:
        e_index_process_overlap_list = e_index_list

    e_words_list = [] 
    for e_r,e_l in e_index_process_overlap_list:
        if e_l > e_r: continue

        h_span_list = from_e_to_corr_span_list(exp,e_l,e_r,flag=0)
        t_span_list = from_e_to_corr_span_list(exp,e_l,e_r,flag=1)

        # if h_span_list!=[(-1,-1)]:
        #     h_span_list =  remove_nested(exp,h_span_list,label2id['h-NW'],order=False)
        # if t_span_list!=[(-1,-1)]:
        #     t_span_list =  remove_nested(exp,t_span_list,label2id['t-NW'],order=False)

        d = {} # key:h,t,e,p

        # exp: e:[4,5,6]
        e_f,e_words = from_span_to_word_list(exp,e_l,e_r,label2id['e-NW'])
        if len(e_words) == 0:
            continue

        
        d['e'] = from_word_list_to_span_list(e_words)
        d['p'] = id2label[exp[e_r,e_l].item()]

        overlap_flag = [False] * exp.shape[0]
        for l,r in d['e']:
            for i in range(l,r+1):
                overlap_flag[i]=True

        for hi in range(len(h_span_list)):
            for tj in range(len(t_span_list)):
                d['h'] = from_word_list_to_span_list(from_span_to_word_list(exp,h_span_list[hi][0],h_span_list[hi][1],label2id['h-NW'])[1])
                d['t'] = from_word_list_to_span_list(from_span_to_word_list(exp,t_span_list[tj][0],t_span_list[tj][1],label2id['t-NW'])[1])

                if not e_f and len(d['h'])==0 and len(d['t'])==0:
                    continue



                res.append(d.copy())
        


    return res

# words: [3,4,5,7]
# return: [(3,5),(7,7)]
def from_word_list_to_span_list(words):
    if len(words)==0:
        return []
    res = []
    l,r = words[0],words[0]
    i = 1
    while i<len(words):
        if words[i] == r + 1:
            r += 1
        else:
            res.append((l,r))
            if i<len(words):
                l,r = words[i],words[i]
        i += 1
    res.append((l,r))
    return res


# exp
# token_range list:19 [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 9], [10, 10], [11, 11], [12, 12], [13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19]]
# token_offset list:19 [[0, 1], [2, 10], [11, 19], [20, 23], [24, 28], [29, 35], [36, 41], [42, 47], [48, 50], [51, 57], [58, 63], [64, 66], [67, 74], [75, 82], [83, 84], [85, 87], [88, 97], [98, 99], [100, 101]]
# token_span: (l,r)
# return (l,r) 
def from_token_span_idx_to_word_char_idx(token_range,token_offset,token_span):
    token_span_l,token_span_r = token_span
    l,r = -1,-1

    lt,rt = 0,len(token_range)
    while lt<=rt:
        m = (lt+rt)//2
        if token_range[m][0]<=token_span_l<=token_range[m][1]:
            l = m
            break
        elif token_range[m][0]>token_span_l:
            rt = m - 1
        else:
            lt = m + 1

    lt,rt = 0,len(token_range)
    while lt<=rt:
        m = (lt+rt)//2
        if token_range[m][0]<=token_span_r<=token_range[m][1]:
            r = m
            break
        elif token_range[m][0]>token_span_r:
            rt = m - 1
        else:
            lt = m + 1

    return (token_offset[l][0],token_offset[r][1])



# res = getQuadruple(exp)

g = 0

