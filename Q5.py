import re, pickle, argparse, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn


GAME_RE = re.compile(r'^Game\s+(\d+):', re.IGNORECASE)
MOVE_RE = re.compile(r'^[BW]\[[A-T][0-9]{1,2}\]$')
NUM_RE  = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?%?')

def _to_nums(line: str):
    out = []
    for tok in NUM_RE.findall(line):
        if tok.endswith('%'):
            tok = tok[:-1]
        try:
            out.append(float(tok))
        except:
            pass
    return out

def parse_file_to_games(fpath: Path):
    with fpath.open('r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i = 0
    games = []
    while i < len(lines):
        m = GAME_RE.match(lines[i])
        if not m:
            i += 1
            continue
        gid = int(m.group(1))
        i += 1
        move_idx, rows = 0, []
        while i < len(lines) and not GAME_RE.match(lines[i]):
            mv = lines[i]
            if MOVE_RE.match(mv):
                color = mv[0]
                move_idx += 1
                vecs, j = [], i + 1
                while j < len(lines) and len(vecs) < 5:
                    cand = _to_nums(lines[j])
                    if len(cand) in (1, 3, 9):
                        vecs.append(cand)
                    j += 1
                i = j
                nine  = [v for v in vecs if len(v) == 9]
                ones  = [v for v in vecs if len(v) == 1]
                three = [v for v in vecs if len(v) == 3]
                if len(nine) < 3 or len(ones) < 1 or len(three) < 1:
                    continue
                policy, value, rankp = nine[0], nine[1], nine[2]
                strength = ones[0][0]
                winrate, lead, uncert = three[0]
                if color == 'W':
                    winrate = 1.0 - winrate
                    lead = -lead
                rows.append({
                    'move_idx': move_idx, 'color': color,
                    **{f'policy_{k+1}': policy[k] for k in range(9)},
                    **{f'value_{k+1}':  value[k]  for k in range(9)},
                    **{f'rankp_{k+1}':  rankp[k]  for k in range(9)},
                    'strength': strength, 'winrate': winrate, 'lead': lead, 'uncertainty': uncert,
                })
            else:
                i += 1
        if rows:
            games.append(pd.DataFrame(rows))
    return games


def _entropy_rows(a: np.ndarray) -> np.ndarray:
    a = np.maximum(a, 0.0)
    s = a.sum(axis=1, keepdims=True) + 1e-12
    p = a / s
    return -(p * (np.log(p + 1e-12))).sum(axis=1)

def _topk_mass_rows(arr: np.ndarray, ks=(1,3,5)):
    idx = np.argsort(-arr, axis=1)
    out = {}
    for k in ks:
        pick = np.take_along_axis(arr, idx[:, :k], axis=1).sum(axis=1)
        out[k] = pick
    return out

def _gini_rows(arr: np.ndarray):
    s = arr.sum(axis=1, keepdims=True) + 1e-12
    p = arr / s
    sorted_p = np.sort(p, axis=1)
    n = arr.shape[1]
    coef = (2*np.arange(1, n+1) - n - 1)
    g = (sorted_p * coef).sum(axis=1) / (n * (sorted_p.sum(axis=1) + 1e-12))
    return np.abs(g)

def _spearman_like_corr_rows(a: np.ndarray, b: np.ndarray):
    ra_idx = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float32)
    rb_idx = np.argsort(np.argsort(b, axis=1), axis=1).astype(np.float32)
    ra = (ra_idx - ra_idx.mean(axis=1, keepdims=True)) / (ra_idx.std(axis=1, keepdims=True)+1e-6)
    rb = (rb_idx - rb_idx.mean(axis=1, keepdims=True)) / (rb_idx.std(axis=1, keepdims=True)+1e-6)
    return (ra*rb).mean(axis=1)

MOVE_FEATURE_NAMES = (
    [f'policy_{i}' for i in range(1,10)] +
    [f'value_{i}'  for i in range(1,10)] +
    [f'rankp_{i}'  for i in range(1,10)] +
    ['strength', 'winrate', 'lead', 'uncertainty']
)

def _base_frame_to_step_matrix(df_moves: pd.DataFrame) -> np.ndarray:
    arr_base = df_moves[[c for c in MOVE_FEATURE_NAMES]].to_numpy(dtype=np.float32, copy=False)

    def _max_ent(frame, base):
        a = frame[[f'{base}_{i}' for i in range(1,10)]].to_numpy(dtype=np.float32, copy=False)
        s = a.sum(axis=1, keepdims=True) + 1e-12
        p = a / s
        ent = -(p * np.log(p + 1e-12)).sum(axis=1, keepdims=True)
        mx  = a.max(axis=1, keepdims=True)
        return a, mx, ent

    A_p, pmax, pent = _max_ent(df_moves,'policy')
    A_v, vmax, vent = _max_ent(df_moves,'value')
    A_r, rkmax,rkent = _max_ent(df_moves,'rankp')

    derived6 = np.concatenate([pmax,pent,vmax,vent,rkmax,rkent], axis=1)

    topk_p = _topk_mass_rows(A_p, ks=(1,3,5))
    topk_v = _topk_mass_rows(A_v, ks=(1,3,5))
    topk_r = _topk_mass_rows(A_r, ks=(1,3,5))
    gini_p = _gini_rows(A_p)[:,None]
    gini_v = _gini_rows(A_v)[:,None]
    gini_r = _gini_rows(A_r)[:,None]
    arg_p  = np.argmax(A_p, axis=1)[:,None].astype(np.float32)/8.0
    arg_v  = np.argmax(A_v, axis=1)[:,None].astype(np.float32)/8.0
    arg_r  = np.argmax(A_r, axis=1)[:,None].astype(np.float32)/8.0
    corr_pv = _spearman_like_corr_rows(A_p, A_v)[:,None]
    corr_pr = _spearman_like_corr_rows(A_p, A_r)[:,None]
    corr_vr = _spearman_like_corr_rows(A_v, A_r)[:,None]

    extra15 = np.concatenate([
        topk_p[1][:,None], topk_p[3][:,None], topk_p[5][:,None],
        gini_p, arg_p,
        topk_v[1][:,None], topk_v[3][:,None], topk_v[5][:,None],
        gini_v, arg_v,
        topk_r[1][:,None], topk_r[3][:,None], topk_r[5][:,None],
        gini_r, arg_r
    ], axis=1).astype(np.float32)

    stat37 = np.concatenate([arr_base, derived6], axis=1)
    d1 = np.vstack([np.zeros((1, stat37.shape[1]), dtype=np.float32),
                    np.diff(stat37, axis=0)]).astype(np.float32)

    key_idx = [27,28,29]
    arr_key = arr_base[:, key_idx]
    d2 = np.vstack([np.zeros((2,3),dtype=np.float32),
                    np.diff(arr_key, n=2, axis=0)]).astype(np.float32)

    lead = arr_base[:,29]; winr = arr_base[:,28]
    dlead = np.concatenate([[0.0], np.diff(lead)]).astype(np.float32)[:,None]
    dwind  = np.concatenate([[0.0], np.diff(winr)]).astype(np.float32)[:,None]
    blunder = (dlead < -5.0).astype(np.float32)  # (T,1)

    color_is_black = (df_moves['color'].values == 'B').astype('float32')[:, None]
    pos_norm = (df_moves['move_idx'].values / max(1, df_moves['move_idx'].max())).astype('float32')[:, None]

    step = np.concatenate([
        arr_base, derived6, d1, d2,
        color_is_black, pos_norm,
        extra15, dlead, dwind, blunder,
        corr_pv, corr_pr, corr_vr
    ], axis=1)
    return step.astype(np.float32)

def center_crop_pad(x: np.ndarray, max_len: int) -> np.ndarray:
    t = x.shape[0]
    if t > max_len:
        s = max(0, (t - max_len)//2)
        x = x[s:s+max_len]
        t = max_len
    out = np.zeros((max_len, x.shape[1]), dtype=np.float32)
    take = min(t, max_len)
    out[:take] = x[:take]
    return out


def compute_meta_side_features(df: pd.DataFrame) -> np.ndarray:
    n = df.shape[0]
    def _ent(base):
        cols = [f'{base}_{i}' for i in range(1,10)]
        if not set(cols).issubset(df.columns) or n == 0:
            return 0.0
        arr = df[cols].to_numpy(dtype=np.float32, copy=False)
        return float(_entropy_rows(arr).mean())

    p_ent = _ent('policy'); v_ent = _ent('value'); r_ent = _ent('rankp')
    log_moves = float(np.log1p(n))
    win_std = float(pd.to_numeric(df['winrate'], errors='coerce').astype('float64').std()) if n>0 else 0.0
    lead_abs_mean = float(np.abs(pd.to_numeric(df['lead'], errors='coerce').astype('float64')).mean()) if n>0 else 0.0
    uncert_mean = float(pd.to_numeric(df['uncertainty'], errors='coerce').astype('float64').mean()) if n>0 else 0.0

    if n > 1:
        lead = pd.to_numeric(df['lead'], errors='coerce').astype('float64').values
        dlead = np.diff(lead)
        blunder_rate = float((dlead < -5.0).mean())
        small_mist   = float((dlead < -2.0).mean())
    else:
        blunder_rate, small_mist = 0.0, 0.0

    def _seg_idx(frac_lo, frac_hi):
        lo = int(np.floor(n*frac_lo)); hi = int(np.floor(n*frac_hi))
        lo = max(0, min(lo, n)); hi = max(0, min(hi, n))
        return lo, max(lo, hi)
    def _seg_entropy(base, a, b):
        cols = [f'{base}_{i}' for i in range(1,10)]
        if n == 0 or not set(cols).issubset(df.columns):
            return 0.0
        arr = df.iloc[a:b][cols].to_numpy(dtype=np.float32, copy=False)
        if arr.size == 0:
            return 0.0
        return float(_entropy_rows(arr).mean())

    h0,h1 = _seg_idx(0.00,0.33)
    m0,m1 = _seg_idx(0.33,0.66)
    t0,t1 = _seg_idx(0.66,1.00)

    seg_feats = [
        _seg_entropy('policy', h0,h1),
        _seg_entropy('policy', m0,m1),
        _seg_entropy('policy', t0,t1),
        _seg_entropy('value',  h0,h1),
        _seg_entropy('value',  m0,m1),
        _seg_entropy('value',  t0,t1),
        blunder_rate, small_mist
    ]
    base7 = [p_ent, v_ent, r_ent, log_moves, win_std, lead_abs_mean, uncert_mean]
    return np.array(base7 + seg_feats, dtype=np.float32)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class BiLSTMClf(nn.Module):
    def __init__(self, in_dim, hidden=224, layers=2, dropout=0.2, num_class=9):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, dropout=dropout,
                            bidirectional=True, batch_first=True)
        self.att  = nn.Linear(hidden*2, 1)
        self.drop = nn.Dropout(dropout)
        self.head_ce   = nn.Linear(hidden*2, num_class)
        self.head_ord  = nn.Linear(hidden*2, num_class-1)
    def forward(self, x, mask):
        h = torch.relu(self.proj(x))
        out,_ = self.lstm(h)
        la = self.att(out).squeeze(-1).masked_fill(~mask, -1e9)
        w = torch.softmax(la, dim=1)
        pooled = torch.sum(out*w.unsqueeze(-1), dim=1)
        z = self.drop(pooled)
        return {'ce_logits': self.head_ce(z), 'ord_logits': self.head_ord(z)}

class TinyTransformer(nn.Module):
    def __init__(self, in_dim, d_model=224, nhead=7, layers=2, dim_ff=640, dropout=0.15, num_class=9):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                         dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head_ce   = nn.Linear(d_model, num_class)
        self.head_ord  = nn.Linear(d_model, num_class-1)
    def forward(self, x, mask):
        h = self.pos(self.proj(x))
        enc = self.encoder(h, src_key_padding_mask=~mask)
        m = mask.float().unsqueeze(-1)
        pooled = (enc*m).sum(1)/(m.sum(1)+1e-6)
        z = self.drop(self.norm(pooled))
        return {'ce_logits': self.head_ce(z), 'ord_logits': self.head_ord(z)}

@torch.no_grad()
def coral_logits_to_proba(logits: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(logits)
    B, K1 = s.shape
    K = K1 + 1
    p = torch.zeros(B, K, device=logits.device, dtype=logits.dtype)
    p[:, 0] = 1 - s[:, 0]
    for j in range(1, K-1):
        p[:, j] = s[:, j-1] - s[:, j]
    p[:, K-1] = s[:, K-2]
    return p


def _make_views(step: np.ndarray, seq_len: int, views: int):
    assert views in (1,3,8)
    V, M = [], []
    t = step.shape[0]
    if t <= seq_len:
        v = center_crop_pad(step, seq_len)
        V.append(v); M.append((np.sum(v != 0, axis=1) > 0))
        return V, M

    if views == 1:
        V.append(center_crop_pad(step, seq_len)); M.append(np.ones(seq_len, dtype=bool))
    elif views == 3:
        V.append(step[:seq_len]);               M.append(np.ones(seq_len, dtype=bool))
        V.append(center_crop_pad(step, seq_len));M.append(np.ones(seq_len, dtype=bool))
        V.append(step[-seq_len:]);              M.append(np.ones(seq_len, dtype=bool))
    else:  
        a = step[:seq_len];               V.append(a); M.append(np.ones(seq_len, dtype=bool))
        b = step[-seq_len:];              V.append(b); M.append(np.ones(seq_len, dtype=bool))
        s = max(0, (t-seq_len)//2); c = step[s:s+seq_len]
        V.append(c); M.append(np.ones(seq_len, dtype=bool))
        q1 = max(0, (t-seq_len)//4); d = step[q1:q1+seq_len]
        V.append(d); M.append(np.ones(seq_len, dtype=bool))
        q3 = max(0, int(3*(t-seq_len)/4)); e = step[q3:q3+seq_len]
        V.append(e); M.append(np.ones(seq_len, dtype=bool))
        h2 = max(0, int(0.15*(t-seq_len))); f = step[h2:h2+seq_len]
        V.append(f); M.append(np.ones(seq_len, dtype=bool))
        t2 = max(0, int(0.85*(t-seq_len))); g = step[t2:t2+seq_len]
        V.append(g); M.append(np.ones(seq_len, dtype=bool))
        v = center_crop_pad(step, seq_len)
        V.append(v); M.append(np.ones(seq_len, dtype=bool))
    return V, M

@torch.no_grad()
def seq_game_proba_ready(model: nn.Module, df: pd.DataFrame,
                         mean: np.ndarray, std: np.ndarray, seq_len: int,
                         device: torch.device, use_fp16: bool, views: int) -> np.ndarray:
    step = _base_frame_to_step_matrix(df).astype(np.float32)
    V, M = _make_views(step, seq_len, views)
    X = torch.from_numpy(np.stack(V, 0)).to(device)       
    mask = torch.from_numpy(np.stack(M, 0)).to(device)    

    mean_t = torch.from_numpy(mean).view(1,1,-1).to(device)
    std_t  = torch.from_numpy(std).view(1,1,-1).to(device)
    if use_fp16:
        X = X.half(); mean_t = mean_t.half(); std_t = std_t.half()
    X = (X - mean_t) / (std_t + 1e-6)

    out = model(X, mask)
    p = 0.7*coral_logits_to_proba(out['ord_logits']) + 0.3*torch.softmax(out['ce_logits'], dim=1)
    return p.mean(0).detach().cpu().numpy()

def build_models_from_states(hp, num_class, tf_states, bl_states, device, use_fp16: bool):
    tf_models, bl_models = [], []
    for st in tf_states:
        m = TinyTransformer(in_dim=hp['feat_dim'], d_model=hp['tf']['hidden'], nhead=hp['tf']['heads'],
                            layers=hp['tf']['layers'], dim_ff=hp['tf']['ff'],
                            dropout=hp['tf']['dropout'], num_class=num_class)
        m.load_state_dict(st, strict=True)
        m.to(device).eval()
        if use_fp16 and device.type == 'cuda':
            m.half()
        tf_models.append(m)
    for st in bl_states:
        m = BiLSTMClf(in_dim=hp['feat_dim'], hidden=hp['bl']['hidden'], layers=hp['bl']['layers'],
                      dropout=hp['bl']['dropout'], num_class=num_class)
        m.load_state_dict(st, strict=True)
        m.to(device).eval()
        if use_fp16 and device.type == 'cuda':
            m.half()
        bl_models.append(m)
    return tf_models, bl_models


def _add_stats(feats, prefix, s: pd.Series):
    x = pd.to_numeric(s, errors='coerce').astype('float64')
    x = x[np.isfinite(x)]
    if x.empty:
        for suf in ('mean','std','min','max','med','p25','p75','skew','dmean','dstd'):
            feats[f'{prefix}_{suf}'] = 0.0
        return
    feats[f'{prefix}_mean'] = float(x.mean())
    feats[f'{prefix}_std']  = float(x.std())
    feats[f'{prefix}_min']  = float(x.min())
    feats[f'{prefix}_max']  = float(x.max())
    feats[f'{prefix}_med']  = float(x.median())
    q25, q75 = np.percentile(x.values, [25, 75])
    feats[f'{prefix}_p25']  = float(q25)
    feats[f'{prefix}_p75']  = float(q75)
    feats[f'{prefix}_skew'] = 0.0  
    dx = x.diff().dropna()
    feats[f'{prefix}_dmean'] = float(dx.mean()) if not dx.empty else 0.0
    feats[f'{prefix}_dstd']  = float(dx.std())  if not dx.empty else 0.0

def _topk_entropy_features(df, base, prefix):
    cols = [f'{base}_{i}' for i in range(1,10)]
    if not set(cols).issubset(df.columns) or df.empty:
        return {
            f'{prefix}_{base}_entropy_mean': 0.0,
            f'{prefix}_{base}_argmax_mode': -1,
            f'{prefix}_{base}_max_mean': 0.0,
            f'{prefix}_{base}_top1_mean': 0.0,
            f'{prefix}_{base}_top3_mean': 0.0,
            f'{prefix}_{base}_top5_mean': 0.0,
        }
    arr = df[cols].to_numpy(dtype=np.float64, copy=False)
    ent = _entropy_rows(arr)
    am = np.argmax(arr, axis=1)
    idx = np.argsort(-arr, axis=1)
    top1 = np.take_along_axis(arr, idx[:, :1], axis=1).mean()
    top3 = np.take_along_axis(arr, idx[:, :3], axis=1).mean()
    top5 = np.take_along_axis(arr, idx[:, :5], axis=1).mean()
    return {
        f'{prefix}_{base}_entropy_mean': float(ent.mean()),
        f'{prefix}_{base}_argmax_mode': int(np.bincount(am).argmax()) if am.size else -1,
        f'{prefix}_{base}_max_mean':    float(np.max(arr, axis=1).mean()),
        f'{prefix}_{base}_top1_mean':   float(top1),
        f'{prefix}_{base}_top3_mean':   float(top3),
        f'{prefix}_{base}_top5_mean':   float(top5),
    }

def _aggregate_one_game(df_game: pd.DataFrame) -> pd.Series:
    feats = {}
    base_cols = [f'policy_{i}' for i in range(1,10)] + \
                [f'value_{i}'  for i in range(1,10)] + \
                [f'rankp_{i}'  for i in range(1,10)] + \
                ['strength','winrate','lead','uncertainty']
    for c in base_cols:
        if c in df_game.columns:
            _add_stats(feats, f'all_{c}', df_game[c])

    feats.update(_topk_entropy_features(df_game,'policy','all'))
    feats.update(_topk_entropy_features(df_game,'value','all'))
    feats.update(_topk_entropy_features(df_game,'rankp','all'))

    n = len(df_game)
    if n > 1:
        lead = pd.to_numeric(df_game['lead'], errors='coerce').astype('float64').values
        dlead = np.diff(lead)
        feats['all_blunder_rate']  = float((dlead < -5.0).mean())
        feats['all_smallmist_rate']= float((dlead < -2.0).mean())
    else:
        feats['all_blunder_rate']  = 0.0
        feats['all_smallmist_rate']= 0.0

    feats['all_n_moves'] = int(df_game.shape[0])
    return pd.Series(feats, dtype='float32')

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    test_dir = os.path.join(base_dir, "test_set")
    model_path = os.path.join(base_dir, "out_dir", "model_stackx.pkl")
    out_csv = os.path.join(base_dir, "out_dir", "submission.csv")
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_dir', type=str, default=test_dir)
    ap.add_argument('--model_path', type=str, default=model_path)
    ap.add_argument('--out_csv', type=str, default=out_csv)
    ap.add_argument('--views',      type=int, default=8, choices=[1,3,8], help='多視角數（8=等價原版；1/3 可加速）')
    ap.add_argument('--fp16',       action='store_true', help='GPU 半精度推論')
    args = ap.parse_args()

    payload = pickle.load(open(args.model_path, 'rb'))
    le = payload['label_encoder']
    num_class = payload['num_class']
    mean = payload['norm']['mean']
    std  = payload['norm']['std']
    hp   = payload['seq_hparams']
    tf_states = payload['seq_states']['tf']
    bl_states = payload['seq_states']['bl']
    tab_model = payload['tabular']['model']
    cols_tab  = payload['tabular']['cols']
    meta_scaler = payload['meta']['scaler']
    meta_clf    = payload['meta']['clf']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fp16 = bool(args.fp16 and device.type == 'cuda')


    tf_models, bl_models = build_models_from_states(hp, num_class, tf_states, bl_states, device, use_fp16)

    test_dir = Path(args.test_dir)
    files = sorted(test_dir.glob('*.txt'))
    if not files:
        raise FileNotFoundError(f"資料夾沒有 .txt：{test_dir}")

    ids, ranks = [], []
    for p in files:
        t0 = time.perf_counter()
        games = parse_file_to_games(p)
        if not games:
            ids.append(p.stem); ranks.append(5)
            print(f"[{p.name}] empty -> rank=5  ({time.perf_counter()-t0:.2f}s)")
            continue

        meta_ps = []
        for df in games:

            ps_tf = [seq_game_proba_ready(m, df, mean, std, hp['seq_len'], device, use_fp16, args.views)
                     for m in tf_models]
            p_tf = np.mean(np.vstack(ps_tf), axis=0)

            ps_bl = [seq_game_proba_ready(m, df, mean, std, hp['seq_len'], device, use_fp16, args.views)
                     for m in bl_models]
            p_bl = np.mean(np.vstack(ps_bl), axis=0)


            def game_tab_proba(df_game):
                s = _aggregate_one_game(df_game)
                X = pd.DataFrame([s]).fillna(0.0)
                for c in cols_tab:
                    if c not in X.columns:
                        X[c] = 0.0
                X = X[cols_tab]
                if hasattr(tab_model, "predict_proba"):
                    return np.array(tab_model.predict_proba(X), dtype=np.float32)[0]
                else:
                    logits = tab_model.decision_function(X)
                    e = np.exp(logits - logits.max(axis=1, keepdims=True))
                    P = e / e.sum(axis=1, keepdims=True)
                    return P[0].astype(np.float32)

            p_tb = game_tab_proba(df)


            side = compute_meta_side_features(df) 
            meta_x = np.concatenate([p_tf, p_bl, p_tb, side], axis=0).reshape(1, -1)
            meta_xs = meta_scaler.transform(meta_x)
            mp = meta_clf.predict_proba(meta_xs)[0]
            meta_ps.append(mp)

        P_final = np.mean(np.vstack(meta_ps), axis=0)  
        cls_idx = int(np.argmax(P_final))
        label   = int(le.inverse_transform([cls_idx])[0])
        ids.append(p.stem)
        ranks.append(label)
        print(f"[{p.name}] rank={label}  ({time.perf_counter()-t0:.2f}s)")

    sub = pd.DataFrame({'id': ids, 'rank': ranks})
    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(outp, index=False, encoding='utf-8-sig')
    print(f"[OK] submission saved -> {outp} (rows={len(sub)})")

if __name__ == '__main__':
    main()
