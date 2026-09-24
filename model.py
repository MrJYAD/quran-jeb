import os, json
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig


class ArabicDecisionModel(nn.Module):
    """jeb - Arabic typed decision model.

    Each option is written into the prompt next to its own [MASK] marker; the model
    scores at that position and softmaxes across the options of that question. The
    answer space is therefore data, not architecture - new schemas need no retraining.
    """

    def __init__(self, base='UBC-NLP/MARBERTv2', head_layers=2, local_dir=None):
        super().__init__()
        if local_dir:
            self.tok = AutoTokenizer.from_pretrained(os.path.join(local_dir, 'tokenizer'))
            cfg = AutoConfig.from_pretrained(os.path.join(local_dir, 'encoder'))
            self.enc = AutoModel.from_config(cfg)
        else:
            self.tok = AutoTokenizer.from_pretrained(base)
            self.enc = AutoModel.from_pretrained(base)
        H = self.enc.config.hidden_size
        layer = nn.TransformerEncoderLayer(d_model=H, nhead=12, dim_feedforward=H * 4,
                                           batch_first=True, activation='gelu', dropout=0.1)
        self.head = nn.TransformerEncoder(layer, num_layers=head_layers)
        self.scorer = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.LayerNorm(H), nn.Linear(H, 1))
        self.mask_id = self.tok.mask_token_id

    def build(self, state, question, max_len=256):
        """Serialize state + question + options, each option preceded by [MASK]."""
        s = ' | '.join(f'{k}: {v}' for k, v in state.items()) if isinstance(state, dict) else str(state)
        crit = question['criteria']
        opts = list(crit.keys()) if isinstance(crit, dict) else list(crit)
        parts = [self.tok.cls_token, s, self.tok.sep_token, question['instructions']]
        for o in opts:
            desc = crit[o] if isinstance(crit, dict) else o
            parts += [self.tok.mask_token, f'{o}: {desc}']
        return ' '.join(parts), opts

    def forward(self, input_ids, attention_mask, mask_positions, option_counts):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.head(h, src_key_padding_mask=~attention_mask.bool())
        return [self.scorer(h[b, mask_positions[b][:option_counts[b]]]).squeeze(-1)
                for b in range(h.size(0))]

    @torch.no_grad()
    def predict(self, state, questions, max_len=256):
        """questions: {id: {type, instructions, criteria}} -> {id: {answer, confidence, probabilities}}"""
        out = {}
        for qid, q in questions.items():
            text, opts = self.build(state, q, max_len)
            enc = self.tok([text], return_tensors='pt', truncation=True, max_length=max_len)
            dev = next(self.parameters()).device
            enc = {k: v.to(dev) for k, v in enc.items()}
            pos = (enc['input_ids'][0] == self.mask_id).nonzero(as_tuple=True)[0][:len(opts)]
            if len(pos) < len(opts):
                out[qid] = {'error': 'options truncated - raise max_len or use fewer options'}
                continue
            logits = self(enc['input_ids'], enc['attention_mask'], [pos], [len(opts)])[0]
            p = F.softmax(logits.float(), -1)
            n, peak = len(opts), float(p.max())
            conf = max(0.0, min(1.0, (n * peak - 1) / (n - 1))) if n > 1 else 1.0
            out[qid] = {
                'answer': opts[int(p.argmax())],
                'confidence': round(conf, 4),
                'probabilities': {o: round(float(v), 4) for o, v in zip(opts, p)},
            }
        return out


def load(path='.', device='cpu'):
    """Load jeb from a local clone or snapshot_download() of IJyad/jeb."""
    from safetensors.torch import load_file
    m = ArabicDecisionModel(local_dir=path)
    m.load_state_dict(load_file(os.path.join(path, 'model.safetensors')))
    return m.to(device).eval()
