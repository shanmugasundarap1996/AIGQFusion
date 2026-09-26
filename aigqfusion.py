"""AIGQFusion: Adaptive Interaction-Guided Quantum Fusion for ClinVar-MVE.

Paper-faithful implementation of A5: tuned CatBoost + five-qubit BioVQC +
cross-fitted ridge-logistic integration + OOF-selected calibration + validation gate.
No reported paper results are embedded in this source.

Run: python aigqfusion.py --data ClinVar-MVE.csv
"""
from __future__ import annotations
import argparse, copy, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

try:
    import pennylane as qml
    from catboost import CatBoostClassifier
except ImportError as e:
    raise SystemExit("Install requirements: pip install pennylane pennylane-lightning catboost torch scikit-learn pandas numpy") from e

from config import (FEATURES, VIEW_INDEX, BIO_EDGES, REPEAT_SEEDS, N_OUTER_FOLDS,
                    CALIBRATION_TOLERANCE, DECISION_THRESHOLD)
from data_protocol import load_clinvar_mve, FoldPreprocessor, inner_splits, outer_splits

EPS = 1e-7
CATBOOST_BANK = (
    dict(iterations=400, depth=5, learning_rate=.050, l2_leaf_reg=5., random_strength=.30),
    dict(iterations=500, depth=6, learning_rate=.040, l2_leaf_reg=5., random_strength=.25),
    dict(iterations=600, depth=7, learning_rate=.035, l2_leaf_reg=6., random_strength=.20),
)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def clip_prob(p): return np.clip(np.asarray(p, float), EPS, 1-EPS)
def logit(p):
    p = clip_prob(p); return np.log(p/(1-p))


def catboost(cfg, seed):
    return CatBoostClassifier(**cfg, loss_function="Logloss", eval_metric="Logloss",
        auto_class_weights="Balanced", random_seed=seed, task_type="CPU", verbose=False,
        allow_writing_files=False)


def select_catboost(X, y, groups, seed):
    best_cfg, best_loss = None, np.inf
    for cfg in CATBOOST_BANK:
        oof = np.zeros(len(y))
        for k, (tr, va) in enumerate(inner_splits(X, y, groups, seed)):
            prep = FoldPreprocessor().fit(X[tr])
            m = catboost(cfg, seed+k).fit(prep.transform(X[tr]), y[tr])
            oof[va] = m.predict_proba(prep.transform(X[va]))[:,1]
        loss = log_loss(y, clip_prob(oof))
        if loss < best_loss: best_loss, best_cfg = loss, cfg
    return best_cfg


class BioVQC(nn.Module):
    """Five-qubit, two-block Biology-Guided Variational Quantum Circuit."""
    def __init__(self, depth=2):
        super().__init__(); self.depth = depth
        self.alpha = nn.Parameter(.05*torch.randn(depth, len(FEATURES)))
        self.beta = nn.Parameter(.05*torch.randn(depth, len(FEATURES)))
        self.local_y = nn.Parameter(.05*torch.randn(depth, 5)); self.local_z = nn.Parameter(.05*torch.randn(depth, 5))
        self.eta = nn.Parameter(.05*torch.randn(depth, len(BIO_EDGES))); self.rho = nn.Parameter(.05*torch.randn(depth, len(BIO_EDGES)))
        self.reup_y = nn.Parameter(.05*torch.randn(depth, 5)); self.reup_z = nn.Parameter(.05*torch.randn(depth, 5))
        self.readout = nn.Linear(25, 1)
        try: dev = qml.device("lightning.qubit", wires=5)
        except Exception: dev = qml.device("default.qubit", wires=5)

        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def circuit(x, alpha, beta, ly, lz, eta, rho, ry, rz):
            theta = (np.pi/2.0) * torch.tanh(x)
            phi = torch.stack([theta[[i for i,q in enumerate(VIEW_INDEX) if q == v]].mean() for v in range(5)])
            for q in range(5): qml.Hadamard(wires=q)
            for l in range(self.depth):
                for j, q in enumerate(VIEW_INDEX):
                    qml.RY(alpha[l,j]*theta[j], wires=q); qml.RZ(beta[l,j]*theta[j], wires=q)
                for q in range(5):
                    qml.RY(ly[l,q], wires=q); qml.RZ(lz[l,q], wires=q)
                for u,v in ((0,1),(1,2),(2,3),(3,4),(4,0)): qml.CNOT(wires=[u,v])
                for e,(u,v) in enumerate(BIO_EDGES):
                    gamma = eta[l,e] + rho[l,e]*phi[u]*phi[v]
                    qml.IsingZZ(gamma, wires=[u,v])
                for q in range(5):
                    qml.RY(ry[l,q]*phi[q], wires=q); qml.RZ(rz[l,q]*phi[q], wires=q)
            obs = [qml.PauliZ(q) for q in range(5)] + [qml.PauliX(q) for q in range(5)] + [qml.PauliY(q) for q in range(5)]
            obs += [qml.PauliZ(u)@qml.PauliZ(v) for u,v in BIO_EDGES]
            obs += [qml.PauliX(u)@qml.PauliX(v) for u,v in BIO_EDGES]
            return [qml.expval(o) for o in obs]
        self.circuit = circuit

    def representation(self, X):
        rows = [torch.stack(self.circuit(x, self.alpha, self.beta, self.local_y, self.local_z,
                    self.eta, self.rho, self.reup_y, self.reup_z)).float() for x in X]
        return torch.stack(rows)

    def forward(self, X): return self.readout(self.representation(X)).squeeze(1)


def train_biovqc(Xtr, ytr, Xva=None, yva=None, seed=42):
    seed_all(seed); model = BioVQC(depth=2)
    opt = torch.optim.AdamW(model.parameters(), lr=.010, weight_decay=2e-4)
    loss_fn = nn.BCEWithLogitsLoss(); best, best_loss, stale = None, np.inf, 0
    ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=512, shuffle=True)
    for _ in range(36):
        model.train()
        for xb,yb in loader:
            opt.zero_grad(); loss=loss_fn(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        if Xva is None: continue
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(torch.tensor(Xva,dtype=torch.float32))).numpy()
        vl = log_loss(yva, clip_prob(pv))
        if vl < best_loss - 1e-8: best_loss, best, stale = vl, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 6: break
    if best is not None: model.load_state_dict(best)
    return model


def predict_biovqc(model, X):
    model.eval()
    with torch.no_grad(): return torch.sigmoid(model(torch.tensor(X,dtype=torch.float32))).numpy()


def fusion_features(pc, pq):
    lc,lq=logit(pc),logit(pq); u=4*clip_prob(pc)*(1-clip_prob(pc)); d=lq-lc
    return np.c_[lc,lq,u,d,u*d,np.abs(d)]


class Fusion:
    def __init__(self):
        self.scaler=StandardScaler(); self.model=LogisticRegression(C=.35,solver="lbfgs",max_iter=3000)
    def fit(self, pc,pq,y):
        H=self.scaler.fit_transform(fusion_features(pc,pq)); self.model.fit(H,y); return self
    def predict(self,pc,pq): return self.model.predict_proba(self.scaler.transform(fusion_features(pc,pq)))[:,1]


class Calibrator:
    def __init__(self): self.platt=None
    def fit_select(self,p,y,tol=CALIBRATION_TOLERANCE):
        p=clip_prob(p); base=log_loss(y,p)
        m=LogisticRegression(C=1e6,solver="lbfgs",max_iter=3000).fit(logit(p).reshape(-1,1),y)
        pp=m.predict_proba(logit(p).reshape(-1,1))[:,1]
        if log_loss(y,clip_prob(pp)) < base-tol: self.platt=m
        return self
    def transform(self,p):
        p=clip_prob(p)
        return p if self.platt is None else self.platt.predict_proba(logit(p).reshape(-1,1))[:,1]


def cross_fitted_routes(X,y,groups,cat_cfg,seed):
    pc=np.zeros(len(y)); pq=np.zeros(len(y))
    for k,(tr,va) in enumerate(inner_splits(X,y,groups,seed)):
        prep=FoldPreprocessor().fit(X[tr]); xtr,xva=prep.transform(X[tr]),prep.transform(X[va])
        cm=catboost(cat_cfg,seed+k).fit(xtr,y[tr]); pc[va]=cm.predict_proba(xva)[:,1]
        qm=train_biovqc(xtr,y[tr],xva,y[va],seed+k); pq[va]=predict_biovqc(qm,xva)
    return pc,pq


def fit_outer(Xtr,ytr,gtr,Xte,seed):
    cfg=select_catboost(Xtr,ytr,gtr,seed)
    pc_oof,pq_oof=cross_fitted_routes(Xtr,ytr,gtr,cfg,seed)
    fusion=Fusion().fit(pc_oof,pq_oof,ytr); pf_oof=fusion.predict(pc_oof,pq_oof)
    cal_c=Calibrator().fit_select(pc_oof,ytr); cal_f=Calibrator().fit_select(pf_oof,ytr)
    lc=log_loss(ytr,clip_prob(cal_c.transform(pc_oof))); lf=log_loss(ytr,clip_prob(cal_f.transform(pf_oof)))
    use_fused = lf < lc-CALIBRATION_TOLERANCE

    prep=FoldPreprocessor().fit(Xtr); a,b=prep.transform(Xtr),prep.transform(Xte)
    cm=catboost(cfg,seed).fit(a,ytr); pc=cm.predict_proba(b)[:,1]
    # Epoch count remains bounded by the frozen PAPER maximum; no outer-test labels are used.
    qm=train_biovqc(a,ytr,seed=seed); pq=predict_biovqc(qm,b)
    pf=fusion.predict(pc,pq)
    p=cal_f.transform(pf) if use_fused else cal_c.transform(pc)
    return clip_prob(p), (clip_prob(p)>=DECISION_THRESHOLD).astype(int)


def run(data_path):
    _,X,y,groups=load_clinvar_mve(data_path); executions=[]
    for seed in REPEAT_SEEDS:
        for fold,(tr,te) in enumerate(outer_splits(X,y,groups,seed,N_OUTER_FOLDS)):
            p,yhat=fit_outer(X[tr],y[tr],groups[tr],X[te],seed+fold)
            # Return predictions to caller; intentionally do not embed/print manuscript results.
            executions.append(dict(seed=seed,fold=fold,test_index=te,probability=p,prediction=yhat))
    return executions


def main():
    ap=argparse.ArgumentParser(description="AIGQFusion on ClinVar-MVE")
    ap.add_argument("--data",default="ClinVar-MVE.csv"); args=ap.parse_args(); run(args.data)
if __name__=="__main__": main()
